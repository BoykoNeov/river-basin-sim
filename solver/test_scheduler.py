"""Multi-rate scheduler clock algebra (M5) -- pure arithmetic, no GPU state.

The scheduler is deliberately a *clock*, not a driver (M5 plan §1.1), which is what
lets these tests drive it with a stub ``dt_fn`` and assert the properties that
actually matter: a step never crosses a sync point, every sync point is landed on
exactly, slow processes fire at exact multiples of their interval regardless of the
fast timestep, and -- since 2026-08-17 -- that the span between two sync points is
filled with **equal** steps rather than full steps plus a remainder.

That last one replaced this file's oldest assertion. See *the tombstone* below.
"""

from __future__ import annotations

import math

import pytest

from solver.scheduler import EPS_T, MultiRateScheduler, SlowProcess


def _const(dt: float):
    return lambda: dt


def _cycle(values: list[float]):
    """A stub dt_fn cycling through ``values`` (stands in for a state-derived dt)."""
    it = iter(values * 10_000)
    return lambda: next(it)


# --- the tombstone ------------------------------------------------------------ #
# `_reference_ticks` is the M1-M4 inline clamping loop, and until 2026-08-17 this
# file asserted that MultiRateScheduler reproduced it float for float -- the
# "pre-M5 runs are bitwise-identical" invariant M4, M5 and M6 all cite.
#
# The scheduler pass (docs/plans/scheduler-equal-steps.md) ended that assertion on
# purpose, because the arithmetic it pinned was a defect: the remainder step this
# reference produces runs at up to one 585th of the steps either side of it, and
# local-inertial answers that with a short-wavelength standing mode that ripples a
# uniform reach by 2.3 m while the mass balance reads 1e-8.
#
# The reference is kept because it documents what changed and lets a reader diff the
# two schedules directly, and because the *event set* it builds is still the live
# one -- `test_the_sync_points_themselves_are_unchanged` below asserts exactly that.
# What it no longer pins is how a span is filled. The four invariants that replace
# it are grouped under "the replacement invariants".


def _reference_ticks(end_time, output_every, events, dt_fn):
    """The M1-M4 `run.py` clamping loop verbatim -> [(t0, dt, t1, is_output), ...]."""
    n_frames = int(round(end_time / output_every)) + 1
    ev = [output_every * k for k in range(1, n_frames)] + list(events) + [end_time]
    out = []
    t = 0.0
    next_output = output_every
    while t < end_time - EPS_T:
        future = [e for e in ev if e > t + EPS_T]
        dt = min(dt_fn(), (min(future) if future else float("inf")) - t)
        t += dt
        is_output = t >= next_output - EPS_T and next_output <= end_time + EPS_T
        if is_output:
            next_output += output_every
        out.append((t - dt, dt, t, is_output))
    return out


# --- the replacement invariants ------------------------------------------------ #
# Four properties that together say everything the retired bitwise assertion said
# which was worth saying, plus the one it could not say (no remainder steps).

_EVENTS = [300.0, 137.5, 900.0]  # rain end + hydrograph knots, unsorted as in run.py
_DTS = [73.3, 41.0, 210.0]


def _sched(**kw):
    kw.setdefault("end_time", 1200.0)
    kw.setdefault("output_every", 300.0)
    kw.setdefault("events", _EVENTS)
    return MultiRateScheduler(**kw)


def test_the_scheduler_never_lengthens_the_step_the_scheme_asked_for():
    """Invariant 1: ``dt <= dt_raw``, every step.

    The scheduler now chooses how *many* steps fill a span, so it does more than
    clamp -- and this is the line it must not cross. Freezing the step count at the
    span start (the rejected reading of "equal steps") violates it by up to 1.59x,
    which is a CFL bug the scheduler invented rather than one the scheme asked for.
    """
    raws: list[float] = []
    it = iter(_DTS * 10_000)

    def dt_fn():
        raws.append(next(it))
        return raws[-1]

    for tick in _sched().ticks(dt_fn):
        assert tick.dt <= raws[-1] + 1e-12, f"step {tick.dt} exceeds dt_raw {raws[-1]}"


def test_no_step_is_a_small_fraction_of_its_neighbour():
    """Invariant 2: the span is filled with equal steps, not a remainder.

    The one the retired assertion could not make, because it asserted the remainder.
    With a constant ``dt_fn`` every step inside a span must be identical; the only
    changes allowed are at a sync point, where the next span's arithmetic starts.
    """
    sched = _sched(end_time=1000.0, output_every=250.0, events=[137.0])
    ticks = list(sched.ticks(_const(97.0)))
    syncs = sorted(set(sched.sync_times))
    for a, b in zip(ticks, ticks[1:], strict=False):
        if any(abs(a.t1 - s) < EPS_T for s in syncs):
            continue  # a span boundary -- a new span may legitimately pick a new dt
        assert a.dt == pytest.approx(b.dt, rel=1e-9), (
            f"dt changed from {a.dt} to {b.dt} inside a span (t={a.t1})"
        )


def test_the_sync_points_themselves_are_unchanged():
    """Invariant 3: frame times, ``n_frames`` and the output tick set do not move.

    These are pure schedule arithmetic that never saw ``dt``, so the pass must not
    have touched them -- asserted against the retired reference, which is why it is
    kept in the file.
    """
    sched = _sched()
    got = [t.t1 for t in sched.ticks(_cycle(_DTS)) if t.is_output]
    want = [t[2] for t in _reference_ticks(1200.0, 300.0, _EVENTS, _cycle(_DTS)) if t[3]]
    assert got == pytest.approx(want)
    assert got == pytest.approx([300.0, 600.0, 900.0, 1200.0])
    assert sched.n_frames == 5
    assert sched.sync_times == [300.0, 600.0, 900.0, 1200.0, *_EVENTS, 1200.0]


def test_the_schedule_is_reproducible():
    """Invariant 4: same inputs, same floats. Determinism (HANDOFF §8/§12) is the
    property that actually mattered; only the sequence moved, once."""
    a = [(t.t0, t.dt, t.t1, t.is_output) for t in _sched().ticks(_cycle(_DTS))]
    b = [(t.t0, t.dt, t.t1, t.is_output) for t in _sched().ticks(_cycle(_DTS))]
    assert a == b


# --- the arithmetic that is load-bearing now ------------------------------------ #


@pytest.mark.parametrize(
    ("span", "dt_raw"),
    [
        (100.0, 100.0),  # exactly one step
        (100.0, 250.0),  # the scheme wants more than the span
        (100.0, 50.0),  # divides exactly
        (100.0, 99.9),  # just under: two steps of 50, not 99.9 + 0.1
        (100.0, 51.0),  # the classic remainder case
        (100.0, 33.4),  # n = 3
        (100.0, 0.0007),  # many steps
    ],
)
def test_a_span_is_filled_with_equal_steps_none_longer_than_the_scheme_asked(span, dt_raw):
    """``n = ceil(span/dt_raw)`` steps of ``span/n``, and the sync point landed on."""
    sched = MultiRateScheduler(end_time=span, output_every=span)
    ticks = list(sched.ticks(_const(dt_raw)))
    assert len(ticks) == max(1, math.ceil(span / dt_raw))
    assert all(t.dt <= dt_raw + 1e-12 for t in ticks)
    assert all(t.dt == pytest.approx(ticks[0].dt, rel=1e-9) for t in ticks)
    assert ticks[-1].t1 == pytest.approx(span)


def test_a_span_of_two_steps_is_the_worst_case_for_requantisation():
    """The residual path back to the defect, bounded and asserted.

    ``dt`` is recomputed from the remaining span every step, so when the scheme's
    ``dt`` drifts down and the step count ticks up, ``dt`` moves by ``1/n`` -- which
    is largest where ``n`` is smallest. A span barely longer than one step is that
    case: the jump can reach 50%, and it must not exceed it.
    """
    span = 100.0
    # dt_raw drifts down just enough to force n: 1 -> 2 -> 3 across the span.
    it = iter([100.0, 60.0, 40.0, 30.0, 25.0] + [25.0] * 100)
    sched = MultiRateScheduler(end_time=span, output_every=span)
    ticks = list(sched.ticks(lambda: next(it)))
    assert ticks[-1].t1 == pytest.approx(span)
    jumps = [abs(b.dt - a.dt) / a.dt for a, b in zip(ticks, ticks[1:], strict=False)]
    assert max(jumps, default=0.0) <= 0.5 + 1e-12, (
        f"re-quantisation jumped by {100 * max(jumps):.1f}%, past the 1/n bound"
    )


def test_steps_never_cross_a_sync_point_and_land_on_every_one():
    sched = MultiRateScheduler(
        end_time=1000.0,
        output_every=250.0,
        events=[137.0, 610.0],
        processes=[SlowProcess("slow", 400.0, lambda t, e: None)],
    )
    ticks = list(sched.ticks(_const(97.0)))
    landed = {round(t.t1, 9) for t in ticks}
    for s in sched.sync_times:
        # No step straddles a sync point ...
        assert not any(t.t0 < s - EPS_T and t.t1 > s + EPS_T for t in ticks), f"crossed {s}"
        # ... and each is the end of some step.
        assert round(s, 9) in landed, f"never landed on {s}"
    assert ticks[-1].t1 == pytest.approx(1000.0)


def test_output_ticks_land_on_the_cadence_and_count_the_frames():
    sched = MultiRateScheduler(end_time=900.0, output_every=300.0)
    ticks = list(sched.ticks(_const(37.0)))
    outs = [t.t1 for t in ticks if t.is_output]
    assert outs == pytest.approx([300.0, 600.0, 900.0])
    assert len(outs) == sched.n_frames - 1  # the t=0 baseline is written by the caller


@pytest.mark.parametrize("dt", [7.0, 60.0, 121.0, 1e9])
def test_slow_activations_are_exact_multiples_regardless_of_the_fast_dt(dt):
    """Determinism (§8/§12): the slow clock is simulated-time driven, so the fast
    sub-stepping cannot move it -- the same activations for any fast timestep."""
    fired: list[tuple[float, float]] = []
    proc = SlowProcess("res", 900.0, lambda t, e: fired.append((t, e)))
    sched = MultiRateScheduler(end_time=3600.0, output_every=1800.0, processes=[proc])
    for tick in sched.ticks(_const(dt)):
        for p, elapsed in tick.due:
            p.advance(tick.t1, elapsed)
    assert [t for t, _ in fired] == pytest.approx([900.0, 1800.0, 2700.0, 3600.0])
    # Each split interval is exactly the cadence, and they tile the run.
    assert [e for _, e in fired] == pytest.approx([900.0] * 4)
    assert sum(e for _, e in fired) == pytest.approx(3600.0)


def test_slow_process_elapsed_covers_a_partial_final_interval():
    """A cadence that does not divide end_time: the last activation is the last
    exact multiple, and no phantom activation is invented at end_time."""
    proc = SlowProcess("res", 400.0, lambda t, e: None)
    sched = MultiRateScheduler(end_time=1000.0, output_every=500.0, processes=[proc])
    fired = [(tick.t1, tick.due[0][1]) for tick in sched.ticks(_const(60.0)) if tick.due]
    assert [t for t, _ in fired] == pytest.approx([400.0, 800.0])
    assert [e for _, e in fired] == pytest.approx([400.0, 400.0])


def test_two_processes_at_different_rates_stay_independent():
    sched = MultiRateScheduler(
        end_time=1200.0,
        output_every=600.0,
        processes=[
            SlowProcess("fast", 300.0, lambda t, e: None),
            SlowProcess("slow", 600.0, lambda t, e: None),
        ],
    )
    by_name: dict[str, list[float]] = {"fast": [], "slow": []}
    for tick in sched.ticks(_const(55.0)):
        for p, _ in tick.due:
            by_name[p.name].append(tick.t1)
    assert by_name["fast"] == pytest.approx([300.0, 600.0, 900.0, 1200.0])
    assert by_name["slow"] == pytest.approx([600.0, 1200.0])


def test_no_slow_processes_means_the_sync_set_is_unchanged():
    """The bitwise-identity argument for pre-M5 runs: without slow processes the
    sync set is exactly the old event list (output times + forcing + end_time)."""
    sched = MultiRateScheduler(end_time=600.0, output_every=200.0, events=[150.0])
    assert sched.sync_times == [200.0, 400.0, 600.0, 150.0, 600.0]


def test_non_positive_timestep_fails_loudly():
    """A zero dt would spin the loop forever; that must be an error, not a hang."""
    sched = MultiRateScheduler(end_time=100.0, output_every=50.0)
    with pytest.raises(RuntimeError, match="non-positive timestep"):
        list(sched.ticks(_const(0.0)))


def test_slow_process_rejects_a_non_positive_interval():
    with pytest.raises(ValueError, match="interval must be > 0"):
        SlowProcess("bad", 0.0, lambda t, e: None)


def test_scheduler_rejects_bad_run_parameters():
    with pytest.raises(ValueError, match="end_time"):
        MultiRateScheduler(end_time=0.0, output_every=1.0)
    with pytest.raises(ValueError, match="output_every"):
        MultiRateScheduler(end_time=1.0, output_every=0.0)


def test_activations_of_a_slow_process():
    p = SlowProcess("p", 250.0, lambda t, e: None)
    assert p.activations(1000.0) == pytest.approx([250.0, 500.0, 750.0, 1000.0])
    assert p.activations(240.0) == []
