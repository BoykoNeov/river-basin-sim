"""Multi-rate scheduler clock algebra (M5) -- pure arithmetic, no GPU state.

The scheduler is deliberately a *clock*, not a driver (M5 plan §1.1), which is what
lets these tests drive it with a stub ``dt_fn`` and assert the properties that
actually matter: a step never crosses a sync point, every sync point is landed on
exactly, slow processes fire at exact multiples of their interval regardless of the
fast timestep, and the whole sequence still matches the inline loop M1-M4 ran (the
bitwise non-regression guard for the refactor).
"""

from __future__ import annotations

import pytest

from solver.scheduler import EPS_T, MultiRateScheduler, SlowProcess


def _const(dt: float):
    return lambda: dt


def _cycle(values: list[float]):
    """A stub dt_fn cycling through ``values`` (stands in for a state-derived dt)."""
    it = iter(values * 10_000)
    return lambda: next(it)


# --- the pre-M5 inline loop, kept as an executable reference ------------------ #
def _reference_ticks(end_time, output_every, events, dt_fn):
    """The M1-M4 `run.py` clamping loop verbatim -> [(t0, dt, t1, is_output), ...].

    Any divergence between this and MultiRateScheduler is a behaviour change in the
    one loop every prior milestone depends on, so it is asserted exactly (float
    equality), not approximately.
    """
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


def test_matches_the_pre_scheduler_inline_loop_exactly():
    """The refactor must not move a single float: same steps, same output ticks."""
    events = [300.0, 137.5, 900.0]  # rain end + hydrograph knots (unsorted, as before)
    sched = MultiRateScheduler(end_time=1200.0, output_every=300.0, events=events)
    got = [(t.t0, t.dt, t.t1, t.is_output) for t in sched.ticks(_cycle([73.3, 41.0, 210.0]))]
    want = _reference_ticks(1200.0, 300.0, events, _cycle([73.3, 41.0, 210.0]))
    assert got == want


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
