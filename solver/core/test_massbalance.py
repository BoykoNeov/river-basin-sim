"""Mass-balance ledger tests (M1/M7) -- the water and sediment gates, on CPU."""

from __future__ import annotations

import json

import numpy as np
import pytest
import warp as wp

from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, SEDIMENT_GATE, MassLedger, SedimentLedger
from solver.core.sediment import arm_sediment
from solver.core.state import State
from solver.processes.morphology import MorphologyProcess, bed_change_bounds

wp.init()

DEV = "cpu"
_EAST_OPEN = {"east": "open", "west": "closed", "north": "closed", "south": "closed"}


def test_closed_no_source_conserves_to_gate():
    """Sloshing water in a closed box with no rain: residual stays under the gate."""
    rng = np.random.default_rng(1)
    bed = rng.uniform(0.0, 3.0, size=(12, 10)).astype(np.float32)
    depth = rng.uniform(1.0, 4.0, size=(12, 10)).astype(np.float32)
    st = State.from_bed(bed, dx=8.0, depth=depth, manning=0.03, device=DEV)
    ledger = MassLedger.from_state(st)

    t = 0.0
    for _ in range(50):
        dt = compute_dt(st, alpha=0.5, dt_max=5.0)
        step(st, dt=dt)
        t += dt
    ledger.record(st, t)

    assert ledger.max_rel_error < MASS_GATE


def test_uniform_rain_balances_to_gate():
    """Rain on a closed basin: inflow_cum matches stored-volume rise to the gate."""
    bed = np.zeros((16, 16), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=0.2, manning=0.03, device=DEV)
    ledger = MassLedger.from_state(st)
    rain = 50.0 / 1000.0 / 3600.0  # 50 mm/hr -> m/s

    t = 0.0
    for _ in range(40):
        dt = compute_dt(st, alpha=0.5, dt_max=5.0)
        step(st, dt=dt, rain=rain)
        ledger.add_rain_step(rain, dt, st.grid.n_cells)
        t += dt
    ledger.record(st, t)

    assert ledger.max_rel_error < MASS_GATE
    # Sanity: volume actually grew by ~inflow.
    assert ledger.series[-1].volume > ledger.v0


def test_infiltration_is_capped_and_banked():
    """Infiltration removes at most the available depth and banks it in loss_cum.

    A single big-rate step over a shallow sheet must drain each cell to exactly 0
    (never negative), and the removed volume must equal the ledger outflow.
    """
    bed = np.zeros((4, 4), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=0.01, device=DEV)  # 1 cm sheet
    st.set_infiltration(np.full((4, 4), 1.0, dtype=np.float32))  # 1 m/s -> huge
    ledger = MassLedger.from_state(st)
    step(st, dt=1.0)  # infil*dt = 1 m >> 0.01 m available -> fully drained
    ledger.record(st, 1.0)

    h = st.h.numpy()
    assert h.min() >= 0.0 and float(h.max()) < 1e-7  # emptied, non-negative
    removed = st.loss_volume(st.grid.cell_area)
    assert removed == pytest.approx(0.01 * st.grid.cell_area * st.grid.n_cells, rel=1e-5)
    assert ledger.series[-1].outflow_cum == pytest.approx(removed, rel=1e-6)


def test_infiltration_uncapped_rate_is_exact():
    """Uncapped infiltration removes exactly rate*time*area (independent of the gate).

    The mass gate is ~0 for any sink by construction (loss_cum mirrors h), so it
    can't check the *rate*. Here the cells never run dry -> the removed volume is
    the pure infiltration rate, catching a mis-scaled kernel.

    Note: use *shallow* water. Banking is exact (loss_cum == the depth h lost), but
    h loses ``rate*dt`` only to float32 precision -- and that is worst when
    ``h >> rate*dt`` (large h -> coarse ULP swallows the tiny decrement). Shallow h
    keeps the ULP fine so the rate reads true."""
    rate = 5.0e-5  # m/s
    st = State.from_bed(np.zeros((5, 5), dtype=np.float32), dx=10.0, depth=0.3, device=DEV)
    st.set_infiltration(np.full((5, 5), rate, dtype=np.float32))
    dt, nsteps = 2.0, 20  # removes 2e-3 m << 0.3 m -> uncapped
    for _ in range(nsteps):
        step(st, dt=dt)
    removed = st.loss_volume(st.grid.cell_area)
    expected = rate * dt * nsteps * st.grid.cell_area * st.grid.n_cells
    assert removed == pytest.approx(expected, rel=1e-3)


def test_rain_and_infiltration_balance_to_gate():
    """Uniform rain into a closed basin with a partial infiltration sink: the
    residual (inflow - infiltration_outflow - dV) stays under the <1e-6 gate.

    The infiltration sink is float64-exact by construction, so the residual here
    is really the float32 rain-accumulation floor -- kept well under the gate by a
    realistic (non-thin) stored volume, the same regime as the M1 rain test."""
    bed = np.zeros((16, 16), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=0.5, device=DEV)
    st.set_infiltration(np.full((16, 16), 5.0 / 1000.0 / 3600.0, dtype=np.float32))  # 5 mm/hr
    ledger = MassLedger.from_state(st)
    rain = 50.0 / 1000.0 / 3600.0  # 50 mm/hr

    t = 0.0
    for _ in range(60):
        dt = compute_dt(st, alpha=0.5, dt_max=5.0)
        step(st, dt=dt, rain=rain)
        ledger.add_rain_step(rain, dt, st.grid.n_cells)
        t += dt
    ledger.record(st, t)

    assert ledger.max_rel_error < MASS_GATE
    assert ledger.series[-1].outflow_cum > 0.0  # the sink actually removed water


def test_rain_field_adds_expected_volume():
    """A spatial rain field adds sum(rate)*area*time and balances to the gate."""
    bed = np.zeros((8, 8), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=0.1, device=DEV)
    # A ramp field so the pattern is non-uniform (per-cell mm/hr -> m/s).
    rate_mm_hr = (np.arange(64, dtype=np.float32).reshape(8, 8) + 1.0) * 2.0
    rain_m_s = (rate_mm_hr / 1000.0 / 3600.0).astype(np.float32)
    st.set_rain_field(rain_m_s)
    ledger = MassLedger.from_state(st)
    rain_sum_m_s = float(rain_m_s.astype(np.float64).sum())

    dt, nsteps = 5.0, 40
    for _ in range(nsteps):
        step(st, dt=dt, rain_scale=1.0)
        ledger.add_inflow(rain_sum_m_s * dt * st.grid.cell_area)
    ledger.record(st, dt * nsteps)
    assert ledger.max_rel_error < MASS_GATE


def test_drain_to_empty_holds_the_gate():
    """A run that drains fully to empty must not trip the gate by denominator collapse.

    This is the M4 §2 hardening case (the EA suite drains domains). Water starts as
    a sheet on a bed sloping toward an open edge -- so it flows out via the flux
    update (leaving a tiny *nonzero* float32 flux-divergence roundoff in the global
    residual) -- and a moderate infiltration sink guarantees the domain empties to
    ``h == 0`` (so ``sum(h) -> 0``). At that point ``abs(inflow)`` and ``abs(v)``
    both collapse toward 0, so *without* the causal peak-volume floor the small
    absolute residual is divided by a near-zero denominator and blows the relative
    error past the gate -- physics fine, denominator collapse. The peak floor keeps
    the denominator at the largest volume the run actually held, so the gate holds.
    """
    ny, nx, dx = 16, 24, 10.0
    xx = np.broadcast_to(np.arange(nx), (ny, nx))
    bed = ((nx - 1 - xx) * dx * 0.02).astype(np.float32)  # 2% slope toward east
    st = State.from_bed(bed, dx=dx, depth=0.5, manning=0.03, device=DEV)
    st.set_open_boundaries(_EAST_OPEN)
    st.set_infiltration(np.full((ny, nx), 2.0e-4, dtype=np.float32))  # empties the pools
    ledger = MassLedger.from_state(st)
    v0 = ledger.v0

    t = 0.0
    for _ in range(4000):
        dt = compute_dt(st, alpha=0.7, dt_max=5.0)
        step(st, dt=dt)
        t += dt
        if float(st.h.numpy().max()) < 1e-9:  # fully drained
            break
    rec = ledger.record(st, t)

    h = st.h.numpy()
    assert np.isfinite(h).all()
    assert float(h.max()) < 1e-6, f"did not drain to empty: h.max()={h.max():.3e}"
    # The domain really emptied: abs(v) collapsed far below the volume it once held.
    assert rec.volume < 1e-9 * v0
    # ...yet the gate holds -- this is the whole point of the peak-volume floor.
    assert ledger.max_rel_error < MASS_GATE, (
        f"drain-to-empty tripped the gate ({ledger.max_rel_error:.2e}); "
        "denominator collapse -- the peak-volume floor is missing or wrong"
    )


def test_kahan_beats_naive_sum():
    """The compensated accumulator recovers precision a naive float sum loses."""
    from solver.core.massbalance import _Kahan

    k = _Kahan()
    naive = 0.0
    for _ in range(1_000_000):
        k.add(1e-8)
        naive += 1e-8
    assert abs(k.total - 0.01) < abs(naive - 0.01) or abs(k.total - 0.01) < 1e-12


# --- The sediment ledger (M7) --------------------------------------------------
# The balance under test is that a domain **closed to bedload** stays closed: every
# metre the bed gains somewhere came from somewhere else in it, plus whatever the
# per-cell bounds refused and banked. Some of the tests below deliberately assert
# something much tighter than SEDIMENT_GATE, because the gate has to survive a
# reach-scale run while these fixtures are small enough to pin the arithmetic.


def _sloping_reach(nx=30, dx=2.0, slope=0.01, depth=0.5, d50=0.002, thickness=None):
    """A wet 1-row reach steep enough to transport, closed at both ends.

    Closed is the point: no water leaves, no bedload can leave, so the sediment
    balance is the pure statement above with no boundary term to disentangle. The
    water slides downhill, the bed moves ~1 mm over the run, and 124 steps is enough
    -- this is a ledger test, not a hydraulics one.
    """
    j = np.arange(nx, dtype=np.float64)
    bed = (2.0 + (nx - 1 - j) * dx * slope).reshape(1, nx).astype(np.float32)
    st = State.from_bed(bed, dx=dx, depth=depth, manning=0.03, device=DEV)
    arm_sediment(st, d50, 0.4)
    lo, hi = (None, None)
    if thickness is not None:
        lo, hi = bed_change_bounds((1, nx), alluvium_thickness=thickness)
    return st, MorphologyProcess(st, 5.0, dz_lo=lo, dz_hi=hi)


def _drive_bed(st, morph, *, end=60.0):
    """Step the reach, activating morphology on its own clock; record every activation."""
    sed = SedimentLedger.from_state(st)
    interval = morph.interval_s
    t, acts = 0.0, 0
    while t < end - 1e-9:
        dt = compute_dt(st, alpha=0.7, dt_max=1.0)
        edge = min((acts + 1) * interval, end)
        if t + dt > edge:  # an interval must be an interval (the scheduler's job, M5)
            dt = edge - t
        step(st, dt=dt)
        t += dt
        if t >= (acts + 1) * interval - 1e-9:
            morph.advance(t, interval)
            acts += 1
            sed.record(st, t)
    assert acts == int(round(end / interval))
    return sed


def test_a_closed_bedload_domain_nets_to_zero_far_inside_the_gate():
    """No bound fires => nothing is banked and the net bed change is float64 round-off.

    This is the strongest form of the invariant and is asserted orders tighter than
    :data:`SEDIMENT_GATE`: with ``dz_unapplied`` identically zero the balance
    collapses to ``sum(dz_cum) == 0``, and the only thing between it and exact is the
    rounding of ``inv_one_minus_p * div / dx`` per cell.
    """
    st, morph = _sloping_reach()
    sed = _drive_bed(st, morph)
    last = sed.series[-1]

    assert (st.sediment.dz_unapplied.numpy() == 0.0).all(), "an unbounded run banked something"
    assert last.banked_volume == 0.0
    assert last.gross_volume > 1e-3, "the bed did not move -- this proves nothing"
    assert sed.max_rel_error < 1e-13, (
        f"a closed bedload domain drifted ({sed.max_rel_error:.2e}); at float64 this is "
        "a conservation bug in the divergence, not round-off"
    )


def test_the_boundary_faces_carry_no_bedload():
    """The standing evidence that the ledger needs no inflow/outflow terms.

    ``accumulate_qs_*`` are launched over *interior* faces only, so the four edge
    face-rows stay exactly zero and ``div(qs_int)`` telescopes to nothing across the
    grid. If a future kernel starts writing them -- a sediment supply BC, an inlet
    hydrograph's load -- this fails first, which is the point: the balance above
    would otherwise silently start blaming the arithmetic.
    """
    st, morph = _sloping_reach()
    _drive_bed(st, morph)

    qs_x = st.sediment.qs_int_x.numpy()
    qs_y = st.sediment.qs_int_y.numpy()
    assert (qs_x[:, 0] == 0.0).all() and (qs_x[:, -1] == 0.0).all()
    assert (qs_y[0, :] == 0.0).all() and (qs_y[-1, :] == 0.0).all()
    # ...and the interior really was carrying something, or the assertion is vacuous.
    # `advance` zeroes the integral, so read the compensation debt it deliberately keeps.
    assert np.abs(st.sediment.qs_comp_x.numpy()[:, 1:-1]).max() > 0.0


def test_a_bound_supplies_the_domain_and_the_ledger_says_how_much():
    """An alluvium floor is a sediment *source*, and the banked term is its size.

    A cell that wanted to erode 1 mm and was held at bedrock still exported that
    sediment across its faces -- the neighbours received it. So the domain gained
    solid from nowhere, ``supplied = -banked``, and the bed's net gain must equal it
    exactly. Silently clamping instead would invent solid mass the way a bare
    ``max(h, 0)`` invents water (M4).
    """
    st, morph = _sloping_reach(thickness=1.0e-5)  # 0.01 mm of alluvium: bedrock, near enough
    sed = _drive_bed(st, morph)
    last = sed.series[-1]

    assert (st.sediment.dz_unapplied.numpy() != 0.0).any(), "no bound fired -- nothing is tested"
    assert last.banked_volume < 0.0, "a floor holds cells *up*, so it supplies (banked < 0)"
    assert last.bed_volume > 0.0
    assert sed.max_rel_error < SEDIMENT_GATE, (
        f"the banking path lost solid volume ({sed.max_rel_error:.2e}) -- a bound "
        "refused something it did not bank"
    )
    # The whole net gain is the floor's supply: nothing else can have produced it.
    assert last.bed_volume == pytest.approx(-last.banked_volume, rel=1e-12)


def test_the_gross_volume_tells_a_balanced_bed_from_a_still_one():
    """Why every record carries ``gross``: the net is zero either way.

    A lake at rest has no shear, so ``theta < theta_c``, MPM returns a bit-exact zero
    and nothing moves. Its residual is zero -- and so is a transporting run's. Only
    the gross displaced volume distinguishes "conserved" from "nothing happened",
    which is why it is reported rather than kept as a private denominator.
    """
    flat = np.full((1, 12), 5.0, dtype=np.float32)
    st = State.from_bed(flat, dx=2.0, depth=0.5, manning=0.03, device=DEV)
    arm_sediment(st, 0.002, 0.4)
    still = _drive_bed(st, MorphologyProcess(st, 5.0), end=20.0)

    assert still.series[-1].gross_volume == 0.0, "still water moved the bed"
    assert still.series[-1].residual == 0.0
    assert still.max_rel_error == 0.0

    moving, morph = _sloping_reach()
    assert _drive_bed(moving, morph).series[-1].gross_volume > 0.0


def _hand_driven(qs_interior, st, morph, sed_ledger, t):
    """Apply one activation from a hand-set transport integral (boundary faces zero)."""
    qs = np.zeros((1, st.grid.nx + 1), dtype=np.float32)
    qs[0, 1:-1] = qs_interior
    st.sediment.qs_int_x.assign(qs)
    morph.advance(t, morph.interval_s)
    return sed_ledger.record(st, t)


def test_a_cell_that_lifts_off_its_floor_keeps_the_balance():
    """Erode into bedrock, bank the refusal, then deposit back off the limit.

    The non-obvious path through ``exner_update``'s bound: the second activation
    stays inside ``[lo, hi]``, so it banks nothing at all, while the *cumulative*
    ``dz_unapplied`` from the first one must still be carried. Getting that wrong --
    zeroing the bank when a cell comes back into range, or re-banking the difference
    -- leaves the balance broken in exactly one direction and only after a reversal.
    """
    nx, dx, floor, p = 3, 2.0, 1.0e-4, 0.4
    st = State.from_bed(np.full((1, nx), 5.0, np.float32), dx=dx, depth=0.5, device=DEV)
    arm_sediment(st, 0.002, p)
    lo, hi = bed_change_bounds((1, nx), alluvium_thickness=floor)
    morph = MorphologyProcess(st, 5.0, dz_lo=lo, dz_hi=hi)
    ledger = SedimentLedger.from_state(st)

    # A single loaded interior face: cell 0 exports, cell 1 receives, cell 2 untouched.
    # `a` is sized so cell 0 wants to erode ~5x deeper than the floor allows.
    a = 5.0 * floor * (1.0 - p) * dx
    first = _hand_driven([a, 0.0], st, morph, ledger, 5.0)
    banked_m = st.sediment.dz_unapplied.numpy().copy()
    assert st.sediment.dz_cum.numpy()[0, 0] == pytest.approx(-floor, rel=1e-6), "floor let go"
    assert banked_m[0, 0] < 0.0 and first.banked_volume < 0.0
    assert first.rel_error < SEDIMENT_GATE

    # Reverse the face: cell 0 now receives and lifts off its floor, inside the bounds.
    second = _hand_driven([-a, 0.0], st, morph, ledger, 10.0)
    assert st.sediment.dz_cum.numpy()[0, 0] > -floor, "the cell never lifted off"
    assert (st.sediment.dz_unapplied.numpy() == banked_m).all(), (
        "an activation that banked nothing changed the bank"
    )
    assert second.banked_volume == pytest.approx(first.banked_volume, rel=1e-12)
    assert ledger.max_rel_error < SEDIMENT_GATE


def test_an_unarmed_state_has_nothing_to_balance():
    """A static bed is not a zero bed change -- it is the absence of the question."""
    st = State.from_bed(np.zeros((3, 4), np.float32), dx=5.0, device=DEV)
    with pytest.raises(ValueError, match="arm_sediment"):
        SedimentLedger.from_state(st)


def test_the_sediment_series_is_serializable_for_the_store():
    """§7.2: a stored run says what its own gauges read, sediment included."""
    st, morph = _sloping_reach()
    attrs = _drive_bed(st, morph).as_attrs()

    assert attrs["sediment_gate"] == SEDIMENT_GATE
    assert attrs["sediment_max_rel_error"] < SEDIMENT_GATE
    series = attrs["sediment_balance_series"]
    assert len(series) == 13  # the t=0 baseline plus one per activation
    assert series[0]["gross_volume"] == 0.0
    assert set(series[0]) == {
        "time", "bed_volume", "banked_volume", "gross_volume", "residual", "rel_error",
    }  # fmt: skip
    json.dumps(attrs)  # what ZarrWriter will do with it
