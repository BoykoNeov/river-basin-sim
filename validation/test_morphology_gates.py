"""The M7 morphology gates that are not the bed wave (build step 8, §3).

Three credibility questions the celerity fixture cannot answer, on Warp's CPU
backend:

* **Does the threshold mean anything in a running reach?** Below ``theta_c`` a bed
  must not move *at all*, and just above it must. The kernel-level version of this
  is ``test_below_threshold_the_capacity_is_bit_exact_zero``
  (:mod:`solver.core.test_sediment`); what is added here is the same claim through
  the whole stack -- a real flow, a real limiter, 20 real activations -- because a
  sign or units error would survive the kernel test only to be revealed by the flow's
  own direction, and a threshold that is *nearly* zero is exactly the failure a
  hand-computed capacity check cannot see.
* **What happens when deposition dries a cell?** M7 plan §1.6 chose the rule
  deliberately: there is no rule. See the test.
* **Is the transport law being used inside its range?** Build step 6 measured MPM's
  shear diverging at the wet/dry guard, and left step 8 to decide between a depth
  guard on the law and keeping the gate scenarios clear of that regime. **The
  decision is the second**, and it is asserted here rather than trusted -- see
  ``test_the_gate_scenarios_transport_inside_the_laws_range``.

Two gates §3 lists are deliberately *not* rebuilt here, because they already hold:

* **Sediment mass conservation** landed at build step 6 -- ``SEDIMENT_GATE``,
  ``SedimentLedger`` and their tests in :mod:`solver.core.test_massbalance`.
* **``test_shoreline_lake_at_rest_on_bumpy_bed`` stays green** is true *by
  construction* rather than by re-testing: it is an HLLC fixture, and
  :class:`solver.io.config.Scenario` refuses ``scheme = "hllc_fv"`` together with
  ``[sediment]``, so morphology cannot reach that code path at all. Re-running it
  under M7 would assert a configuration the config layer forbids.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core.channels import arm_channels
from solver.core.grid import H_DRY
from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, SEDIMENT_GATE, MassLedger, SedimentLedger
from solver.core.sediment import SHIELDS_CRITICAL, arm_sediment, shields_from_flow
from solver.core.state import State
from solver.io.config import Inflow
from solver.processes.inflow import InflowInjector
from solver.processes.morphology import MorphologyProcess
from validation.bedwave import BedWave, drive

wp.init()
DEV = "cpu"

# Enough activations for a transporting reach to move a clearly measurable bed, and
# short enough that the pair costs a few seconds. The sub-threshold arm needs an
# explicit end time for a reason worth naming: below `theta_c` the analytical
# celerity is exactly zero, so `BedWave.run_s` (migration / c_b) is infinite and the
# fixture cannot size its own run.
_ACTIVATIONS = 20


def _threshold_run(ratio: float):
    """Drive the reach re-grained to ``ratio * theta_c``; return the fixture and run."""
    fx = BedWave().at_shields(ratio)
    res = drive(fx, end_s=fx.warmup_s + _ACTIVATIONS * fx.interval_s)
    return fx, res


def _achieved_shields(fx: BedWave, res) -> np.ndarray:
    """Shields number along the reach from the flow the run really had.

    Face discharge against cell depth, interior only -- the same pairing
    ``test_the_design_point_is_what_the_solver_actually_delivers`` uses, and the
    reason the *achieved* range is printed rather than the design value: the bump
    crest is locally shallower and reads a higher ``theta`` than the reach mean, and
    a threshold gate that ignored that would be asserting the wrong number.
    """
    return np.asarray(shields_from_flow(res.face_q[1:-1][:-1], res.depth[1:-1], fx.manning, fx.d50))


def test_a_reach_just_under_the_threshold_moves_no_bed_at_all():
    """0.9 ``theta_c``: the bed change is **bit-exact** zero, not merely small.

    ``theta`` is exactly inversely proportional to ``d50`` (the shear carries no grain
    size) and ``d50`` does not enter the hydraulics at all, so this is the same reach,
    the same flow and the same bump as the celerity gate, coarsened to 42.9 mm gravel
    -- one variable. :func:`solver.core.sediment.mpm_capacity` returns a bit-exact
    zero below threshold, which is what makes the assertion sharp enough to catch a
    sign error, a units error, or a velocity-independent term wired in by accident.

    The bump is deliberately **kept** in this arm. Its crest is locally shallower and
    so reads a higher ``theta`` than the reach mean -- measured, 0.921 ``theta_c``
    against a 0.900 design -- and that 8% margin is the point: a fixture that flattened
    the bed to be safe would not be testing the same geometry as its partner.
    """
    fx, res = _threshold_run(0.90)
    theta = _achieved_shields(fx, res)

    print(
        f"\n[under threshold] d50 {1000 * fx.d50:.2f} mm -> design theta "
        f"{fx.shields:.5f} ({fx.shields / SHIELDS_CRITICAL:.3f} x theta_c)\n"
        f"  achieved theta along the reach {theta.min():.5f} .. {theta.max():.5f} "
        f"({theta.min() / SHIELDS_CRITICAL:.3f} .. {theta.max() / SHIELDS_CRITICAL:.3f} x)\n"
        f"  {res.activations} activations: {int((res.dz_cum != 0.0).sum())} cells moved, "
        f"bed change {1000 * res.dz_cum.min():+.6f} .. {1000 * res.dz_cum.max():+.6f} mm\n"
        f"  mass {res.mass_rel_error:.2e}"
    )

    assert res.mass_rel_error < MASS_GATE
    assert res.activations == _ACTIVATIONS
    # Nowhere in the reach crossed the threshold, crest included -- so "no cell moved"
    # is a statement about the law and not about where the flow happened to be.
    assert theta.max() < SHIELDS_CRITICAL
    assert (res.dz_cum == 0.0).all(), "sub-threshold flow moved the bed"


def test_a_reach_just_over_the_threshold_moves_it():
    """1.2 ``theta_c``: the same reach transports, so the pair is a discriminator.

    Without this half, ``test_a_reach_just_under_the_threshold_moves_no_bed_at_all``
    would pass just as well on a solver that had morphology switched off entirely --
    the same non-vacuity trap ``c938981`` fixed for the sediment ledger.
    """
    fx, res = _threshold_run(1.20)
    theta = _achieved_shields(fx, res)
    moved = int((res.dz_cum != 0.0).sum())

    print(
        f"\n[over threshold] d50 {1000 * fx.d50:.2f} mm -> design theta "
        f"{fx.shields:.5f} ({fx.shields / SHIELDS_CRITICAL:.3f} x theta_c)\n"
        f"  achieved theta along the reach {theta.min():.5f} .. {theta.max():.5f} "
        f"({theta.min() / SHIELDS_CRITICAL:.3f} .. {theta.max() / SHIELDS_CRITICAL:.3f} x)\n"
        f"  {res.activations} activations: {moved} of {fx.nx} cells moved, "
        f"bed change {1000 * res.dz_cum.min():+.4f} .. {1000 * res.dz_cum.max():+.4f} mm\n"
        f"  mass {res.mass_rel_error:.2e}"
    )

    assert res.mass_rel_error < MASS_GATE
    # The whole reach is over threshold, so this is the clean partner of the 0.9 arm
    # rather than a run that transports only in the one place it happens to be deep.
    assert theta.min() > SHIELDS_CRITICAL
    # Essentially every free cell moved -- the exact figure is printed rather than
    # asserted, because a cell whose two face integrals happened to cancel would read
    # zero legitimately and a knife-edge equality would go brittle on any change to
    # the reach. What is exact is the pinned ends: they are held at zero by the
    # fixture's own sediment BC and are not evidence either way.
    assert moved >= fx.nx - 4
    assert res.dz_cum[0] == 0.0 and res.dz_cum[-1] == 0.0
    assert np.abs(res.dz_cum).max() > 1e-3


def test_the_gate_scenarios_transport_inside_the_laws_range():
    """M7's regime decision, asserted: **keep the scenarios in channel flow.**

    Build step 6 measured MPM's shear diverging at the wet/dry guard --
    ``tau/rho = g n^2 q^2 / h^(7/3)`` at fixed ``q`` -- so a 1 mm overland sheet at
    0.5 m/s with ``n = 0.03`` and ``d50 = 2 mm`` reads ``theta = 0.68``, **14x
    theta_c**, and the 16x16 rain-on-a-bowl scenario in :mod:`solver.test_run` scours
    5.6 cm in its first 150 s activation. Step 8 had to choose between giving the law
    a depth guard and keeping the *gate* scenarios clear of that regime.

    **It keeps them clear, and does not touch the law.** A relative-submergence guard
    would silently break what it was not aimed at: ``sed_boulders``
    (:mod:`solver.test_run`) asserts a 1 m grain size moves no bed and passes today
    *because* ``theta < theta_c``, so any ``h >= k*d50`` cut-off would keep that test
    green for a second, unrelated reason and it would stop testing the threshold at
    all. The bowl is likewise the only witness that sediment moves for step 5's
    scenario test, step 7's store keystone and the datum comparison. Changing the law
    to fix a scenario-selection problem trades a stated limitation for three silent
    ones.

    So the guard is a **property of the gates**, checked here. The discriminator is
    relative submergence ``h/d50`` rather than an absolute depth, because it is the
    dimensionless group the law is actually about: MPM is a channel bedload law, and
    ``h/d50 = 0.5`` means the "flow" is shallower than a single grain -- not a regime
    any bedload formula was calibrated in, or extrapolates into. The celerity fixture
    and both threshold arms sit two orders above that.
    """
    gate = BedWave()
    arms = {
        "celerity gate": gate,
        "threshold, under": gate.at_shields(0.90),
        "threshold, over": gate.at_shields(1.20),
    }
    print("")
    for name, fx in arms.items():
        print(
            f"[regime] {name:18} h {fx.normal_depth:.4f} m, d50 {1000 * fx.d50:5.2f} mm "
            f"-> h/d50 = {fx.relative_submergence:6.1f}, theta = "
            f"{fx.shields / SHIELDS_CRITICAL:.2f} x theta_c"
        )
    # For contrast, the regime the bowl scenario lives in and this decision fences off.
    sheet = float(shields_from_flow(0.5 * 1e-3, 1e-3, 0.03, 0.002))
    print(
        f"[regime] {'bowl overland sheet':18} h {1e-3:.4f} m, d50 {2.0:5.2f} mm "
        f"-> h/d50 = {1e-3 / 0.002:6.1f}, theta = {sheet / SHIELDS_CRITICAL:.2f} x theta_c"
        "   <- outside the law, and deliberately not a gate scenario"
    )

    for name, fx in arms.items():
        assert fx.relative_submergence > 20.0, f"{name} transports outside MPM's range"
    # The contrast is the whole argument: without it, "> 20" could be vacuous.
    assert 1e-3 / 0.002 < 1.0
    assert sheet > 10.0 * SHIELDS_CRITICAL


# --- the M7 demo's scenario-design claim (build step 9) ------------------------
# `test_the_gate_scenarios_transport_inside_the_laws_range` above checks the *step-8
# fixtures*, analytically, from a steady normal depth. A shipped scenario is a harder
# case and the demo is the first one: it starts dry, it carries a sub-grid channel, and
# it advances a wetting front, so "the reach is in regime" is no longer a single number
# -- some cells always sit at the wet/dry guard. What `scenarios/reach_alluvial.toml`
# claims instead is a *distribution*: essentially all of the bed change happens where
# the law applies. This pair gates that claim, on the same geometry, with the forcing
# as the only variable.

_DEMO_D50 = 0.008  # m, the demo's grain size
_REGIME_FLOOR = 35.0  # h_col/d50; the lowest relative submergence the step-8 gates run at


def _channel_reach(ny: int = 48, nx: int = 9, dx: float = 100.0):
    """A coarse valley with a sub-grid channel down the middle -- the demo in miniature.

    Deliberately at the demo's own cell size and channel width, because the quantity
    under test is the ratio between them: the storage curve stands a channel cell's
    water ``dx/w`` deeper than its cell mean, and reading the regime off the cell mean
    instead would call a 1.6 m river a 0.11 m sheet (which is what it does).
    """
    rows = np.arange(ny, dtype=np.float64)[:, None]
    cols = np.arange(nx, dtype=np.float64)[None, :]
    mid = (nx - 1) / 2.0
    # 0.15% down-valley, banks rising 0.4 m per cell away from the river.
    bed = 20.0 - 0.15 * rows + 0.4 * np.abs(cols - mid)
    w = np.where(np.abs(cols - mid) < 0.5, 8.0, 0.0) * np.ones_like(rows)
    d = np.where(w > 0.0, 1.5, 0.0)
    return bed.astype(np.float32), w.astype(np.float32), d.astype(np.float32), dx


def _regime_share(rain_mm_hr: float, *, activations: int = 12, interval_s: float = 900.0):
    """Drive the reach; return (in-regime share of the interior, interior volume,
    peak interior |dz|, outlet-row volume share).

    The share is volume-weighted on purpose. A *minimum* over cells that moved is the
    wrong statistic for any dry-start run -- the wetting front always puts some cell at
    the guard, so a min-based gate could only ever fail -- while the share answers the
    question a quoted bed change actually rests on: did the metres come from cells the
    law describes?

    **The open-boundary row is excluded, and separately reported.** That is not a
    convenience: a boundary face carries no bedload (never being updated *is* the closed
    sediment BC) while the local-inertial open boundary removes water from the edge
    *cell*, so the water leaves and its load does not, and the cell aggrades. M7 plan §4
    predicts it for this scenario by name and measures 0.053 m per activation on the
    free-ended celerity fixture. Nothing in ``[sediment]`` can pin a cell *down*
    (``alluvium_thickness`` bounds the floor only), so the sanctioned remedy is to keep
    quoted figures clear of the boundary and say which -- which means measuring it, not
    dropping it silently. Here it is 81% of the gross volume in the last two rows, and
    folding that into a regime statistic would conflate a boundary artefact with the
    question of whether the law was used inside its range.
    """
    bed, w, d, dx = _channel_reach()
    ny, nx = bed.shape
    st = State.from_bed(bed, dx=dx, manning=np.full_like(bed, 0.06), device=DEV)
    arm_channels(st, w, d, np.where(w > 0.0, 0.03, 0.06).astype(np.float32))
    arm_sediment(st, np.full(bed.shape, _DEMO_D50, dtype=np.float32), 0.4)
    morph = MorphologyProcess(st, interval_s)
    # Q into the head of the channel, spread over four cells the way the demo spreads
    # it -- the whole discharge into one cell digs a crater that dominates the volume.
    # Sized to stay *within bank*: normal depth at this q is ~0.9 m against a 1.5 m
    # bank, which is the demo's condition. An earlier 20 m^3/s overbanked (h_n ~ 1.5 m)
    # and turned the in-channel arm into a second sheet-flow case -- the in-regime share
    # fell 89.5% -> 51.9% as the run lengthened, because it was measuring the floodplain.
    q_each = 8.0 / 4.0
    mid = nx // 2
    inj = InflowInjector(
        [Inflow(cell=(r, mid), hydrograph=[(0.0, q_each), (1.0e9, q_each)]) for r in range(4)],
        st.grid,
        DEV,
    )
    st.set_open_boundaries({"south": "open"})
    rain_m_s = rain_mm_hr / 1000.0 / 3600.0
    if rain_m_s > 0.0:
        st.arm_source_compensation()

    t, acts = 0.0, 0
    while acts < activations:
        dt = min(compute_dt(st, alpha=0.7, dt_max=20.0), (acts + 1) * interval_s - t)
        inj.apply(st, t, dt)
        step(st, dt, rain=rain_m_s)
        t += dt
        if t >= (acts + 1) * interval_s - 1e-9:
            morph.advance(t, interval_s)
            acts += 1

    dz = np.abs(st.sediment.bed_change_numpy())
    h = st.h.numpy().astype(np.float64)
    # The storage curve, host-side: below bank full a channel cell's water all stands
    # in the channel, dx/w deeper than the cell mean (solver.core.channels).
    has = (w > 0.0) & (d > 0.0)
    w_s = np.where(has, w, 1.0).astype(np.float64)
    h_bf = w_s * d / dx
    h_col = np.where(has, np.where(h <= h_bf, h * dx / w_s, d + (h - h_bf)), h)
    vol = dz * dx * dx
    outlet_share = float(vol[-1].sum()) / float(vol.sum()) if vol.sum() > 0.0 else 0.0
    interior = slice(0, ny - 1)  # everything but the open south edge
    vol_i, sub_i = vol[interior], h_col[interior] / _DEMO_D50
    gross = float(vol_i.sum())
    in_regime = float(vol_i[sub_i >= _REGIME_FLOOR].sum())
    return (
        (in_regime / gross if gross > 0.0 else 0.0),
        gross,
        float(dz[interior].max()),
        outlet_share,
    )


def test_an_inflow_driven_channel_keeps_its_transport_where_the_law_lives():
    """The demo's design claim, and the rain sheet that breaks it -- same geometry.

    ``scenarios/reach_alluvial.toml`` drops rainfall entirely, which looks like a
    stylistic choice and is not: MPM is a channel bedload law and its shear diverges as
    ``h -> H_DRY`` at fixed ``q``, so a millimetric rain-on-grid sheet transports
    furiously in a regime shallower than a single grain. Measured on the shipped
    scenario before the rain was removed: **83% of the run's gross bed change** sat in
    a rain artefact, and an absent ``[rainfall]`` table was silently supplying 50 mm/hr
    (:mod:`solver.io.test_config`).

    So the pair varies only the forcing. Both arms run the same coarse valley, the same
    sub-grid channel, the same grain, the same discharge into the same four head cells;
    one adds ``reach_basin``'s 15 mm/hr storm on top. The claim is not "the in-channel
    arm is in regime" -- that alone would pass by construction on any well-drawn reach
    -- it is that the share **collapses** when the sheet is added, which is what makes
    dropping the rain a design decision rather than a preference.
    """
    dry, dry_gross, dry_peak, dry_outlet = _regime_share(0.0)
    wet, wet_gross, wet_peak, wet_outlet = _regime_share(15.0)
    print("")
    print(
        f"[regime] inflow only   : {100 * dry:5.1f}% of {dry_gross:11.1f} m3 in regime "
        f"(h_col/d50 >= {_REGIME_FLOOR:g}), peak |dz| {1000 * dry_peak:10.2f} mm, "
        f"outlet row {100 * dry_outlet:4.1f}% of gross"
    )
    print(
        f"[regime] + 15 mm/hr    : {100 * wet:5.1f}% of {wet_gross:11.1f} m3 in regime, "
        f"peak |dz| {1000 * wet_peak:10.2f} mm, outlet row {100 * wet_outlet:4.1f}% of gross"
    )
    assert dry_gross > 0.0, "the in-channel arm moved no bed at all -- nothing is gated"
    assert dry > 0.95, f"the demo's own pattern is out of regime ({100 * dry:.1f}%)"
    # The contrast is the whole argument. Without it, "> 0.95" says nothing about
    # whether the choice of forcing was load-bearing -- and the interior share is the
    # honest place to make it, since the boundary artefact is present in both arms.
    assert wet < 0.5 * dry, f"a rain sheet did not move the transport out of regime ({wet:.2f})"
    assert wet_gross > 100.0 * dry_gross, "the sheet should transport far more, out of range"


def test_deposition_can_dry_a_cell_and_the_ordinary_guard_is_what_dries_it():
    """M7 plan §1.6: there is no drying rule, and that is the design.

    The naive fear is that a rising bed eats the water above it. It cannot: ``h`` is
    volume per unit *plan area* and no Exner kernel reads or writes it, so
    ``eta = z + h`` rises **with** the bed and a cell can never be buried. That is
    asserted first, immediately after the activation -- the water is bit-for-bit
    untouched and the surface has risen by exactly the bed change.

    Drying is therefore **hydraulic and later**: the raised cell is now a mound, and
    the ordinary momentum update runs the water off it over the following fast steps
    until the existing ``H_DRY`` guard calls it dry -- the same guard that decides
    what dry means everywhere else in the solver. What must hold through all of it is
    that the water went *somewhere*, which is the real content of the gate and is why
    the mass ledger is carried across the drying.

    The deposition is hand-loaded into the transport integral rather than grown from a
    flow, deliberately: making a real flow deposit 5 cm onto a 2 cm sheet means
    running MPM in exactly the millimetric regime
    ``test_the_gate_scenarios_transport_inside_the_laws_range`` fences off, so the
    scenario would be measuring the artefact instead of the rule. The mechanism under
    test is the bed/water coupling, and the integral is its input.
    """
    nx, dx, depth0 = 5, 10.0, 0.02
    st = State.from_bed(np.zeros((1, nx), dtype=np.float32), dx=dx, depth=depth0,
                        manning=0.03, device=DEV)  # fmt: skip
    sed = arm_sediment(st, 0.002, 0.4)
    morph = MorphologyProcess(st, 10.0)
    ledger = MassLedger.from_state(st)
    sed_ledger = SedimentLedger.from_state(st)

    # dz = -(1/(1-p)) * div(qs_int) / dx, so a face pair of +-0.15 m^2 lifts cell 2 by
    # 5 cm -- more than the 2 cm of water standing on it -- and takes 2.5 cm off each
    # neighbour. Net zero, which the sediment ledger checks independently below.
    qs = np.zeros((1, nx + 1), dtype=np.float32)
    qs[0, 2], qs[0, 3] = 0.15, -0.15
    sed.qs_int_x.assign(qs)

    h_before = st.h.numpy().copy()
    morph.advance(10.0, 10.0)
    z = st.z.numpy()[0]
    h_after = st.h.numpy()
    eta = st.eta.numpy()[0]

    print(
        f"\n[deposition] after the activation:\n"
        f"  bed   {np.round(z, 4)}\n"
        f"  depth {np.round(h_after[0], 4)} (unchanged: {np.array_equal(h_after, h_before)})\n"
        f"  eta   {np.round(eta, 4)}"
    )
    # The bed rose, the water is bit-for-bit untouched, and the surface rose with it.
    assert z[2] == pytest.approx(0.05, abs=1e-6)
    assert np.array_equal(h_after, h_before), "Exner touched the water"
    assert eta == pytest.approx(z + h_after[0], abs=1e-6)
    assert (z <= eta).all(), "the bed was lifted above its own water surface"

    # Now let the water run off the mound. Nothing here is a morphology step: the bed
    # is frozen from this point (no further activation), so what dries the cell is the
    # ordinary local-inertial update and the ordinary H_DRY guard.
    volume0 = float(h_after.sum()) * dx * dx
    steps = 0
    while st.h.numpy()[0, 2] >= H_DRY and steps < 5000:
        step(st, dt=compute_dt(st, alpha=0.7, dt_max=5.0))
        steps += 1
    rec = ledger.record(st, 0.0)
    h_end = st.h.numpy()
    volume1 = float(h_end.sum()) * dx * dx
    sed_rec = sed_ledger.record(st, 10.0)

    print(
        f"  after {steps} fast steps: depth {np.round(h_end[0], 6)}\n"
        f"  cell 2 depth {h_end[0, 2]:.3e} m (H_DRY {H_DRY:.0e})\n"
        f"  water volume {volume0:.9f} -> {volume1:.9f} m3 "
        f"(rel {abs(volume1 - volume0) / volume0:.2e}), ledger {rec.rel_error:.2e}\n"
        f"  sediment balance {sed_rec.rel_error:.2e}, {sed_rec.gross_volume:.4f} m3 moved"
    )

    assert steps < 5000, "the cell never dried"
    assert h_end[0, 2] < H_DRY
    # The whole point: the water left, it did not vanish. Closed box, no sources.
    assert rec.rel_error < MASS_GATE
    assert volume1 == pytest.approx(volume0, rel=1e-6)
    # ... and the bed change balances too, so "deposited" really came from "eroded".
    assert sed_rec.rel_error < SEDIMENT_GATE
    assert sed_rec.gross_volume > 0.0


def test_a_bed_that_dries_a_cell_keeps_its_neighbours_wet():
    """The partner assertion: drying is local, and the water is next door.

    Without it, ``test_deposition_can_dry_a_cell_...`` would pass on a solver that
    dried the *whole* domain, since a uniformly dry box also conserves mass exactly.
    """
    nx, dx, depth0 = 5, 10.0, 0.02
    st = State.from_bed(np.zeros((1, nx), dtype=np.float32), dx=dx, depth=depth0,
                        manning=0.03, device=DEV)  # fmt: skip
    sed = arm_sediment(st, 0.002, 0.4)
    morph = MorphologyProcess(st, 10.0)
    qs = np.zeros((1, nx + 1), dtype=np.float32)
    qs[0, 2], qs[0, 3] = 0.15, -0.15
    sed.qs_int_x.assign(qs)
    morph.advance(10.0, 10.0)

    for _ in range(2000):
        step(st, dt=compute_dt(st, alpha=0.7, dt_max=5.0))
    h = st.h.numpy()[0]
    print(f"\n[drying is local] depth {np.round(h, 6)}  bed {np.round(st.z.numpy()[0], 4)}")

    assert h[2] < H_DRY, "the raised cell should be dry"
    # The two scoured neighbours are where the water went, and they are deeper than
    # they started.
    assert h[1] > depth0 and h[3] > depth0
    assert (h[[0, 1, 3, 4]] >= H_DRY).all(), "drying was not local to the mound"


def test_re_graining_moves_the_shields_number_and_nothing_else():
    """The threshold pair's premise: ``at_shields`` changes ``d50``, not the reach.

    If re-graining moved the hydraulics too, the two arms would be different reaches
    and their contrast would mean nothing. The claim is checked where it can actually
    fail -- on the derived design point and on the constructed state -- rather than by
    running the pair with morphology off and comparing the depth fields. **That
    comparison is vacuous**: unarmed, ``d50`` is not read by any kernel at all, so the
    three arms are literally the same computation and bit-identity is guaranteed
    whatever ``at_shields`` did to the geometry. A test that cannot fail is worse than
    no test, because its name promises otherwise.

    What *is* asserted: the bed, the seeded depth and the roughness the fixture builds
    are bit-identical across the arms; every hydraulic quantity of the design point is
    unchanged; and ``theta`` lands exactly on the ratio that was asked for. The last
    one is the non-vacuous half -- it is what would break if ``theta`` were not exactly
    inversely proportional to ``d50``.
    """
    gate = BedWave()
    ref_state = gate.state(DEV)
    print("")
    for ratio in (0.90, 1.20):
        fx = gate.at_shields(ratio)
        st = fx.state(DEV)
        print(
            f"[re-grained] {ratio:.2f} theta_c: d50 {1000 * gate.d50:.2f} -> "
            f"{1000 * fx.d50:5.2f} mm, theta {gate.shields:.5f} -> {fx.shields:.5f}, "
            f"h_n {fx.normal_depth:.6f} m, q {fx.unit_discharge:.6f} m2/s, "
            f"Fr {fx.froude:.4f}"
        )
        # theta is exactly theta_c * ratio -- the relation the pair is built on.
        assert fx.shields == pytest.approx(ratio * SHIELDS_CRITICAL, rel=1e-12)
        assert fx.relative_submergence == pytest.approx(gate.normal_depth / fx.d50)
        # ... and nothing hydraulic moved, in the derivation or in the built state.
        assert fx.normal_depth == gate.normal_depth
        assert fx.unit_discharge == gate.unit_discharge
        assert fx.froude == gate.froude
        assert fx.manning == gate.manning and fx.slope == gate.slope
        assert np.array_equal(st.z.numpy(), ref_state.z.numpy())
        assert np.array_equal(st.h.numpy(), ref_state.h.numpy())
        assert np.array_equal(st.n.numpy(), ref_state.n.numpy())


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
