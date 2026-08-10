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

from solver.core.grid import H_DRY
from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, SEDIMENT_GATE, MassLedger, SedimentLedger
from solver.core.sediment import SHIELDS_CRITICAL, arm_sediment, shields_from_flow
from solver.core.state import State
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
    # Every cell except the two pinned ends, which are held at zero by the fixture's
    # own sediment BC and are therefore not evidence either way.
    assert moved == fx.nx - 2 * fx.pinned_cells
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


def test_the_celerity_fixture_and_the_threshold_pair_agree_on_the_flow():
    """Re-graining changes ``theta`` and nothing else -- the pair's premise, checked.

    ``at_shields`` claims ``d50`` moves the Shields number without touching the
    hydraulics. If that were false the threshold pair would be comparing two different
    reaches and its contrast would mean nothing, so it is measured rather than argued:
    all three arms must settle at the same depth and the same unit discharge.
    """
    gate = BedWave()
    end = gate.warmup_s + 3.0 * gate.interval_s
    ref = drive(gate, morphology=False, end_s=end)
    h_ref = ref.median_depth(gate.interior)
    q_ref = ref.median_unit_discharge(gate.interior)

    print("")
    for ratio in (0.90, 1.20):
        fx = gate.at_shields(ratio)
        res = drive(fx, morphology=False, end_s=end)
        h = res.median_depth(fx.interior)
        q = res.median_unit_discharge(fx.interior)
        print(
            f"[same flow] {ratio:.2f} theta_c (d50 {1000 * fx.d50:5.2f} mm): "
            f"depth {h:.6f} m vs {h_ref:.6f}, q {q:.6f} vs {q_ref:.6f}"
        )
        # Bit-exact, in fact: d50 is not read by any hydraulic kernel. Asserted as an
        # equality so a future change that *did* couple them cannot hide in a tolerance.
        assert np.array_equal(res.depth, ref.depth)
        assert np.array_equal(res.face_q, ref.face_q)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
