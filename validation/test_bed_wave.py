"""Bed-wave celerity fixture (M7 build step 3) on Warp's CPU backend.

What this file proves is that the fixture in :mod:`validation.bedwave` is
**measurable**: the reach delivers the flow its design point was derived from, a low
bump migrates a countable number of cells at the celerity the implemented law
predicts, and nothing else the run does to the bed is big enough to be mistaken for
that. The *gates* -- sediment mass conservation, interval independence, the
morphological-CFL assertion, the tolerances tightened against the real slow process
-- are M7 build step 8 and are deliberately not here; the tolerances below are wide
on purpose, because a step-3 failure should mean "the geometry is wrong", not "the
physics moved by 3%".

Three findings from sizing it, each now asserted rather than remembered:

* the reach lands on its design normal depth to four decimals, so the analytical
  ``c_b`` in the gate's denominator is the celerity of the flow that actually ran;
* a **free** end grows a sediment-boundary artefact large enough to destroy the
  measurement -- a 0.91 m sill at the outlet in 26 activations, backwater lifting the
  reach 30% -- so the fixture pins its end cells with the bound
  :func:`~solver.core.sediment.exner_update` already carries;
* the 900 s activation interval that follows the reservoir would move this bed wave
  3.1 cells per activation, which is why the fixture carries its own.

**The harness here is provisional.** ``_drive`` hand-wires what
``solver/processes/morphology.py`` will own at build step 5: accumulate the transport
integral every fast step, apply Exner and rebuild the bed at each activation. It
accumulates *after* :func:`solver.core.local_inertial.step` returns, which for the
local-inertial scheme is **the same inputs** as the in-``step`` hook step 5 will
install -- ``eta`` is computed once at the top of the step and never rewritten, the
face discharges are final after the limiter, and the post-continuity sinks touch only
``h``. So the numbers recorded here transfer unchanged, and step 5 owes exactly one
assertion that the equivalence still holds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
import warp as wp

from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.sediment import (
    accumulate_qs_x,
    clear_transport_integral,
    exner_update,
    morphological_courant,
    rebuild_bed,
    shields_from_flow,
)
from solver.processes.inflow import InflowInjector
from validation.bedwave import BedWave

wp.init()
DEV = "cpu"

# Inert at this scale (the state-derived step is ~0.46 s here) and kept only so the
# harness cannot wander off if a future fixture is much deeper.
_DT_MAX = 5.0
_EPS = 1e-9  # activation-time comparison slack, in seconds


@dataclass
class Run:
    """What one fixture run leaves behind, host-side."""

    bed: np.ndarray  # (nx,) final bed elevation, float64
    depth: np.ndarray  # (nx,) final water depth
    face_q: np.ndarray  # (nx+1,) final face discharge per unit width
    dz_cum: np.ndarray  # (nx,) cumulative bed change, float64
    banked_m: float  # metres of bed change the bounds refused (for the ledger)
    mass_rel_error: float
    steps: int
    activations: int
    t: float

    def median_depth(self, where: slice) -> float:
        return float(np.median(self.depth[where]))

    def median_unit_discharge(self, where: slice) -> float:
        return float(np.median(self.face_q[1:-1][where]))


def _drive(fx: BedWave, *, morphology: bool = True, end_s: float | None = None) -> Run:
    """Run the fixture: local-inertial water, plus (optionally) the M7 kernels.

    Provisional -- see the module docstring. ``end_s`` stops early (the design-point
    check needs only the warm-up and a few intervals past it); ``morphology=False``
    leaves the bed untouched, and takes no notice of activation boundaries because
    there are none.
    """
    nx = fx.nx
    st = fx.state(DEV)
    inj = InflowInjector(fx.inflows(), st.grid, DEV)
    ledger = MassLedger.from_state(st)

    z0 = wp.array(fx.bed(), dtype=wp.float32, device=DEV)
    d50 = wp.array(np.full((1, nx), fx.d50, np.float32), dtype=wp.float32, device=DEV)
    qs_int = wp.zeros((1, nx + 1), dtype=wp.float32, device=DEV)
    qs_comp = wp.zeros((1, nx + 1), dtype=wp.float32, device=DEV)
    # One row means there are no interior y-faces; the divergence still reads the
    # y integral, so it is allocated (and stays zero) rather than special-cased.
    qs_int_y = wp.zeros((2, nx), dtype=wp.float32, device=DEV)
    dz_cum = wp.zeros((1, nx), dtype=wp.float64, device=DEV)
    dz_unapplied = wp.zeros((1, nx), dtype=wp.float64, device=DEV)
    lo_h, hi_h = fx.bed_bounds()
    dz_lo = wp.array(lo_h, dtype=wp.float32, device=DEV)
    dz_hi = wp.array(hi_h, dtype=wp.float32, device=DEV)

    end = fx.end_time_s if end_s is None else float(end_s)
    t, steps, acts = 0.0, 0, 0
    while t < end - _EPS:
        dt = compute_dt(st, alpha=0.7, dt_max=_DT_MAX)
        # Land exactly on the warm-up boundary and on every activation, so an
        # interval is an interval (the scheduler does this for real runs, M5). With
        # morphology off there is nothing to land on -- and the activation counter
        # never advances either, so keeping the clamp would freeze `dt` at zero and
        # spin here forever the first time `t` passed one interval.
        if morphology:
            edge = (
                fx.warmup_s if t < fx.warmup_s - _EPS else fx.warmup_s + (acts + 1) * fx.interval_s
            )
        else:
            edge = end
        edge = min(edge, end)
        if t + dt > edge:
            dt = edge - t
        assert dt > 0.0, f"harness made no progress at t={t} (edge={edge}, acts={acts})"

        transporting = morphology and t >= fx.warmup_s - _EPS
        ledger.add_inflow(inj.apply(st, t, dt))
        step(st, dt=dt)
        steps += 1
        if transporting:
            # Post-`step` == the step-5 in-`step` hook for LI: post-limiter face
            # discharge, `eta` still the one those fluxes were computed from.
            wp.launch(
                accumulate_qs_x,
                dim=(1, nx - 1),
                inputs=[st.qx, st.eta, st.z, st.n, d50, float(dt), qs_int, qs_comp],
                device=DEV,
            )
        t += dt

        if morphology and t >= fx.warmup_s + (acts + 1) * fx.interval_s - _EPS:
            wp.launch(
                exner_update,
                dim=(1, nx),
                inputs=[
                    qs_int, qs_int_y, dz_lo, dz_hi, float(fx.dx),
                    1.0 / (1.0 - fx.porosity), dz_cum, dz_unapplied,
                ],
                device=DEV,
            )  # fmt: skip
            wp.launch(rebuild_bed, dim=(1, nx), inputs=[z0, dz_cum, st.z], device=DEV)
            clear_transport_integral(qs_int, qs_int_y)
            acts += 1
    ledger.record(st, t)

    return Run(
        bed=st.z.numpy()[0].astype(np.float64),
        depth=st.h.numpy()[0].astype(np.float64),
        face_q=st.qx.numpy()[0].astype(np.float64),
        dz_cum=dz_cum.numpy()[0],
        banked_m=float(dz_unapplied.numpy().sum()),
        mass_rel_error=ledger.max_rel_error,
        steps=steps,
        activations=acts,
        t=t,
    )


def test_the_fixture_is_sized_for_the_law_not_for_convenience():
    """Every constraint the design point was derived under, asserted (arithmetic only).

    These are the numbers that decide whether the gate measures the transport law or
    something else, and they are cheap enough to check that there is no excuse for
    letting a future edit break one silently. Each ``assert`` maps to a numbered
    constraint in :mod:`validation.bedwave`'s docstring.
    """
    fx = BedWave()
    print("\n" + fx.describe())
    lo, hi = fx.window

    assert fx.shields_margin > 3.0, "(1) too close to threshold to measure celerity"
    assert fx.shields < 1.0, "(1) far outside the Shields range MPM was calibrated in"
    assert fx.bump_slenderness < 0.05, "(2) bump too long vs h/S: the surface adjusts"
    assert fx.froude < 0.6, "(3) too close to critical for a scheme without advection"
    assert fx.bump_sigma_cells >= 5.0, "(4) bump under-resolved"
    assert fx.bump_amplitude_m < 0.02 * fx.normal_depth, "(6) amplitude is not linear"
    assert fx.courant < 0.25, "(7) the bed wave crosses too much of a cell per activation"

    # (5) the reach holds everything that happens in it, with the measurement window
    # clear of both ends and wide enough for the whole migration.
    assert lo > fx.pinned_cells + fx.bump_sigma_cells
    assert hi < fx.nx - fx.pinned_cells
    assert fx.bump_cell + fx.migration_cells + 2 * fx.bump_sigma_cells < hi
    assert fx.migration_m > 2.0 * fx.bump_sigma_m, "the shift must beat the diffusion smear"

    # Quasi-steady: the bed wave must be far slower than the water it rides on, or
    # the analytical celerity's fixed-discharge linearisation is meaningless.
    assert fx.celerity < 0.01 * np.sqrt(9.81 * fx.normal_depth)

    # (7), the other half: the reservoir's 900 s cadence would make this a splitting
    # artefact rather than a result -- which is what the pre-run print exists to say.
    assert morphological_courant(fx.celerity, 900.0, fx.dx) > 1.0
    assert fx.activations * fx.interval_s == pytest.approx(fx.morph_time_s)


def test_the_design_point_is_what_the_solver_actually_delivers():
    """The reach really reaches the flow the celerity was derived at.

    The check that would force a redo of the geometry, which is why it is at build
    step 3 and not step 8: ``c_b`` is evaluated at the design ``(q, h)``, so if the
    reach settled anywhere else the gate would compare a measured celerity against
    the wrong denominator -- and a 1.01 ratio against the wrong number is worse than
    a 0.9 against the right one. It also confirms the water-only warm-up is long
    enough, and that with morphology off the bed is bit-for-bit untouched.

    Run a few intervals *past* the warm-up rather than stopping on it: an unarmed run
    has no activations to land on, and the harness must therefore not try to land on
    them -- when it did, ``dt`` clamped to zero at the first boundary and the loop
    spun forever, which no test that stopped exactly at the warm-up could see.
    """
    fx = BedWave()
    res = _drive(fx, morphology=False, end_s=fx.warmup_s + 3.0 * fx.interval_s)

    h = res.median_depth(fx.interior)
    q = res.median_unit_discharge(fx.interior)
    # From the medians, not per cell: `qx` lives on faces and `h` on centres, and the
    # fixture's claim is about the reach's steady state, not about one cell's pairing.
    theta = float(shields_from_flow(q, h, fx.manning, fx.d50))
    c_flow = fx.celerity_at(q, h)

    print(
        f"\n[design point] after {res.t:g} s ({res.steps} steps, morphology off)\n"
        f"  depth    {h:.4f} m vs design {fx.normal_depth:.4f} "
        f"({100 * (h - fx.normal_depth) / fx.normal_depth:+.2f}%)\n"
        f"  q        {q:.4f} m2/s vs design {fx.unit_discharge:.4f}\n"
        f"  theta    {theta:.4f} vs design {fx.shields:.4f}\n"
        f"  c_b      {c_flow:.5e} m/s at the achieved flow vs {fx.celerity:.5e} designed "
        f"({100 * (c_flow - fx.celerity) / fx.celerity:+.2f}%)\n"
        f"  mass     {res.mass_rel_error:.2e}"
    )

    assert res.mass_rel_error < MASS_GATE
    assert h == pytest.approx(fx.normal_depth, rel=0.01)
    assert q == pytest.approx(fx.unit_discharge, rel=0.01)
    assert theta == pytest.approx(fx.shields, rel=0.02)
    assert c_flow == pytest.approx(fx.celerity, rel=0.05)
    # Morphology off: nothing in this file may touch the bed.
    assert (res.bed == fx.bed()[0]).all()
    assert (res.dz_cum == 0.0).all()


def test_the_bump_migrates_at_the_analytical_bed_wave_celerity():
    """The sizing claim: the bump crosses ~16 cells at ~``c_b``, and it is the *bump*.

    Wide tolerances by design (step 8 owns the gate). What this must catch is a
    fixture that cannot be measured: a wave that diffuses away, a wave buried under
    boundary artefacts, a wave that never leaves its cell. The celerity is compared
    against ``c_b`` **at the achieved flow**, and all three estimators are printed --
    their disagreement is the diffusion and the upstream deposition plateau, not
    noise (:mod:`validation.bedwave`).
    """
    fx = BedWave()
    res = _drive(fx)
    mig = fx.measure(res.bed)
    c_flow = fx.celerity_at(res.median_unit_discharge(fx.interior), res.median_depth(fx.interior))

    print(
        f"\n[bed wave] {res.steps} steps, {res.activations} activations, "
        f"{mig.elapsed_s:g} s of morphology\n"
        f"  c_b analytic {c_flow:.5e} m/s (achieved flow) / {fx.celerity:.5e} (design)\n"
        f"  xcorr    {mig.xcorr_cells:6.3f} cells -> {mig.xcorr_celerity:.5e} m/s "
        f"({mig.xcorr_celerity / c_flow:.3f} x)   <- gated\n"
        f"  crest    {mig.peak_cells:6.3f} cells -> {mig.peak_celerity:.5e} m/s "
        f"({mig.peak_celerity / c_flow:.3f} x)\n"
        f"  centroid {mig.centroid_cells:6.3f} cells -> {mig.centroid_celerity:.5e} m/s "
        f"({mig.centroid_celerity / c_flow:.3f} x)\n"
        f"  crest height {1000 * mig.bump_peak_m:.2f} mm of "
        f"{1000 * mig.initial_amplitude_m:.2f} initial "
        f"(retained {mig.amplitude_retained:.2f}); background outside the window "
        f"{1000 * mig.background_m:.2f} mm (signal/background {mig.signal_to_background:.1f})\n"
        f"  bed change {1000 * res.dz_cum.min():+.2f} .. {1000 * res.dz_cum.max():+.2f} mm; "
        f"banked {res.banked_m:.4f} m; mass {res.mass_rel_error:.2e}"
    )

    assert res.mass_rel_error < MASS_GATE
    assert res.activations == fx.activations
    # The wave is still a wave, and it is the thing being measured.
    assert 0.5 < mig.amplitude_retained < 1.05
    assert mig.signal_to_background > 5.0
    # ... and it travelled downstream at the celerity of the law that moved it.
    assert mig.xcorr_celerity == pytest.approx(c_flow, rel=0.20)
    assert 0.6 < mig.peak_celerity / c_flow < 1.4
    assert 0.5 < mig.centroid_celerity / c_flow < 1.4


def test_a_free_end_grows_a_sill_which_is_why_the_fixture_pins_them():
    """The sediment BC, measured on both sides rather than asserted in a comment.

    A boundary face carries no bedload (it is never updated -- that *is* the closed
    BC), so the inlet cell exports with no supply and the outlet cell imports with no
    export. Free-ended, the outlet sill's backwater ruins the reach's flow before the
    bump has moved; pinned, the same two cells are an equilibrium feed and an
    equilibrium sink and the reach holds its design depth. Nothing is clamped away:
    what the bound refuses is banked in metres for the sediment ledger (step 6).

    This is also the fixture's forward requirement on build step 5: ``morphology.py``
    must take explicit per-cell bounds, because ``alluvium_thickness = 0`` pins the
    floor and leaves the ceiling open -- it cannot hold an outlet down.
    """
    short = replace(BedWave(), migration_cells=4.0)  # ~26 activations; enough to be loud
    free = _drive(replace(short, pinned_cells=0))
    pinned = _drive(short)

    for name, res in (("free", free), ("pinned", pinned)):
        print(
            f"\n[{name} ends] {res.activations} activations: "
            f"inlet dz={res.dz_cum[0]:+.4f} m  outlet dz={res.dz_cum[-1]:+.4f} m  "
            f"reach depth={res.median_depth(short.interior):.4f} m "
            f"(design {short.normal_depth:.4f})  banked={res.banked_m:+.4f} m  "
            f"mass={res.mass_rel_error:.2e}"
        )

    assert free.mass_rel_error < MASS_GATE
    assert pinned.mass_rel_error < MASS_GATE

    # Free: the two ends run away, and the outlet's backwater takes the reach with it.
    assert free.dz_cum[0] < -0.5, "the inlet cell should scour with no supply"
    assert free.dz_cum[-1] > 0.5, "the outlet cell should aggrade with no export"
    assert free.median_depth(short.interior) > 1.1 * short.normal_depth
    assert free.banked_m == 0.0  # nothing was refused, so nothing is owed

    # Pinned: exactly zero at the ends, the reach untouched, and the refused metres
    # are banked rather than lost.
    assert pinned.dz_cum[0] == 0.0
    assert pinned.dz_cum[-1] == 0.0
    assert pinned.median_depth(short.interior) == pytest.approx(short.normal_depth, rel=0.01)
    assert pinned.banked_m > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
