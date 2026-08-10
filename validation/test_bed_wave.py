"""Bed-wave celerity: the M7 gate, on Warp's CPU backend.

**This is the gate now** (M7 build step 8). It was authored at build step 3 to prove
the fixture in :mod:`validation.bedwave` is *measurable* -- the reach delivers the
flow its design point was derived from, a low bump migrates a countable number of
cells at the celerity the implemented law predicts, and nothing else the run does to
the bed is big enough to be mistaken for that. Step 8 promotes it rather than
tightening it: the +-20% on the shape estimator is what the sizing evidence supports
(a 7% spread across a 32x range of activation intervals), and shrinking it to +-5%
because the design point happens to read 0.993 would be fitting the gate to one run.
The crest fit and the centroid stay **printed, not gated**, for the reason
:mod:`validation.bedwave` gives: they fail differently and their disagreement is
information.

What step 8 adds here is the other half of §3 -- **interval independence**, and the
**morphological-CFL assertion** that fences it from above. The threshold pair, the
deposition-dries-a-cell rule and the MPM regime check live next door in
:mod:`validation.test_morphology_gates`.

Three findings from sizing it, each now asserted rather than remembered:

* the reach lands on its design normal depth to four decimals, so the analytical
  ``c_b`` in the gate's denominator is the celerity of the flow that actually ran;
* a **free** end grows a sediment-boundary artefact large enough to destroy the
  measurement -- a 0.91 m sill at the outlet in 26 activations, backwater lifting the
  reach 30% -- so the fixture pins its end cells with the bound
  :func:`~solver.core.sediment.exner_update` already carries;
* the 900 s activation interval that follows the reservoir would move this bed wave
  3.1 cells per activation, which is why the fixture carries its own -- and step 8
  gates that from *both* sides, because the interval is fenced below as well
  (:mod:`validation.bedwave` constraint (7), M7 plan §4).

**The harness is no longer provisional, and no longer lives here.** A private
``_drive`` was written at build step 3 hand-wiring what
:mod:`solver.processes.morphology` would own at step 5 -- accumulate every fast step,
apply Exner and rebuild the bed at each activation -- and accumulating *after*
:func:`solver.core.local_inertial.step` returned. Step 5 landed the real thing and
this file drove it: the accumulation happens **inside** ``step`` (off
``state.sediment``, after the limiter and before continuity) and the activation is
``MorphologyProcess.advance``. Step 8 needed it from a second gate file, so it is
:func:`validation.bedwave.drive` now, with ``alpha`` and ``dt_max`` exposed because
constraint (7)'s finding is only reproducible by varying them. Every number recorded
below reproduced bit for bit across both moves, which is what the equivalence held
out for:
``test_the_in_step_hook_is_the_same_inputs_as_accumulating_after_the_step``
(:mod:`solver.processes.test_morphology`) asserts the property directly, and this
fixture is the evidence that it transfers to a measured result.

The warm-up needs no enable flag either, and that is a consequence of the same
design: the state is armed **at** the warm-up boundary, where ``z0`` is still the
pristine bed because nothing has moved it, and morphology begins with the next step.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from solver.core.massbalance import MASS_GATE
from solver.core.sediment import morphological_courant, shields_from_flow
from validation.bedwave import BedWave, drive

wp.init()


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
    res = drive(fx, morphology=False, end_s=fx.warmup_s + 3.0 * fx.interval_s)

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
    """**The M7 celerity gate**: the bump crosses ~16 cells at ~``c_b``, and it is the
    *bump*.

    Gated at +-20% on the shape estimator, which is the number the sizing evidence
    supports rather than the number this run happens to hit -- see the module
    docstring on why step 8 promotes this tolerance instead of shrinking it. What it
    must catch is a fixture that has stopped measuring the law: a wave that diffuses
    away, a wave buried under boundary artefacts, a wave that never leaves its cell.
    The celerity is compared against ``c_b`` **at the achieved flow**, and all three
    estimators are printed -- their disagreement is the diffusion and the upstream
    deposition plateau, not noise (:mod:`validation.bedwave`).
    """
    fx = BedWave()
    res = drive(fx)
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
        f"banked {res.banked_m:.4f} m; mass {res.mass_rel_error:.2e}\n"
        f"  morphological Courant: measured peak {res.courant:.3f} vs {fx.courant:.3f} designed"
    )

    assert res.mass_rel_error < MASS_GATE
    assert res.activations == fx.activations
    # The process measures its own morphological Courant number from the flow it
    # really has (solver.core.sediment.celerity_field), so this is a cross-check of
    # that diagnostic against the fixture's independently derived `c_b`: the peak is
    # over every cell, including the bump crest where the bed is locally faster, so
    # it may sit above the design value -- but not by a factor.
    assert 0.5 * fx.courant < res.courant < 2.0 * fx.courant
    assert res.courant < 0.25, "(7) the bed wave crosses too much of a cell per activation"
    # The wave is still a wave, and it is the thing being measured.
    assert 0.5 < mig.amplitude_retained < 1.05
    assert mig.signal_to_background > 5.0
    # ... and it travelled downstream at the celerity of the law that moved it.
    assert mig.xcorr_celerity == pytest.approx(c_flow, rel=0.20)
    assert 0.6 < mig.peak_celerity / c_flow < 1.4
    assert 0.5 < mig.centroid_celerity / c_flow < 1.4


# The interval is fenced on both sides, and step 8 gates both. Above, the bed wave
# must not cross a cell per activation. Below, the limit is not the splitting at all
# -- it is the scheme (validation.bedwave constraint (7), M7 plan §4) -- so the
# independence gate is asserted **over the band the Courant gate admits**, which is
# what is true, rather than over "halve it and compare", which is not.
_INDEPENDENCE_S = 4680.0  # 104 activations at 45 s == 52 at 90 s, exactly


def test_the_final_bed_does_not_depend_on_the_activation_interval():
    """Interval independence (M7 plan §3), over the band the Courant gate admits.

    45 s and 90 s applied to the **same** 4680 s of morphology -- 104 activations
    against 52, chosen so the two runs cover an identical span rather than the 4635 s
    and 4680 s the fixture's own rounding would give them. Matching removes a
    confound but, measured, it is a *small* one: 0.300 mm matched against 0.324 mm
    mismatched, so the 0.16 cells of extra travel account for 0.02 mm and the rest is
    real interval sensitivity. Worth recording, because the first draft of this
    docstring asserted the opposite from arithmetic and the run disagreed.

    This is what §1.3's time-integrated flux buys: the transport carries its own
    elapsed time, so halving the number of applications does not halve the bed
    change. What it does **not** buy is freedom to shrink the interval without limit
    -- see ``test_a_shorter_interval_is_fenced_by_the_scheme_not_by_the_splitting``.
    """
    coarse = replace(BedWave(), interval_s=90.0)
    fine = replace(BedWave(), interval_s=45.0)
    end = fine.warmup_s + _INDEPENDENCE_S

    res_c = drive(coarse, end_s=end)
    res_f = drive(fine, end_s=end)
    dep_c = coarse.departure(res_c.bed)
    dep_f = fine.departure(res_f.bed)
    diff = np.abs(dep_f - dep_c)
    lo, hi = fine.window
    signal = float(np.abs(dep_f).max())

    print(
        f"\n[interval independence] {_INDEPENDENCE_S:g} s of morphology both ways\n"
        f"  90 s: {res_c.activations:3d} activations, Cr {res_c.courant:.3f}, "
        f"xcorr {coarse.measure(res_c.bed).xcorr_cells:.3f} cells\n"
        f"  45 s: {res_f.activations:3d} activations, Cr {res_f.courant:.3f}, "
        f"xcorr {fine.measure(res_f.bed).xcorr_cells:.3f} cells\n"
        f"  max |bed difference| {1000 * diff.max():.4f} mm whole reach, "
        f"{1000 * diff[lo:hi].max():.4f} mm in the window, "
        f"rms {1000 * np.sqrt((diff[lo:hi] ** 2).mean()):.4f} mm\n"
        f"  ... against a {1000 * signal:.4f} mm signal "
        f"({100 * diff.max() / signal:.2f}%)"
    )

    assert res_c.activations * 90.0 == pytest.approx(res_f.activations * 45.0)
    assert res_c.mass_rel_error < MASS_GATE
    assert res_f.mass_rel_error < MASS_GATE
    # Both intervals sit inside the admitted band, or this compares two artefacts.
    assert res_c.courant < 1.0 and res_f.courant < 1.0
    # The gate: the bed the run ends with is the same bed, to a stated fraction of
    # the bed change itself -- not of the bump amplitude, which the run erodes.
    assert diff.max() < 0.05 * signal


def test_a_flat_reach_stays_flat_at_the_fixtures_own_interval():
    """Nothing should happen to a uniform bed, and at 45 s nothing does.

    The bump removed, the flow is uniform, so the transport is uniform, so the
    divergence is zero and **every** millimetre of bed change this run produces is
    spurious. That makes it the sharpest available statement that the celerity gate
    measures the transport law rather than the scheme's own noise: the floor has to be
    small against the 15 mm wave the gate tracks.

    It is also the **lower fence** on the activation interval, and the reason the
    fixture does not simply use a shorter one. Shortening it means more sync-point
    activations, every one of which clamps ``dt`` and hands local-inertial an abrupt
    shorten-then-restore. Measured on exactly this flat reach: +-0.16 mm at 90 s,
    +-0.11 mm at 45 s, **+-8.85 mm at 22.5 s**, **+-29.3 mm at 11.25 s**. Those are
    not gated here, because they are a property of the scheme rather than of this
    fixture and a fix would legitimately change them -- they are recorded, with the
    controls that isolate the mechanism, in M7 plan §4 (*"a clamped step is not a free
    step"*). What is gated is the one the gate depends on.
    """
    fx = replace(BedWave(), bump_amplitude_m=0.0)
    res = drive(fx)
    spurious = float(np.abs(res.dz_cum).max())

    print(
        f"\n[flat reach] {res.activations} activations at {fx.interval_s:g} s: "
        f"spurious bed change {1000 * res.dz_cum.min():+.4f} .. "
        f"{1000 * res.dz_cum.max():+.4f} mm "
        f"({100 * spurious / BedWave().bump_amplitude_m:.2f}% of the 15 mm bump)  "
        f"depth {res.median_depth(fx.interior):.4f} m  mass {res.mass_rel_error:.2e}"
    )

    assert res.mass_rel_error < MASS_GATE
    assert res.median_depth(fx.interior) == pytest.approx(fx.normal_depth, rel=0.01)
    assert spurious < 0.05 * BedWave().bump_amplitude_m


def test_an_interval_that_moves_the_bed_a_cell_per_activation_is_caught():
    """The morphological-CFL assertion (M7 plan §3): the diagnostic fires, loudly.

    The upper fence. At the reservoir's 900 s cadence this bed wave crosses ~3 cells
    per activation, which is a splitting artefact rather than a result -- M5's
    *"54,000 m^3 into one 40 m cell is a 34 m column"*, in morphology. The point of
    gating it is that the run would otherwise finish and hand back a bed that looks
    entirely plausible: mass is conserved to the same 1e-8, the wave is still roughly
    where a wave should be, and nothing else says the splitting has stopped meaning
    anything.

    What is asserted is the **process's own measurement**
    (:func:`solver.core.sediment.celerity_field` over the flow the run really had),
    not the fixture's analytical ``c_b`` -- that one is already checked as arithmetic
    in ``test_the_fixture_is_sized_for_the_law_not_for_convenience``. A diagnostic
    that only agrees with a hand-derived number is not a diagnostic.
    """
    fx = replace(BedWave(), interval_s=900.0)
    res = drive(fx)
    mig = fx.measure(res.bed)
    c_flow = fx.celerity_at(res.median_unit_discharge(fx.interior), res.median_depth(fx.interior))

    print(
        f"\n[over-Courant] {res.activations} activations at {fx.interval_s:g} s "
        f"(design Cr {fx.courant:.2f})\n"
        f"  measured peak Courant {res.courant:.3f}\n"
        f"  xcorr {mig.xcorr_celerity / c_flow:.3f} c_b, crest retained "
        f"{mig.amplitude_retained:.3f}, signal/background {mig.signal_to_background:.1f}\n"
        f"  mass {res.mass_rel_error:.2e} -- still inside the gate, which is the point"
    )

    # The run completes and conserves mass; nothing in the water tells you anything
    # is wrong. Only the morphological Courant number does.
    assert res.mass_rel_error < MASS_GATE
    assert res.courant > 1.0, "the over-Courant configuration was not detected"
    # ... and the fixture's own interval is on the right side of the same fence, so
    # this is a discriminator rather than a number that is always large.
    assert BedWave().courant < 1.0
    # The bump **grew** -- 1.63 of its initial height, where the same wave properly
    # resolved keeps 0.72 and diffusion is the only thing that can happen to it. That
    # is the artefact's signature, and it is worth pinning because the celerity
    # estimator does *not* catch this run: it reads 0.95 c_b here and would sail
    # through the +-20% gate. The two gates are not substitutes for each other.
    assert mig.amplitude_retained > 1.0, "a resolved bed wave can only diffuse"


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
    free = drive(replace(short, pinned_cells=0))
    pinned = drive(short)

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
