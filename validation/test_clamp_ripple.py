"""The sync-point clamp gate (``docs/plans/scheduler-equal-steps.md``).

The defect this exists to catch has no other symptom. Until 2026-08-17
:class:`solver.scheduler.MultiRateScheduler` filled the span to each sync point with
full steps **plus a remainder**, and that remainder could be a tiny fraction of the
step either side of it -- one 585th, measured. Local-inertial answers an abrupt
shorten-then-restore with a short-wavelength standing mode in the depth field, so a
reach that should be uniform rippled by **2.3 metres** at an 11.25 s cadence while
the **mass balance read 1e-8**. Mass was conserved perfectly; only the water's
position was wrong, and there was no gate in the repository that could tell.

The scheduler takes equal steps now, so these tests pass -- but they are written to
fail if it ever stops, which is the only reason the numbers below are worth keeping.
Run this file against a scheduler with the remainder restored and five of its six
tests go red.

**Water only. Sediment is never armed.** This is not a morphology bug -- it predates
M5, since M1-M4 ran the same clamping algebra inline -- and arming sediment here
would only let a reader think it was.

Why the gate is written on **curvature** and not on the obvious depth spread
---------------------------------------------------------------------------

``max - min`` over a window cannot tell an oscillation from a slope. This reach has a
legitimate monotone drawdown toward its open toe (-0.33 mm at cell 220 falling to
-0.42 mm at cell 239) and a decaying alternating train in the first few cells where
the inflow point source adjusts. Both feed the spread statistic, so the "clean"
number depends on how far in the window is trimmed -- cells [36, 204) read 0.239 mm
against [20, 220)'s 0.302 mm -- and choosing the trim that gives the nicest answer is
calibration, not validation.

The clamp's failure mode is specifically **short-wavelength**: it alternates cell to
cell. Backwater does not, and a decaying train does so over a handful of cells. So
the discriminating quantity is the second difference of depth along the reach, which
is near zero for any smooth profile at *any* window. Measured on the untrimmed
window, at the worst cadence, it separates the two by more than three orders of
magnitude -- and no window had to be chosen to get that.

A sign-change *count* was tried and rejected: float32 noise alternates everywhere, so
it reads 0.5-0.9 of its maximum even on a clean run.

Where the bounds come from
--------------------------

Swept over ten run lengths from 1800 s to 5835 s and six cadences, ``max|d2h|`` on the
untrimmed interior (mm):

======================  =====================  =======================
cadence                 clamped (the defect)   equal steps (the fix)
======================  =====================  =======================
none (control)          0.013 .. 0.018         0.009 .. 0.017
900 s                   0.013 .. 1.893         0.010 .. 0.022
300 s                   0.857 .. 8.198         0.011 .. 0.053
**45 s**                **12.417 .. 25.584**   0.011 .. 0.016
22.5 s                  56.967 .. 237.962      0.012 .. 2.079
**11.25 s**             **67.439 .. 4369.768** 0.012 .. 0.921
======================  =====================  =======================

:data:`CURVATURE_MM` is 5.0: above every reading the fix produced anywhere in that
sweep (worst 2.079, a 2.4x margin) and below every reading the defect produced at the
two gated cadences (worst case 12.417, a 2.5x margin). The two gated cadences are the
ones where the separation is unambiguous over the whole sweep; 300 s and 900 s are
*intermittently* bad under the clamp (0.013 to 1.893 at 900 s) and would make a flaky
gate, which is itself worth recording -- the shipped 900 s cadence was never clean,
it was only usually clean.

The fix's own residual is intermittent for the same reason and has a named mechanism:
re-quantization, when the state-derived step drifts across a boundary and the span's
step count ticks by one. That jump is bounded by ``1/n``, so it is largest where ``n``
is smallest. :mod:`solver.test_scheduler` asserts that bound directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.processes.inflow import InflowInjector
from solver.scheduler import MultiRateScheduler
from validation.bedwave import DT_MAX, BedWave

# --- the gate's constants, all derived in the module docstring ------------------

CURVATURE_MM = 5.0
"""Bound on ``max|d2h|`` over the interior, in mm. See *Where the bounds come from*."""

DT_JUMP = 0.01
"""Bound on the **mean** step-to-step relative change in ``dt`` over the steady tail.

The mechanism gate: it fails first, and for the right reason, if a remainder step is
ever reintroduced. A remainder is one huge jump *per span*, so the clamp's mean lands
at 29-44 %; equal steps produce at most one small re-quantization per span and the
mean is ~1e-6. See :meth:`Ripple.mean_dt_jump` for why the mean and not the max.
"""

# `MASS_GATE` is imported from solver.core.massbalance rather than restated here, so
# it cannot drift from the project's. It is asserted below and is **explicitly not the
# point**: it reads 1e-8 on both sides of this defect, which is exactly why the defect
# survived four milestones.

END_S = 2400.0
"""1200 s of water-only warm-up (the fixture reaches normal depth by ~1000 s) plus
1200 s of measurement. Long enough that the clamp is unambiguous at both gated
cadences over the whole sweep, short enough that three runs cost ~2 s on CPU."""

CADENCES = (45.0, 11.25)
"""Sync cadences gated. See the docstring on why 900 s and 300 s are not."""


def uniform_reach() -> BedWave:
    """The bed-wave fixture with its bump removed: a reach that should stay flat.

    Reusing :class:`~validation.bedwave.BedWave` rather than building a new fixture
    is deliberate -- it is already a derived, documented design point (mild slope,
    subcritical, one row so there are no y-faces), and the artefact was first found
    on it while sizing M7's interval-independence gate.
    """
    return replace(BedWave(), bump_amplitude_m=0.0)


@dataclass(frozen=True)
class Ripple:
    """What one scheduled run leaves behind, host-side."""

    depth: np.ndarray  # (nx,) final depth, float64
    dts: np.ndarray  # the dt actually stepped, in order
    mass_rel_error: float
    steps: int
    interior: slice

    @property
    def curvature_mm(self) -> float:
        """``max|d2h|`` along the reach, in mm -- the gated statistic."""
        return 1000.0 * float(np.abs(np.diff(self.depth[self.interior], 2)).max())

    @property
    def spread_mm(self) -> float:
        """``max - min`` interior depth, in mm. Printed, never asserted (docstring)."""
        seg = self.depth[self.interior]
        return 1000.0 * float(seg.max() - seg.min())

    def dt_jumps(self, after_s: float) -> np.ndarray:
        """Relative step-to-step changes in ``dt`` past ``after_s``.

        The warm-up is excluded: while the reach fills, the state-derived ``dt``
        genuinely drifts, and that smooth drift is not the artefact -- controls in
        the M7 plan established that only *abrupt* changes excite the mode.
        """
        tail = self.dts[np.cumsum(self.dts) > after_s]
        if tail.size < 2:
            return np.zeros(0)
        return np.abs(np.diff(tail)) / tail[:-1]

    def mean_dt_jump(self, after_s: float) -> float:
        """**The gated one.** Mean relative step-to-step change in ``dt``.

        The mean rather than the max, because the two differ in what they are
        sensitive to. A remainder step is a *per-span* event, so under the clamp
        every span contributes one huge jump and the mean lands at 29-44 %. Equal
        steps produce at most one small re-quantization per span and usually none,
        so the mean is ~1e-6 -- a separation of five orders of magnitude that does
        not depend on catching the single worst step in the run.
        """
        j = self.dt_jumps(after_s)
        return float(j.mean()) if j.size else 0.0

    def max_dt_jump(self, after_s: float) -> float:
        """Largest relative step-to-step change in ``dt``. Printed, not gated.

        Bounded by ``1/n`` (see :meth:`solver.scheduler.MultiRateScheduler.ticks`),
        and ``n`` falls to 2 in the last steps of a span, so the honest worst case
        for a correct implementation is ~50 % on one step per span. Gating on it
        would mean either a bound so loose it no longer separates or a flaky test.
        """
        j = self.dt_jumps(after_s)
        return float(j.max()) if j.size else 0.0

    def describe(self, label: str) -> str:
        return (
            f"[clamp-ripple] {label}: {self.steps} steps, "
            f"max|d2h| = {self.curvature_mm:.4f} mm, spread = {self.spread_mm:.4f} mm, "
            f"dt {self.dts.min():.6f}..{self.dts.max():.6f} s "
            f"(jump mean {100 * self.mean_dt_jump(2000.0):.5f}% "
            f"max {100 * self.max_dt_jump(2000.0):.2f}%), "
            f"mass {self.mass_rel_error:.2e}"
        )


def drive_scheduled(fx: BedWave, cadence: float | None, end_s: float = END_S) -> Ripple:
    """Run the uniform reach under the **real** scheduler at a given sync cadence.

    ``cadence`` becomes ``output_every``, which is what puts sync points on the
    clock; ``None`` means "no sync point but the end", the control. Driving
    :meth:`~solver.scheduler.MultiRateScheduler.ticks` rather than a local copy of
    its arithmetic is the whole point -- the clamping algebra *is* the thing under
    test, and a hand-rolled reimplementation would keep passing after the real one
    regressed.

    No slow processes and no forcing events: the inflow hydrograph is constant, so
    the sync set is the output cadence and ``end_time``, and the cadence is the only
    variable.
    """
    st = fx.state("cpu")
    inj = InflowInjector(fx.inflows(), st.grid, "cpu")
    ledger = MassLedger.from_state(st)
    sched = MultiRateScheduler(
        end_time=end_s,
        output_every=end_s if cadence is None else cadence,
    )

    dts: list[float] = []
    for tick in sched.ticks(lambda: compute_dt(st, alpha=0.7, dt_max=DT_MAX)):
        ledger.add_inflow(inj.apply(st, tick.t0, tick.dt))
        step(st, dt=tick.dt)
        dts.append(tick.dt)
    ledger.record(st, end_s)

    return Ripple(
        depth=st.h.numpy()[0].astype(np.float64),
        dts=np.asarray(dts, dtype=np.float64),
        mass_rel_error=ledger.max_rel_error,
        steps=len(dts),
        interior=fx.interior,
    )


# --- the gates ------------------------------------------------------------------


@pytest.mark.parametrize("cadence", CADENCES)
def test_a_frequent_sync_cadence_does_not_ripple_the_reach(cadence: float, capsys) -> None:
    """A uniform steady reach must stay smooth however often the clock is interrupted.

    The discriminating gate. Under the pre-fix clamp this reads 12-26 mm at 45 s and
    67-4370 mm at 11.25 s; under equal steps it reads hundredths of a millimetre.
    """
    fx = uniform_reach()
    r = drive_scheduled(fx, cadence)
    with capsys.disabled():
        print("\n" + fx.describe())
        print(r.describe(f"cadence {cadence:g} s"))

    assert r.mass_rel_error < MASS_GATE, (
        f"mass balance {r.mass_rel_error:.2e} -- note this is NOT what this test "
        "gates; it reads ~1e-8 whether or not the reach is rippling"
    )
    assert r.curvature_mm < CURVATURE_MM, (
        f"interior depth curvature {r.curvature_mm:.4f} mm exceeds {CURVATURE_MM} mm "
        f"at a {cadence:g} s sync cadence (spread {r.spread_mm:.4f} mm). The reach is "
        "uniform and steady, so every millimetre of this is spurious: the scheduler "
        "is handing local-inertial an abrupt shorten-then-restore. See "
        "docs/plans/scheduler-equal-steps.md."
    )


def test_the_answer_does_not_depend_on_how_often_the_clock_is_interrupted(capsys) -> None:
    """Choosing an output cadence must not choose a different reach.

    This is the property a user actually relies on when they shorten ``output_every``
    or a slow process's ``interval_s``, and the one the M7 plan's remedy note had to
    warn against because it did not hold. Every cadence shares one measurement
    window, so any artefact common to them cancels.
    """
    fx = uniform_reach()
    control = drive_scheduled(fx, None)
    runs = {c: drive_scheduled(fx, c) for c in CADENCES}

    with capsys.disabled():
        print("\n" + control.describe("no sync points (control)"))
        for c, r in runs.items():
            delta = 1000.0 * float(np.abs(r.depth[fx.interior] - control.depth[fx.interior]).max())
            print(f"{r.describe(f'cadence {c:g} s')}\n{'':17}vs control: {delta:.4f} mm")

    assert control.curvature_mm < CURVATURE_MM, (
        f"the control itself is rippling ({control.curvature_mm:.4f} mm) -- the "
        "fixture, not the scheduler, has changed"
    )
    for c, r in runs.items():
        delta = 1000.0 * float(np.abs(r.depth[fx.interior] - control.depth[fx.interior]).max())
        assert delta < CURVATURE_MM, (
            f"a {c:g} s sync cadence moved the interior depth by {delta:.4f} mm "
            f"against an uninterrupted run of the same reach"
        )


@pytest.mark.parametrize("cadence", CADENCES)
def test_no_step_is_a_small_fraction_of_its_neighbour(cadence: float, capsys) -> None:
    """The mechanism gate: ``dt`` must not jump between adjacent steps.

    A remainder step is not "slightly shorter" -- at an 11.25 s cadence the pre-fix
    scheduler routinely ran one at a **585th** of the step either side of it. This
    fails first, and for the right reason, if a remainder is ever reintroduced.
    """
    fx = uniform_reach()
    r = drive_scheduled(fx, cadence)
    mean, worst = r.mean_dt_jump(2000.0), r.max_dt_jump(2000.0)
    with capsys.disabled():
        print(
            f"\n[clamp-ripple] cadence {cadence:g} s: dt jump mean {100 * mean:.5f}% "
            f"max {100 * worst:.2f}%"
        )
    assert mean < DT_JUMP, (
        f"dt changes by {100 * mean:.3f}% between adjacent steps on average at a "
        f"{cadence:g} s cadence (dt ranges {r.dts.min():.6f}..{r.dts.max():.6f} s, "
        f"worst single jump {100 * worst:.1f}%). The scheduler is filling the span "
        "with full steps plus a remainder."
    )


def test_the_step_never_exceeds_the_cfl_limit_against_a_real_scheme(capsys) -> None:
    """``dt <= dt_raw``, every step, against a **real** state-derived ``compute_dt``.

    :mod:`solver.test_scheduler` asserts the same invariant against a stub. This is
    the version that could catch an interaction with a ``dt`` that actually responds
    to the state -- a stub cycling fixed values cannot drift mid-span, which is the
    only situation where the arithmetic has a choice to make.

    The scheduler chooses *how many* steps fill a span, so it does more than clamp;
    what it must never do is hand back a step longer than the state-derived one,
    which would be a CFL violation it invented. (Freezing the step count at the span
    start -- the other reading of "equal steps" -- does exactly that, by up to 1.59x.)
    """
    fx = uniform_reach()
    st = fx.state("cpu")
    inj = InflowInjector(fx.inflows(), st.grid, "cpu")
    sched = MultiRateScheduler(end_time=END_S, output_every=11.25)

    raws: list[float] = []

    def dt_fn() -> float:
        raws.append(compute_dt(st, alpha=0.7, dt_max=DT_MAX))
        return raws[-1]

    worst = 0.0
    for tick in sched.ticks(dt_fn):
        worst = max(worst, tick.dt / raws[-1])
        inj.apply(st, tick.t0, tick.dt)
        step(st, dt=tick.dt)

    with capsys.disabled():
        print(f"\n[clamp-ripple] largest dt/dt_state_derived over {len(raws)} steps: {worst:.6f}")
    assert worst <= 1.0 + 1e-12, (
        f"the scheduler returned a step {worst:.4f}x the one the scheme computed"
    )
