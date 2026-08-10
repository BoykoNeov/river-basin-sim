"""The bed-wave celerity fixture (M7 build step 3).

A small synthetic straight channel, sized so a low bed bump migrates a measurable
number of cells inside a run that costs ~2 s on Warp's CPU backend. It exists this
early in the build order because **sizing it is a derivation, not a preference**:
the run length follows from the analytical celerity, the activation interval from
the morphological Courant number, and the cell size and bump length from two
competing accuracy limits (below). Discovering any of that at build step 8 would
mean redoing the geometry with the gate already written against it.

This is a **gate fixture, not the demo** (M7 plan §4): ``reach_basin``'s channel
moves millimetres per activation, so a bed wave there will not cross one cell in a
12 h run. The morphology demo (``scenarios/reach_alluvial.toml``, build step 9)
answers a different question and cannot share a scenario with this. There is
deliberately no ``.toml``: every validation gate in this harness builds its state
in Python, and until build step 5 wires ``solver/processes/morphology.py`` the run
loop refuses ``[sediment]`` outright.

**One row, flow along +x.** ``theta`` on a face is computed from that face's own
normal discharge, so a diagonal flow reads roughly half the Shields number in each
component (``solver.core.sediment`` docstring). A fixture that measures the *law*
rather than the projection must be axis-aligned, and ``ny = 1`` makes it exactly so
-- there are no y-faces at all.

The design point
----------------

======================  ===========  =============================================
quantity                value        why this value
======================  ===========  =============================================
slope                   0.002        mild: sets ``h/S = 748 m``, see (2)
Manning n               0.035        natural gravel reach
d50                     8 mm         gravel; the M7 plan's own table uses it
unit discharge          2.5 m^2/s    with the above, ``theta = 4.8 theta_c``, (1)
normal depth            1.496 m      wide-channel Manning, and what the solver hits
Froude                  0.44         subcritical and well clear of 1, see (3)
c_b                     8.63e-3 m/s  ``bed_celerity`` at the design flow
dx                      2.5 m        ``sigma`` = 6 cells, see (4)
nx                      240          600 m: migration + bump + margins, see (5)
bump amplitude          15 mm        1% of the depth: linear, see (6)
bump sigma              15 m         ``sigma/(h/S) = 0.02``, see (2)
activation interval     45 s         ``c_b*interval/dx = 0.155``, see (7)
migration               16 cells     40 m in 4636 s, ~13 000 fast steps
======================  ===========  =============================================

Every one of these is a constraint, and they pull against each other:

1. **``theta`` must clear ``theta_c`` with room.** MPM's excess goes as
   ``(theta - theta_c)^1.5``, so a fixture sitting just over threshold measures the
   threshold rather than the celerity. 4.8x is comfortable and still inside the
   Shields range MPM was calibrated for -- unlike a fast erosive flume, which is
   cheap to run and physically absurd.
2. **The bump must be short compared to the backwater adjustment length ``h/S``.**
   This is the constraint that decides whether the analytical celerity is the right
   reference at all (see *What the analytic reference assumes*). Three design points
   were run: this one at ``sigma/(h/S) = 0.02`` lands within 1% of ``c_b``; a steeper
   ``Fr = 0.77`` point at 0.065 read 0.92 and *stayed* there when ``dx`` was halved
   twice (0.85 / 0.92 / 0.92), so its 8% deficit is physics and not the grid; and a
   shallow steep flume at 0.30 read 0.37, and moved only between 0.27 and 0.38 over
   its own amplitude and bump-length sweeps. Those points differ in Froude as well,
   so this is a joint observation rather than a clean one-variable sweep -- but the
   direction is unambiguous and the mechanism is named in (3).
3. **Froude well below 1.** The surface adjustment over the bump scales as
   ``1/(1 - Fr^2)``, which is 1.24 here and 2.5 at ``Fr = 0.77``; the local-inertial
   scheme also drops advection, so a near-critical fixture measures the scheme's
   weakest regime instead of the transport law.
4. **The bump must be resolved.** ``sigma = 6`` cells. Measured *on this design
   point*, halving and doubling ``dx`` at a fixed physical bump and a fixed Courant
   number gives 0.973 (3 cells) / **0.993** (6) / 1.001 (12), so 6 cells is past the
   knee at a 0.8% cost -- while the crest keeps 0.64 / 0.72 / 0.77 of its height, so
   what a coarser grid mostly costs is the *amplitude*, and the shape estimator is
   what survives it. The rejected ``Fr = 0.77`` point was far more grid-sensitive at
   the same 3 cells (0.85), which is the other half of why it was rejected.
5. **The reach must be longer than everything happening in it.** The bump starts
   200 m in and migrates 40 m; upstream of it the backwater deposits a broad, low
   plateau that reaches back ~``h/S``, and the measurement window has to sit clear
   of both ends (see :meth:`BedWave.window`).
6. **Small amplitude.** 15 mm on 1.5 m of water. At 30 mm the crest travelled 3%
   fast and the shape 2% fast -- nonlinear steepening, since ``c_b`` rises with bed
   elevation -- which is real physics but not what the gate is written against.
7. **The activation interval is the fixture's own, not the 900 s default, and it is
   fenced on both sides.** Above, at 900 s, this bed wave crosses 3.1 cells per
   activation, which is a splitting artefact rather than a result (M7 plan §1.3).
   Below, the limit is not the splitting at all but the **scheme**: a shorter
   interval means more sync-point activations, every one of which clamps ``dt`` and
   so hands local-inertial an abrupt shorten-then-restore that excites a
   short-wavelength mode (M7 plan §4, *"a clamped step is not a free step"*).
   Measured on this fixture with the **bump removed**, so every departure below is
   spurious: +-0.16 mm at 90 s, **+-0.11 mm at 45 s**, +-8.85 mm at 22.5 s and
   +-29.3 mm at 11.25 s -- against a 15 mm bump. 45 s sits inside the band, and
   ``xcorr`` celerity over ``c_b`` across it reads 0.99 at 45 s, 0.996 at 90 s and
   0.97 at 180 s.

   **The first draft of this constraint said the opposite and it was wrong.** It
   read *"short intervals couple the water to the bed more tightly (the physical
   answer is slightly slower than the rigid-lid celerity)"* -- i.e. it took the
   short-interval readings for convergence toward a coupled answer and quoted them
   as physics. They are a numerical artefact, and the discriminator is that they
   appear with **no bed at all**: the same clamp cadence applied to water alone,
   with sediment never armed, ripples a steady 1.4959 m reach by 74 mm at 22.5 s
   and destroys it at 11.25 s, while the mass gate reads 1e-8 throughout.

**The ends are pinned, and that is a sediment boundary condition, not a fudge.**
Boundary faces carry no bedload (they are never updated, which is the closed BC),
so the inlet cell exports without supply and the outlet cell imports without
export. Measured on this fixture over 26 activations with free ends: the inlet
scoured **1.12 m**, the outlet grew an **0.91 m** sill, and its backwater lifted the
reach from 1.495 m to 1.944 m -- a 30% depth error, which would corrupt the
transport everywhere before the bump had moved. Pinning the two end cells with the
``dz_lo == dz_hi == 0`` bound that :func:`solver.core.sediment.exner_update` already
carries turns them into an equilibrium feed and an equilibrium sink: the flux
through face 1 is still computed from real flow, so cell 1 receives exactly the
supply a uniform reach would deliver, and what the bound refuses is *banked* for the
sediment ledger rather than discarded. With the ends pinned the reach holds
1.4952 m against a design 1.4959 m.

That mechanism is per-cell bounds handed to the kernel, so **build step 5's
``morphology.py`` must accept explicit ``dz_lo``/``dz_hi`` arrays (or a frozen-cell
mask) at construction**, not only bounds derived from ``[sediment]``:
``alluvium_thickness = 0`` pins ``lo`` and leaves ``hi = +inf``, so it cannot hold an
outlet down. The sill is not fixture-specific either -- any open-boundary morphology
run grows one, ``reach_alluvial.toml`` included (M7 plan §4).

What the analytic reference assumes
-----------------------------------

:func:`solver.core.sediment.bed_celerity` linearises Exner at **fixed unit discharge
and a rigid water surface**: the bump thins the flow by its own height, which raises
the shear and so the capacity. Reality sits between that and the opposite limit,
where the surface follows the bed so closely that the depth never changes and the
bed wave has no celerity at all. Which limit a fixture is in is set by
``sigma/(h/S)``, which is why constraint (2) is the one that decides whether the
reference is meaningful. Here the two agree to 1%, and the fixture **prints the
achieved flow next to the design flow** so a future change that quietly moves the
reach off its design point shows up as a shifted denominator rather than a shifted
ratio.

No shear partitioning, either: transport sees the *total* Manning shear, and a
natural ``n`` includes form drag the grains never feel. The fixture is
self-consistent because its analytical reference uses the same ``n`` -- but a field
capacity computed this way is an over-estimate, and that is a property of
``sediment.py``, not of this file.

Measuring the migration
-----------------------

Three estimators, because they fail differently and the disagreement is
information (:class:`Migration`):

* **cross-correlation lag** against the initial bump -- gated. It uses the whole
  shape, so it survives the crest flattening that diffusion causes (the crest here
  keeps 0.72 of its height over the run), and it was stable to within 7% across every
  interval, amplitude and grid in the sweep. At the design point it reads
  **0.993 c_b**.
* **crest position**, parabolic sub-cell fit -- printed (1.004 ``c_b`` here). Sharper
  in principle, but a diffused crest is flat, and on a small-amplitude bump the fit
  wandered as far as 0.76 of ``c_b`` with the shape estimator unmoved. Diagnostic,
  not a gate.
* **centroid** above zero -- printed (0.942 ``c_b`` here), and kept precisely because
  it is the one that broke. The backwater deposition plateau upstream of the bump is
  real bed change inside any window wide enough to hold the migration, and it drags a
  centroid upstream: on the rejected ``Fr = 0.77`` design it read 0.63 while the shape
  estimator read 0.85. It is quoted so that "the bed moved" and "the *bump* moved"
  stay visibly different claims.

Each estimator is applied to the initial bed as well and differenced, so the
window's asymmetry cancels instead of biasing the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from solver.core.grid import GRAVITY
from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MassLedger
from solver.core.sediment import (
    SHIELDS_CRITICAL,
    arm_sediment,
    bed_celerity,
    capacity_from_flow,
    morphological_courant,
    shields_from_flow,
)
from solver.core.state import State
from solver.io.config import Inflow
from solver.processes.inflow import InflowInjector
from solver.processes.morphology import MorphologyProcess

# Water in at the head, out at the toe. The east edge is the M3 post-interior sink
# (solver.core.boundaries): it passes water and, by construction, no sediment.
EAST_OPEN = {"east": "open", "west": "closed", "north": "closed", "south": "closed"}


def wide_normal_depth(unit_discharge: float, manning: float, slope: float) -> float:
    """Wide-channel Manning normal depth ``h = (q n / sqrt(S))^(3/5)`` (m).

    The wide form (hydraulic radius ``~ h``) is the right one here: the fixture is a
    single row with closed side walls, so the local-inertial face update carries
    friction per unit width exactly as this expression assumes. It is what the
    solver settles at to four decimals, which is what makes the design ``theta`` and
    the design ``c_b`` the real denominators of the gate rather than nominal ones.
    """
    return (unit_discharge * manning / math.sqrt(slope)) ** 0.6


@dataclass(frozen=True)
class Migration:
    """How far the bump moved, by three estimators, plus what could hide the signal."""

    xcorr_cells: float  # cross-correlation lag against the initial bump -- gated
    peak_cells: float  # crest displacement, parabolic sub-cell fit -- printed
    centroid_cells: float  # centroid of the positive departure -- printed
    elapsed_s: float  # morphological time (excludes the water-only warm-up)
    dx: float
    bump_peak_m: float  # crest height above the reference plane at the end
    background_m: float  # largest |bed change| outside the measurement window
    initial_amplitude_m: float

    def _celerity(self, cells: float) -> float:
        return cells * self.dx / self.elapsed_s

    @property
    def xcorr_celerity(self) -> float:
        """The gated number: shape displacement over elapsed morphological time."""
        return self._celerity(self.xcorr_cells)

    @property
    def peak_celerity(self) -> float:
        return self._celerity(self.peak_cells)

    @property
    def centroid_celerity(self) -> float:
        return self._celerity(self.centroid_cells)

    @property
    def signal_to_background(self) -> float:
        """Crest height over the largest bed change outside the window.

        Below ~5 the estimators are no longer measuring the bump: they are measuring
        whatever else the run did to the bed (see the module docstring on the
        centroid). Asserted, so the gate cannot degrade quietly into that.
        """
        if self.background_m <= 0.0:
            return math.inf
        return self.bump_peak_m / self.background_m

    @property
    def amplitude_retained(self) -> float:
        """Crest height over its initial value -- the wave must still be a wave."""
        return self.bump_peak_m / self.initial_amplitude_m


@dataclass(frozen=True)
class BedWave:
    """The celerity fixture's geometry and its derived design point.

    Frozen, so a test can ``dataclasses.replace`` one number (the interval, the
    pinned-cell count) and keep every derived quantity consistent with it. The
    defaults are the design point derived in the module docstring; nothing here
    reads a scenario file.
    """

    dx: float = 2.5
    nx: int = 240
    slope: float = 0.002
    manning: float = 0.035
    d50: float = 0.008
    porosity: float = 0.4
    unit_discharge: float = 2.5  # m^2/s (per unit width == per row, ny = 1)
    bump_amplitude_m: float = 0.015
    bump_sigma_m: float = 15.0
    bump_cell: int = 80
    migration_cells: float = 16.0
    interval_s: float = 45.0
    warmup_s: float = 1200.0  # water-only spin-up; normal depth is reached by ~1000 s
    outlet_bed_m: float = 5.0  # bed elevation at the open toe (a plain, small datum)
    pinned_cells: int = 1  # end cells held at dz = 0 -- the sediment BC, see docstring

    # --- the flow the fixture is designed around ------------------------------

    @property
    def normal_depth(self) -> float:
        return wide_normal_depth(self.unit_discharge, self.manning, self.slope)

    @property
    def discharge_m3s(self) -> float:
        """What an :class:`~solver.io.config.Inflow` at the head must deliver."""
        return self.unit_discharge * self.dx

    @property
    def froude(self) -> float:
        h = self.normal_depth
        return self.unit_discharge / h / math.sqrt(GRAVITY * h)

    @property
    def shields(self) -> float:
        return float(
            shields_from_flow(self.unit_discharge, self.normal_depth, self.manning, self.d50)
        )

    @property
    def shields_margin(self) -> float:
        """``theta / theta_c`` -- constraint (1); below ~3 the gate measures threshold."""
        return self.shields / SHIELDS_CRITICAL

    def at_shields(self, ratio: float) -> BedWave:
        """This fixture re-grained to sit at ``ratio * theta_c``, flow untouched.

        ``theta = tau / (rho s' g d50)`` and ``tau`` carries no grain size at all, so
        ``theta`` is **exactly** inversely proportional to ``d50`` -- and ``d50`` does
        not enter the hydraulics either (Manning ``n`` is separate). So the threshold
        pair (M7 plan §3, *"no transport below theta_c"*) is one variable: the same
        reach, the same flow, the same bump, a different grain size. Building a second
        fixture at a different slope or discharge would move the hydraulics too and
        the pair would stop being a clean contrast.
        """
        if ratio <= 0.0:
            raise ValueError(f"shields ratio must be > 0, got {ratio}")
        return replace(self, d50=self.d50 * self.shields / (ratio * SHIELDS_CRITICAL))

    @property
    def relative_submergence(self) -> float:
        """``h/d50`` -- how many grain diameters deep the flow is.

        The regime check M7 build step 8 gates its scenarios on. MPM is a *channel
        bedload* law and says nothing about a sheet thinner than the grains it is
        moving: this fixture reads 187, while the millimetric overland sheet at the
        wet/dry guard that build step 6 measured reads **0.5** (M7 plan §4).
        """
        return self.normal_depth / self.d50

    @property
    def capacity(self) -> float:
        """Bedload capacity per unit width at the design flow (m^2/s)."""
        return float(
            capacity_from_flow(self.unit_discharge, self.normal_depth, self.manning, self.d50)
        )

    @property
    def adjustment_length_m(self) -> float:
        """Backwater adjustment length ``h/S`` -- the yardstick for the bump length."""
        return self.normal_depth / self.slope

    @property
    def bump_slenderness(self) -> float:
        """``sigma / (h/S)`` -- constraint (2), the one that validates the reference."""
        return self.bump_sigma_m / self.adjustment_length_m

    # --- the bed wave ---------------------------------------------------------

    @property
    def celerity(self) -> float:
        """Analytical ``c_b`` at the design flow (m/s) -- the gate's denominator."""
        return float(
            bed_celerity(
                self.unit_discharge,
                self.normal_depth,
                self.manning,
                self.d50,
                self.porosity,
            )
        )

    def celerity_at(self, unit_discharge: float, depth: float) -> float:
        """``c_b`` at a *measured* flow, so the gate can use the flow it really got."""
        return float(bed_celerity(unit_discharge, depth, self.manning, self.d50, self.porosity))

    @property
    def courant(self) -> float:
        """Cells the bed wave crosses per activation -- must stay well under 1."""
        return morphological_courant(self.celerity, self.interval_s, self.dx)

    @property
    def migration_m(self) -> float:
        return self.migration_cells * self.dx

    @property
    def run_s(self) -> float:
        """Design morphological run length: how long ``migration_cells`` of travel takes.

        The *target*. What actually runs is :attr:`morph_time_s`, a whole number of
        activations -- a trailing part-interval would accumulate transport it never
        applied, which is a silent few-tenths-of-a-percent error in the denominator
        of every celerity in the gate.
        """
        return self.migration_m / self.celerity

    @property
    def activations(self) -> int:
        return max(1, int(round(self.run_s / self.interval_s)))

    @property
    def morph_time_s(self) -> float:
        """Morphological time the run really covers: ``activations * interval_s``."""
        return self.activations * self.interval_s

    @property
    def end_time_s(self) -> float:
        return self.warmup_s + self.morph_time_s

    @property
    def bump_sigma_cells(self) -> float:
        return self.bump_sigma_m / self.dx

    # --- geometry -------------------------------------------------------------

    def reference_plane(self) -> np.ndarray:
        """The unperturbed bed, ``(nx,)`` float64 -- the datum every measurement uses.

        Falling in +x (west high, east low), so the flow runs toward the open toe.
        Kept in float64 and *not* re-read from the state: the bed the run ends with
        is a float32 field whose departure from this plane is millimetric, and
        differencing two float32 elevations at ~6 m would throw away a third of the
        signal's digits.
        """
        j = np.arange(self.nx, dtype=np.float64)
        return self.outlet_bed_m + (self.nx - 1 - j) * self.dx * self.slope

    def bump_profile(self) -> np.ndarray:
        """The initial bed perturbation, ``(nx,)`` float64. Gaussian, so it is smooth
        at every scale the scheme can see -- a step or a triangle would put its own
        truncation error into the celerity."""
        j = np.arange(self.nx, dtype=np.float64)
        sig = self.bump_sigma_cells
        return self.bump_amplitude_m * np.exp(-0.5 * ((j - self.bump_cell) / sig) ** 2)

    def bed(self) -> np.ndarray:
        """The initial bed as the solver wants it: ``(1, nx)`` float32."""
        return (self.reference_plane() + self.bump_profile()).reshape(1, self.nx).astype(np.float32)

    def state(self, device: str = "cpu") -> State:
        """A state seeded **at** normal depth with the toe open.

        Seeding the design depth rather than a dry bed is what keeps the warm-up to
        ~1000 s; the discharges still start at zero, so the reach establishes its own
        steady profile and the fixture never asserts a flow it hand-placed.
        """
        st = State.from_bed(
            self.bed(),
            dx=self.dx,
            depth=self.normal_depth,
            manning=self.manning,
            device=device,
        )
        st.set_open_boundaries(EAST_OPEN)
        return st

    def inflows(self) -> list[Inflow]:
        """A constant hydrograph at the head cell, in m^3/s (the §7.1 unit)."""
        q = self.discharge_m3s
        return [Inflow(cell=(0, 0), hydrograph=[(0.0, q), (1.0e9, q)])]

    def bed_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell bounds on the *cumulative* bed change for ``exner_update``.

        Unbounded everywhere except the ``pinned_cells`` at each end, which are held
        at exactly zero -- the equilibrium sediment BC the module docstring derives.
        ``pinned_cells = 0`` returns the free-ended bed, which is what the
        boundary-artefact test measures rather than assumes.
        """
        lo = np.full((1, self.nx), -np.inf, dtype=np.float32)
        hi = np.full((1, self.nx), np.inf, dtype=np.float32)
        k = self.pinned_cells
        if k > 0:
            lo[0, :k] = hi[0, :k] = 0.0
            lo[0, self.nx - k :] = hi[0, self.nx - k :] = 0.0
        return lo, hi

    # --- measurement ----------------------------------------------------------

    @property
    def window(self) -> tuple[int, int]:
        """Half-open cell range the estimators look at.

        Three sigma either side of where the bump starts and ends, so it holds the
        whole wave over the whole run and still stands clear of the pinned ends and
        of the outlet drawdown. It does **not** exclude the upstream deposition
        plateau -- nothing can, since the plateau overlaps the bump's own tail --
        which is why the gated estimator is a shape correlation.
        """
        pad = int(math.ceil(3.0 * self.bump_sigma_cells))
        lo = self.bump_cell - pad
        hi = self.bump_cell + int(math.ceil(self.migration_cells)) + pad
        return lo, min(hi, self.nx)

    @property
    def interior(self) -> slice:
        """Cells whose flow is the design flow: clear of the head cell and the toe."""
        pad = max(8, int(math.ceil(2.0 * self.bump_sigma_cells)))
        return slice(pad, self.nx - pad)

    def departure(self, z: np.ndarray) -> np.ndarray:
        """Bed elevation minus the reference plane, ``(nx,)`` float64 (metres)."""
        return np.asarray(z, dtype=np.float64).reshape(self.nx) - self.reference_plane()

    def measure(self, z_final: np.ndarray) -> Migration:
        """Displacement of the bump between the initial bed and ``z_final``.

        Every estimator is evaluated on *both* beds and differenced, so the window's
        asymmetry (it extends downstream to hold the migration) cancels rather than
        biasing the answer.
        """
        lo, hi = self.window
        ref = self.bump_profile()
        final = self.departure(z_final)

        peak = _parabolic_peak(final[lo:hi]) - _parabolic_peak(ref[lo:hi])
        centroid = _positive_centroid(final[lo:hi]) - _positive_centroid(ref[lo:hi])
        lag = _xcorr_lag(ref, final, lo, hi, int(math.ceil(2.0 * self.migration_cells)))

        # Background: the largest bed change outside the window, skipping the pinned
        # cells themselves (they are zero by definition) but not their neighbours,
        # which is where a boundary artefact would first show up.
        k = self.pinned_cells
        outside = np.concatenate([np.abs(final[k:lo]), np.abs(final[hi : self.nx - k])])
        return Migration(
            xcorr_cells=lag,
            peak_cells=peak,
            centroid_cells=centroid,
            elapsed_s=self.morph_time_s,
            dx=self.dx,
            bump_peak_m=float(final[lo:hi].max()),
            background_m=float(outside.max()) if outside.size else 0.0,
            initial_amplitude_m=self.bump_amplitude_m,
        )

    # --- the habit of saying the numbers out loud -----------------------------

    def describe(self) -> str:
        """The design point, for a test to print. Nothing is hidden in a tolerance."""
        lo, hi = self.window
        return (
            f"[bedwave] {self.nx} x 1 cells @ {self.dx:g} m = {self.nx * self.dx:g} m reach, "
            f"S={self.slope:g}, n={self.manning:g}, d50={1000 * self.d50:g} mm, "
            f"p={self.porosity:g}\n"
            f"          flow: q={self.unit_discharge:g} m2/s (Q={self.discharge_m3s:g} m3/s)  "
            f"h_n={self.normal_depth:.4f} m  Fr={self.froude:.2f}  "
            f"theta={self.shields:.4f} ({self.shields_margin:.1f}x theta_c)  "
            f"q_s={self.capacity:.4e} m2/s\n"
            f"          bump: {1000 * self.bump_amplitude_m:g} mm x sigma {self.bump_sigma_m:g} m "
            f"({self.bump_sigma_cells:g} cells) at cell {self.bump_cell}  "
            f"h/S={self.adjustment_length_m:.0f} m  sigma/(h/S)={self.bump_slenderness:.3f}\n"
            f"          wave: c_b={self.celerity:.5e} m/s  {self.migration_cells:g} cells "
            f"({self.migration_m:g} m) in {self.run_s:.0f} s  "
            f"{self.activations} activations @ {self.interval_s:g} s "
            f"= {self.morph_time_s:g} s (Cr={self.courant:.3f})\n"
            f"          window cells [{lo}, {hi})  pinned ends={self.pinned_cells}  "
            f"warm-up {self.warmup_s:g} s"
        )


# --- driving it ----------------------------------------------------------------
# This lived in `validation.test_bed_wave` as a private `_drive` while it was the
# provisional step-3 harness. Build step 5 landed the real process and step 8 needs
# it from a second gate file, so it is part of the durable fixture now. `alpha` and
# `dt_max` are parameters rather than constants because constraint (7)'s finding is
# only reproducible by varying them: capping the fast step so it divides the
# interval exactly stops the clamp firing at all, which is how the artefact was told
# apart from the step size.

_EPS_T = 1e-9  # activation-time comparison slack, in seconds
DT_MAX = 5.0
"""Inert at this scale (the state-derived step is ~0.46 s) and kept only so the
harness cannot wander off if a future fixture is much deeper."""


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
    courant: float = 0.0  # largest morphological Courant number the process measured

    def median_depth(self, where: slice) -> float:
        return float(np.median(self.depth[where]))

    def median_unit_discharge(self, where: slice) -> float:
        return float(np.median(self.face_q[1:-1][where]))


def drive(
    fx: BedWave,
    *,
    morphology: bool = True,
    end_s: float | None = None,
    alpha: float = 0.7,
    dt_max: float = DT_MAX,
) -> Run:
    """Run the fixture: local-inertial water, plus (optionally) the M7 morphology.

    ``end_s`` stops early (the design-point check needs only the warm-up and a few
    intervals past it); ``morphology=False`` never arms the state, so nothing is
    launched, the bed is untouched, and there are no activation boundaries to land on.

    The bed update is :class:`~solver.processes.morphology.MorphologyProcess` driven
    by hand rather than by the scheduler, because the fixture's water-only warm-up is
    not a scheduler concept -- the sync-point algebra itself is M5's and is tested
    there. The physics is not hand-wired: the transport integral accumulates inside
    ``step`` and the activation is one ``advance`` call.

    **The clamp below is the scheduler's, deliberately.** Landing exactly on each
    activation is what :class:`~solver.scheduler.MultiRateScheduler` does for every
    real run (``dt = min(dt, next_sync - t)``), so the fixture inherits the artefact
    constraint (7) describes rather than dodging it. A harness that skipped the clamp
    would measure a cleaner reach than any scenario can actually run.
    """
    st = fx.state("cpu")
    inj = InflowInjector(fx.inflows(), st.grid, "cpu")
    ledger = MassLedger.from_state(st)
    morph: MorphologyProcess | None = None

    end = fx.end_time_s if end_s is None else float(end_s)
    t, steps, acts = 0.0, 0, 0
    while t < end - _EPS_T:
        dt = compute_dt(st, alpha=alpha, dt_max=dt_max)
        # Land exactly on the warm-up boundary and on every activation, so an
        # interval is an interval. With morphology off there is nothing to land on --
        # and the activation counter never advances either, so keeping the clamp
        # would freeze `dt` at zero and spin here forever the first time `t` passed
        # one interval.
        if morphology:
            edge = (
                fx.warmup_s
                if t < fx.warmup_s - _EPS_T
                else fx.warmup_s + (acts + 1) * fx.interval_s
            )
        else:
            edge = end
        edge = min(edge, end)
        if t + dt > edge:
            dt = edge - t
        assert dt > 0.0, f"harness made no progress at t={t} (edge={edge}, acts={acts})"

        # Arm *at* the warm-up boundary: `z0` is captured here and is still the
        # pristine bed, so morphology begins with the very next step and every
        # earlier step ran the untouched M6 kernels. The **bounds** are handed in
        # whole -- they are the fixture's equilibrium sediment BC, which is exactly
        # the case `[sediment]` cannot express (M7 plan §2).
        if morphology and morph is None and t >= fx.warmup_s - _EPS_T:
            arm_sediment(st, fx.d50, fx.porosity)
            lo_h, hi_h = fx.bed_bounds()
            morph = MorphologyProcess(st, fx.interval_s, dz_lo=lo_h, dz_hi=hi_h)

        ledger.add_inflow(inj.apply(st, t, dt))
        step(st, dt=dt)  # accumulates the transport integral in-step once armed
        steps += 1
        t += dt

        if morph is not None and t >= fx.warmup_s + (acts + 1) * fx.interval_s - _EPS_T:
            morph.advance(t, fx.interval_s)
            acts += 1
    ledger.record(st, t)

    sed = st.sediment
    return Run(
        bed=st.z.numpy()[0].astype(np.float64),
        depth=st.h.numpy()[0].astype(np.float64),
        face_q=st.qx.numpy()[0].astype(np.float64),
        dz_cum=(
            sed.bed_change_numpy()[0] if sed is not None else np.zeros(fx.nx, dtype=np.float64)
        ),
        banked_m=0.0 if sed is None else float(sed.dz_unapplied.numpy().sum()),
        mass_rel_error=ledger.max_rel_error,
        steps=steps,
        activations=acts,
        t=t,
        courant=0.0 if morph is None else morph.peak_courant,
    )


# --- estimators ---------------------------------------------------------------
# Host-side and deliberately dumb: no SciPy, and each one is a few lines so a
# reader can see what it would take to fool it.


def _parabolic_peak(a: np.ndarray) -> float:
    """Crest position in cells, with a three-point parabolic sub-cell fit."""
    k = int(np.argmax(a))
    if k == 0 or k == len(a) - 1:
        return float(k)
    y0, y1, y2 = float(a[k - 1]), float(a[k]), float(a[k + 1])
    den = y0 - 2.0 * y1 + y2
    if den == 0.0:
        return float(k)
    return float(k) + 0.5 * (y0 - y2) / den


def _positive_centroid(a: np.ndarray) -> float:
    """Centroid (cells) of the positive part -- the estimator the plateau defeats."""
    w = np.clip(a, 0.0, None)
    total = w.sum()
    if total <= 0.0:
        return 0.0
    return float((w * np.arange(len(w), dtype=np.float64)).sum() / total)


def _xcorr_lag(ref: np.ndarray, final: np.ndarray, lo: int, hi: int, max_lag: int) -> float:
    """Downstream shift (cells) maximising ``ref`` shifted into ``final``'s window.

    Explicit slices rather than ``np.roll``: the reference is the full-domain bump,
    so shifting it by ``k`` is reading ``ref[lo-k : hi-k]`` -- no wrap-around, and a
    lag can never fold a downstream tail back onto the upstream side. Integer lag,
    then the same parabolic refinement the crest estimator uses.
    """
    max_lag = min(max_lag, lo)  # keep every shifted window inside the domain
    seg = final[lo:hi]
    corr = np.array(
        [float(np.dot(ref[lo - k : hi - k], seg)) for k in range(max_lag + 1)], dtype=np.float64
    )
    if int(np.argmax(corr)) == max_lag:
        # The correlation peak is *at* the search limit, so the real lag is somewhere
        # beyond it and the parabolic fit would silently report the limit as the
        # answer. Loud, because a re-parametrised fixture is exactly how this happens.
        raise ValueError(
            f"cross-correlation saturated at its {max_lag}-cell search limit: the bump "
            "moved further than the fixture expected"
        )
    return _parabolic_peak(corr)
