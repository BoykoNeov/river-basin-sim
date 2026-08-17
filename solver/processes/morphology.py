"""Morphology on the slow clock -- Exner applied at activations (M7, HANDOFF §8/§9).

The second consumer of the multi-rate seam :mod:`solver.scheduler` opened at M5,
and the first one that writes to a field the flood scheme reads every step. The
split of labour is the same as the reservoir's, and the same sentence covers both:
*a slow process reads what the fast loop accumulated, applies it once, and does
nothing at all in between.*

**The two halves live in different modules on purpose.**

* The **fast half** is :func:`solver.core.sediment.accumulate_transport`, launched
  from inside :func:`solver.core.local_inertial.step`. It integrates ``q_s*dt`` onto
  the faces every step, so what the bed moves by is a proper time integral rather
  than an instantaneous sample scaled by the interval -- ``q_s`` goes as roughly
  ``h^1.5 S_f^1.5``, so sampling a passing flood wave once and multiplying by 900 s
  misrepresents it badly in both directions depending on where the sample lands
  (M7 plan §1.3).
* The **slow half** is here: at each activation form ``dz = -1/(1-p)*div(qs_int)/dx``,
  add it into the float64 ``dz_cum``, rebuild ``z`` from the pristine ``z0``, refresh
  ``eta``, and zero the integral. The bed still changes only on the slow clock, so
  the operator splitting is intact.

**Bounds are handed in whole, not derived from ``[sediment]``** -- the requirement
the celerity fixture (:mod:`validation.bedwave`) made of this module before it
existed. ``alluvium_thickness = 0`` pins the *floor* and leaves the ceiling open, so
nothing in the config table can hold an outlet cell **down**, and an open-boundary
run without pinned ends grows a sill: measured on that fixture, the outlet aggraded
0.053 m per activation (0.91 m in 26) and its backwater lifted the whole reach's
depth by 30% -- a hydraulic error long before it is a visible one. So
:func:`bed_change_bounds` builds ``(dz_lo, dz_hi)`` from whatever a caller has --
an alluvium floor, a set of frozen cells, or both -- and :class:`MorphologyProcess`
takes the pair directly.

**What the bounds refuse is banked, never clamped away** (M7 plan §1.5). Structure
cells are frozen because a dam is engineered, not alluvial -- nothing else stops the
flow scouring one out from under its own release rule -- and the mask is applied at
the *cell*: the divergence is formed from real face fluxes everywhere, then that
cell's ``dz`` is discarded and the discarded metres accumulate in ``dz_unapplied``
for the sediment ledger (M7 build step 6) to convert at ``A*(1-p)``. Zeroing the
faces bounding a structure instead would also stop sediment routing *through* it,
which is a different physical statement and not the one M7 makes.

**Water is untouched.** ``h`` is volume per unit plan area and no kernel here reads
or writes it, so ``eta = z + h`` simply rises with the bed and water volume is
conserved by construction -- the same argument that made M6's channels exactly
mass-conserving. A cell can still deposit its way dry, and there is deliberately no
special rule for that: the existing ``H_DRY`` guard already decides what dry means
everywhere else (M7 plan §1.6).

**Ordering within a tick.** When a reservoir and this process are both due, the run
loop advances the reservoir first, because its rule reads a *stage* (``z + h``) and
that reading should be of the bed the interval's water actually flowed over, not of
one this activation is about to move. Morphology last also means the bed the next
interval steps on is the one every diagnostic recorded at this activation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from solver.core.local_inertial import compute_eta, compute_eta_channels
from solver.core.sediment import (
    SedimentError,
    courant_summary,
    exner_update,
    rebuild_bed,
)
from solver.core.state import State
from solver.scheduler import EPS_T, SlowProcess


def bed_change_bounds(
    shape: tuple[int, int],
    *,
    alluvium_thickness: np.ndarray | float | None = None,
    frozen_cells: object = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Build the per-cell ``(dz_lo, dz_hi)`` bounds on the **cumulative** bed change.

    Unbounded by default (``-inf, +inf``), which is the "unlimited alluvium, nothing
    engineered" case and costs one comparison per cell per activation.

    ``alluvium_thickness`` (metres, scalar or field) is the erodible layer below the
    initial bed: it sets ``dz_lo = -thickness`` and leaves the ceiling open, so a
    floored cell can always be lifted back off its limit by deposition. A thickness
    of zero is bedrock -- the way to spell "the bed here cannot move", which a zero
    ``d50`` deliberately does *not* mean (:func:`solver.core.sediment.validate_grain_size`).

    ``frozen_cells`` is an iterable of ``(i, j)`` held at exactly zero in both
    directions: structure cells, and the equilibrium ends of a fixture reach.
    """
    ny, nx = shape
    lo = np.full((ny, nx), -np.inf, dtype=np.float32)
    hi = np.full((ny, nx), np.inf, dtype=np.float32)
    if alluvium_thickness is not None:
        thickness = np.asarray(alluvium_thickness, dtype=np.float32)
        if thickness.ndim and thickness.shape != (ny, nx):
            raise SedimentError(f"alluvium thickness shape {thickness.shape} != grid {(ny, nx)}")
        if (thickness < 0).any():
            raise SedimentError("alluvium thickness must be >= 0 m (0 = bedrock, not a hole)")
        lo = np.broadcast_to(-thickness, (ny, nx)).astype(np.float32).copy()
    for i, j in frozen_cells:
        if not (0 <= i < ny and 0 <= j < nx):
            raise SedimentError(f"frozen cell {(i, j)} is outside the {ny}x{nx} grid")
        lo[i, j] = 0.0
        hi[i, j] = 0.0
    return lo, hi


@dataclass
class BedChangeRecord:
    """One slow-clock activation of the bed update.

    Volumes are **solid** volumes (``A*(1-p)*dz``), the unit the sediment ledger
    (M7 build step 6) balances in, so that step is purely additive over this series.
    ``banked_m3`` is cumulative rather than per-activation because ``dz_unapplied``
    is itself a running total -- a bound refuses a little of nearly every activation
    once it is active, and the ledger wants the total owed, not the last instalment.
    """

    time: float
    interval_s: float  # the elapsed simulated time this activation applied
    applied_m3: float  # solid volume the bed gained *this* activation (signed)
    cumulative_m3: float  # ... and since the run began
    banked_m3: float  # cumulative solid volume the bounds refused (never discarded)
    dz_min_m: float  # deepest scour and highest fill so far, for a quick eyeball
    dz_max_m: float
    celerity_m_s: float  # fastest bed wave anywhere in the domain, at this instant
    courant: float  # ... as cells crossed per interval; > 1 is a splitting artefact
    # The same Courant number reduced three more ways, because one field maximum of a
    # one-sided bound cannot describe a reach (:class:`~solver.core.sediment.CourantSummary`).
    # Companions, never substitutes: `courant` above is unchanged and is still what
    # the run warns on and what the validation suite asserts.
    courant_moving: float  # peak over cells that carried bed change this activation
    courant_in_regime: float  # peak over cells the transport law actually applies to
    over_courant_share: float  # fraction of this activation's gross |dz| over the gate
    courant_cells: int  # cells over the gate ...
    live_cells: int  # ... out of the cells transporting at all

    def as_dict(self) -> dict:
        return {
            "time": self.time,
            "interval_s": self.interval_s,
            "applied_m3": self.applied_m3,
            "cumulative_m3": self.cumulative_m3,
            "banked_m3": self.banked_m3,
            "dz_min_m": self.dz_min_m,
            "dz_max_m": self.dz_max_m,
            "celerity_m_s": self.celerity_m_s,
            "courant": self.courant,
            "courant_moving": self.courant_moving,
            "courant_in_regime": self.courant_in_regime,
            "over_courant_share": self.over_courant_share,
            "courant_cells": self.courant_cells,
            "live_cells": self.live_cells,
        }


class MorphologyProcess:
    """Applies Exner to the bed on the slow clock (operator splitting).

    Construct **after** the state is armed for morphology
    (:func:`solver.core.sediment.arm_sediment`, which captures ``z0`` and therefore
    must run once the bed is final -- after barriers), then hand
    :meth:`as_slow_process` to the scheduler.
    """

    def __init__(
        self,
        state: State,
        interval_s: float,
        *,
        dz_lo: np.ndarray | None = None,
        dz_hi: np.ndarray | None = None,
        name: str = "morphology",
    ):
        sed = state.sediment
        if sed is None:
            raise SedimentError(
                "morphology needs a state armed with arm_sediment(); arming captures z0 "
                "from the final bed, which this process rebuilds z from at every activation"
            )
        if interval_s <= 0:
            raise SedimentError(f"morphology interval_s must be > 0, got {interval_s}")
        self.state = state
        self.sediment = sed
        self.interval_s = float(interval_s)
        self.name = name
        self.records: list[BedChangeRecord] = []
        self._peak_courant = 0.0
        self._peak_moving = 0.0
        self._peak_in_regime = 0.0
        self._peak_over_share = 0.0
        self._peak_courant_cells = 0
        self._peak_live_cells = 0

        shape = state.grid.shape
        if (dz_lo is None) != (dz_hi is None):
            raise SedimentError("give both dz_lo and dz_hi, or neither")
        if dz_lo is None:
            dz_lo, dz_hi = bed_change_bounds(shape)
        lo = np.ascontiguousarray(dz_lo, dtype=np.float32)
        hi = np.ascontiguousarray(dz_hi, dtype=np.float32)
        if lo.shape != shape or hi.shape != shape:
            raise SedimentError(f"bed-change bounds {lo.shape}/{hi.shape} != grid {shape}")
        if (lo > hi).any():
            raise SedimentError("bed-change bounds cross: dz_lo must be <= dz_hi everywhere")
        self.dz_lo = wp.array(lo, dtype=wp.float32, device=state.device)
        self.dz_hi = wp.array(hi, dtype=wp.float32, device=state.device)
        frozen = (lo == 0.0) & (hi == 0.0)
        self.frozen_cells = int(frozen.sum())
        # A frozen cell has a finite floor too; count it once, as the stronger limit.
        self.floored_cells = int((np.isfinite(lo) & ~frozen).sum())

    # --- the SlowProcess interface -------------------------------------------- #
    def as_slow_process(self) -> SlowProcess:
        """Wrap this process as a scheduler slow process at its own cadence."""
        return SlowProcess(name=self.name, interval=self.interval_s, advance=self.advance)

    def advance(self, t: float, dt_slow: float) -> BedChangeRecord:
        """Apply one interval's accumulated transport to the bed; return the record.

        ``dt_slow`` is the exact elapsed simulated time the scheduler hands over. It
        does **not** scale the update -- the transport integral already carries its
        own time, which is the whole point of accumulating it in the fast loop -- and
        is used only to report the morphological Courant number over the interval
        that actually elapsed.

        It is nonetheless **checked against the configured interval**, because an
        activation covering more than one interval is a silently coarsened splitting:
        the integral stays exact and mass is fine, but the bed jumps by several
        intervals at once and ``interval_s`` stops meaning what the interval-
        independence gate (M7 plan §3) assumes. A scheduler-driven run cannot produce
        it -- activations are sync points, so a step is clamped to land on one, and a
        ``dt_max`` far coarser than ``interval_s`` still yields one activation per
        interval (verified against :class:`~solver.scheduler.MultiRateScheduler`).
        The check is therefore aimed at a **hand-driven** caller, which is exactly how
        the celerity fixture drives this process and how a future harness will.
        """
        if dt_slow > self.interval_s * (1.0 + 1e-6) + EPS_T:
            raise SedimentError(
                f"morphology was handed {dt_slow} s to apply but is configured for "
                f"{self.interval_s} s: more than one interval has collapsed into a single "
                "bed update, which coarsens the operator splitting without changing the "
                "mass balance -- so it would not show up as an error anywhere else"
            )
        st, sed, g = self.state, self.sediment, self.state.grid
        # **The .copy() is load-bearing**, and it fails in the direction of the answer
        # a reader is hoping for. On the CPU backend `warp.array.numpy()` hands back a
        # view of the array's own storage, so without it this snapshot would be
        # rewritten by the Exner launch below, difference to zero everywhere, and make
        # every companion diagnostic read "no bed moved anywhere near the gate".
        # Gated in `solver/test_morphology.py` by requiring the differenced field to
        # reproduce `applied_m3`, which is computed independently just below.
        prev_dz = sed.dz_cum.numpy().copy()
        before = float(prev_dz.sum())

        wp.launch(
            exner_update,
            dim=g.shape,
            inputs=[
                sed.qs_int_x, sed.qs_int_y, self.dz_lo, self.dz_hi, float(g.dx),
                float(sed.inv_one_minus_p), sed.dz_cum, sed.dz_unapplied,
            ],
            device=st.device,
        )  # fmt: skip
        wp.launch(rebuild_bed, dim=g.shape, inputs=[sed.z0, sed.dz_cum, st.z], device=st.device)
        # `eta` is a function of the bed and is now one activation stale. The next
        # fast step recomputes it first thing, so this changes no physics -- but a
        # state whose eta disagrees with its own z is a trap for anything that reads
        # between ticks (the output writer, a diagnostic, an interrupted run).
        self._refresh_eta()
        sed.clear_integral()

        solid = g.cell_area * (1.0 - sed.porosity)
        dz = sed.dz_cum.numpy()
        cumulative = float(dz.sum())
        # `dz - prev_dz` is what this activation applied, which is what the companion
        # reductions weight by; the cumulative field is what the record reports.
        cs = courant_summary(st, dz - prev_dz, dt_slow)
        self._peak_courant = max(self._peak_courant, cs.courant)
        self._peak_moving = max(self._peak_moving, cs.courant_moving)
        self._peak_in_regime = max(self._peak_in_regime, cs.courant_in_regime)
        self._peak_over_share = max(self._peak_over_share, cs.over_courant_share)
        self._peak_courant_cells = max(self._peak_courant_cells, cs.courant_cells)
        self._peak_live_cells = max(self._peak_live_cells, cs.live_cells)
        rec = BedChangeRecord(
            time=float(t),
            interval_s=float(dt_slow),
            applied_m3=(cumulative - before) * solid,
            cumulative_m3=cumulative * solid,
            banked_m3=float(sed.dz_unapplied.numpy().sum()) * solid,
            dz_min_m=float(dz.min()),
            dz_max_m=float(dz.max()),
            celerity_m_s=cs.celerity_m_s,
            courant=cs.courant,
            courant_moving=cs.courant_moving,
            courant_in_regime=cs.courant_in_regime,
            over_courant_share=cs.over_courant_share,
            courant_cells=cs.courant_cells,
            live_cells=cs.live_cells,
        )
        self.records.append(rec)
        return rec

    def _refresh_eta(self) -> None:
        """Recompute ``eta`` from the rebuilt bed, through the run's own storage curve."""
        st, g = self.state, self.state.grid
        chan = st.channels
        if chan is None:
            wp.launch(compute_eta, dim=g.shape, inputs=[st.h, st.z, st.eta], device=st.device)
        else:
            wp.launch(
                compute_eta_channels,
                dim=g.shape,
                inputs=[st.h, st.z, chan.w, chan.d, float(g.dx), st.eta],
                device=st.device,
            )

    # --- what a run reports ---------------------------------------------------- #
    @property
    def peak_courant(self) -> float:
        """Largest morphological Courant number seen at any activation so far.

        A bed wave crossing more than a cell per activation is a splitting artefact
        rather than a result -- the analogue of M5's *"54,000 m^3 into one 40 m cell
        is a 34 m column"*. Recorded here at every activation; **gating** it against
        a scenario is M7 build step 8.

        **Read it beside :meth:`courant_breakdown`.** This is a field maximum of a
        one-sided bound, and on any run that advances a wetting front it reports the
        wet/dry guard rather than the reach -- 39 271 against 19.4 on the M7 demo,
        whose bed is nonetheless right to 0.9% when the interval is halved. It is
        still what the run warns on and what the validation suite asserts,
        deliberately and *after* measuring the alternatives
        (``docs/plans/morph-courant-diagnostic.md`` §2.4): every filtered version of
        this number is quieter on a run that is far worse than this one.
        """
        return self._peak_courant

    def courant_breakdown(self) -> str:
        """One line of context for :attr:`peak_courant`, for the run's own output.

        Every figure in it is a maximum over activations and says so, because that is
        what :attr:`peak_courant` is; mixing a peak with a snapshot would be its own
        small lie.
        """
        return (
            f"{self._peak_in_regime:.2f} over cells the law applies to, "
            f"up to {self._peak_courant_cells} of {self._peak_live_cells} cells over "
            f"the gate carrying up to {100 * self._peak_over_share:.0f}% of an "
            f"activation's bed change"
        )

    @property
    def series(self) -> list[dict]:
        """The bed-change history, for the canonical store's ``.zattrs``."""
        return [r.as_dict() for r in self.records]

    def summary(self) -> str:
        bounds = []
        if self.frozen_cells:
            bounds.append(f"{self.frozen_cells} frozen cells")
        if self.floored_cells:
            bounds.append(f"{self.floored_cells} cells on an alluvium floor")
        limits = f", {' + '.join(bounds)}" if bounds else ", unbounded"
        return f"{self.sediment.summary()}, every {self.interval_s:g} s{limits}"

    def as_attrs(self) -> dict:
        """Static description of the morphology configuration (§7.2 ``.zattrs``)."""
        return {
            **self.sediment.as_attrs(),
            "interval_s": self.interval_s,
            "frozen_cells": self.frozen_cells,
            "floored_cells": self.floored_cells,
        }
