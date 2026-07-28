"""Reservoir operations -- the slow process that makes the scheduler real (M5).

HANDOFF §8: *"slow processes (reservoir daily rules, sediment morphology on a long
clock) advance at sync points via operator splitting."* This module is that, for
reservoirs, and it is deliberately the **first** consumer of
:class:`solver.scheduler.SlowProcess` so the multi-rate seam is exercised by
something real rather than asserted by a unit test.

Two halves, per M5 plan §1.2 (*a dam is geometry plus a rule*):

* :func:`apply_barriers` raises the bed to the crest at the structure's cells,
  **once, before stepping**. Impoundment and overtopping are then ordinary
  shallow-water physics the validated schemes already handle -- no new momentum
  term, no face-level special case, nothing to re-validate. A ``levee`` is only
  this.
* :class:`ReservoirOperator` is the release rule. At each slow-clock activation it
  reads the pool stage, asks the rule for a discharge ``Q``, and moves
  ``V = Q * dt_slow`` from the pool to the outlet in **one** operator-split step.
  Between activations it does nothing at all, and the flood scheme sub-cycles
  freely -- which is the whole point of the multi-rate design.

  *Delivering a whole interval at once is the cost of that splitting*, and it is
  visible: 60 m^3/s over a 900 s interval is 54,000 m^3, which arriving in a single
  40 m cell is a 34 m column -- absurd as an instantaneous state even though it
  drains within the interval. So the outlet is a **box**, and spreading the release
  over a reach (rather than a point) is the recommended configuration; a single cell
  is just the 1x1 case.

**Mass exactness (the part that is easy to get wrong).** The transfer is internal,
so the global ledger must see *zero* net change. Three things make that true to the
bit rather than approximately:

1. the withdrawal kernel banks each cell's **actual float32 depth change**
   (``f64(before) - f64(after)``) into a float64 accumulator -- the same idiom M3's
   infiltration uses -- so the host learns exactly what left, not what was asked for;
2. the requested volume is capped by what the pool holds, so a rule that outruns its
   reservoir delivers less rather than inventing water;
3. the delivery kernel measures the depth the outlet cell actually gained (float32
   addition rounds too) and banks the shortfall into ``loss_cum``, so
   ``removed - delivered`` is accounted rather than silently lost.

Determinism (HANDOFF §8/§12): activations are exact multiples of ``interval_s`` on
the simulated clock; the pool scale factor is a host-side float64 sum of the float32
field; every kernel writes each cell from exactly one thread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from solver.core.grid import H_DRY
from solver.core.state import State
from solver.io.config import Structure
from solver.scheduler import SlowProcess


def apply_barriers(bed: np.ndarray, structures: list[Structure]) -> np.ndarray:
    """Raise the bed to each structure's crest at its cells (returns a new array).

    Elevations must already be in the **stepping** datum (see
    :func:`solver.run.shift_for_datum`). ``max`` rather than assignment, so a
    barrier never *lowers* terrain -- a crest below the natural ground is a no-op,
    which is the physically sensible reading of "a dam this high here".
    """
    if not structures:
        return bed
    out = np.array(bed, dtype=np.float32, copy=True)
    ny, nx = out.shape
    for s in structures:
        for i, j in s.cells:
            if not (0 <= i < ny and 0 <= j < nx):
                raise ValueError(
                    f"structure '{s.name}': cell {(i, j)} is outside the {ny}x{nx} grid"
                )
            out[i, j] = max(float(out[i, j]), float(s.crest_m))
    return out


@wp.kernel
def _withdraw_pool(
    h: wp.array2d(dtype=wp.float32),
    taken: wp.array2d(dtype=wp.float64),
    row0: wp.int32,
    col0: wp.int32,
    scale: wp.float32,
):
    """Remove ``scale * h`` from each pool cell, banking the exact depth removed.

    Withdrawing in proportion to depth (rather than levelling the pool, or draining
    one intake cell) keeps the operation local, positive and cap-free: ``scale`` is
    at most 1, so no cell can go negative, and a shallow cell contributes little.
    ``taken`` is a pool-sized float64 scratch -- summed on the host to learn the
    volume that *actually* left, float32 rounding included.
    """
    a, b = wp.tid()
    i = row0 + a
    j = col0 + b
    avail = h[i, j]
    h_new = avail - avail * scale
    h[i, j] = h_new
    taken[a, b] = wp.float64(avail) - wp.float64(h_new)


@wp.kernel
def _deliver_release(
    h: wp.array2d(dtype=wp.float32),
    loss_cum: wp.array2d(dtype=wp.float64),
    row0: wp.int32,
    col0: wp.int32,
    add_depth: wp.float64,
):
    """Add the released depth evenly over the outlet box, banking the shortfall.

    ``h + d`` rounds in float32, so the depth a cell actually gained is not exactly
    its share of what was removed from the pool. Each cell banks its own difference
    in ``loss_cum`` (a real, tiny, *accounted* loss) instead of quietly unbalancing
    the ledger; summed over the box that is exactly ``removed - delivered``.
    """
    a, b = wp.tid()
    i = row0 + a
    j = col0 + b
    avail = h[i, j]
    h_new = avail + wp.float32(add_depth)
    h[i, j] = h_new
    delivered = wp.float64(h_new) - wp.float64(avail)
    loss_cum[i, j] = loss_cum[i, j] + (add_depth - delivered)


@dataclass
class ReleaseRecord:
    """One slow-clock activation of a release rule."""

    time: float
    stage: float | None  # pool water-surface elevation (stepping datum), None if dry
    discharge_m3_s: float  # what the rule asked for
    volume_m3: float  # what was actually moved (capped by the pool)

    def as_dict(self) -> dict:
        return {
            "time": self.time,
            "stage": self.stage,
            "discharge_m3_s": self.discharge_m3_s,
            "volume_m3": self.volume_m3,
        }


class ReservoirOperator:
    """Applies one structure's release rule on the slow clock (operator splitting).

    Construct after the :class:`~solver.core.state.State` exists (it arms the
    float64 loss accumulator the delivery shortfall is banked into), then hand
    :meth:`as_slow_process` to the scheduler.
    """

    def __init__(self, structure: Structure, state: State):
        if structure.release_rule == "none":
            raise ValueError(f"structure '{structure.name}' has no release rule to operate")
        self.structure = structure
        self.state = state
        self.records: list[ReleaseRecord] = []

        ny, nx = state.grid.shape
        r0, c0, r1, c1 = structure.pool
        if not (0 <= r0 <= r1 < ny and 0 <= c0 <= c1 < nx):
            raise ValueError(
                f"structure '{structure.name}': pool {structure.pool} is outside the {ny}x{nx} grid"
            )
        o0, p0, o1, p1 = structure.outlet
        if not (0 <= o0 <= o1 < ny and 0 <= p0 <= p1 < nx):
            raise ValueError(
                f"structure '{structure.name}': outlet {structure.outlet} is outside the "
                f"{ny}x{nx} grid"
            )
        self._pool = (r0, c0, r1, c1)
        self._pool_shape = (r1 - r0 + 1, c1 - c0 + 1)
        self._outlet = (o0, p0)
        self._outlet_shape = (o1 - o0 + 1, p1 - p0 + 1)
        self._outlet_cells = self._outlet_shape[0] * self._outlet_shape[1]
        self._taken = wp.zeros(self._pool_shape, dtype=wp.float64, device=state.device)
        # The stage gauge: the lowest-bed cell in the pool box (see pool_stage).
        pool_z = state.z.numpy()[r0 : r1 + 1, c0 : c1 + 1]
        a, b = np.unravel_index(int(np.argmin(pool_z)), pool_z.shape)
        self._gauge = (r0 + int(a), c0 + int(b))
        state.arm_loss_accumulator()

    # --- the SlowProcess interface -------------------------------------------- #
    def as_slow_process(self) -> SlowProcess:
        """Wrap this operator as a scheduler slow process at its own cadence."""
        return SlowProcess(
            name=f"reservoir:{self.structure.name}",
            interval=self.structure.interval_s,
            advance=self.advance,
        )

    def pool_stage(self) -> float | None:
        """Water-surface elevation at the pool's gauge cell (m), or ``None`` if dry.

        The gauge is a **stilling well at the deepest point of the pool** -- the
        lowest-bed cell in the box, fixed at construction -- and the reading is
        ``z + h`` there. For a still pool that is exactly its level, and it is what a
        real reservoir gauge measures.

        It is deliberately *not* ``max(z + h)`` over the box, which the M5 demo caught
        being badly wrong: a pool box that spans the valley walls (as any realistic one
        does) reports the surface of a millimetre of water stranded high on a wall, so
        a draw-down rule keeps releasing at full discharge from a reservoir that has
        already emptied. The deepest cell cannot be fooled that way: if it is dry,
        there is no pool.
        """
        gi, gj = self._gauge
        h = float(self.state.h.numpy()[gi, gj])
        if h <= H_DRY:
            return None
        return float(self.state.z.numpy()[gi, gj]) + h

    def advance(self, t: float, dt_slow: float) -> ReleaseRecord:
        """Move one slow interval's release from the pool to the outlet.

        Called by the run loop at a scheduler sync point with the exact elapsed
        simulated time since this operator last ran. Records what happened (the
        series ends up in the store's ``.zattrs``) and returns it.
        """
        stage = self.pool_stage()
        q = self.structure.discharge_at(stage)
        volume = 0.0
        if q > 0.0:
            volume = self._transfer(q * float(dt_slow))
        rec = ReleaseRecord(time=float(t), stage=stage, discharge_m3_s=q, volume_m3=volume)
        self.records.append(rec)
        return rec

    # --- the mass-exact transfer ----------------------------------------------- #
    def _transfer(self, requested_m3: float) -> float:
        """Move up to ``requested_m3`` from the pool to the outlet; return what moved."""
        st = self.state
        area = st.grid.cell_area
        r0, c0, r1, c1 = self._pool
        pool_h = st.h.numpy()[r0 : r1 + 1, c0 : c1 + 1].astype(np.float64)
        stored = float(pool_h.sum()) * area
        if stored <= 0.0:
            return 0.0
        # Cap at what the pool holds: a rule that outruns its reservoir delivers
        # less, it never invents water.
        scale = min(1.0, requested_m3 / stored)

        wp.launch(
            _withdraw_pool,
            dim=self._pool_shape,
            inputs=[st.h, self._taken, r0, c0, np.float32(scale)],
            device=st.device,
        )
        removed_m3 = float(self._taken.numpy().sum()) * area  # exactly what left, in f64
        if removed_m3 <= 0.0:
            return 0.0
        wp.launch(
            _deliver_release,
            dim=self._outlet_shape,
            inputs=[
                st.h,
                st.loss_cum,
                self._outlet[0],
                self._outlet[1],
                removed_m3 / area / self._outlet_cells,
            ],
            device=st.device,
        )
        return removed_m3

    @property
    def series(self) -> list[dict]:
        """The release history, for the canonical store's ``.zattrs``."""
        return [r.as_dict() for r in self.records]


def build_operators(structures: list[Structure], state: State) -> list[ReservoirOperator]:
    """Build an operator for every structure that carries a release rule."""
    return [ReservoirOperator(s, state) for s in structures if s.release_rule != "none"]
