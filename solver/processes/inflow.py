"""Inflow hydrographs -- prescribed discharge point sources (M3, HANDOFF §8).

An inflow hydrograph injects a time-varying discharge ``Q(t)`` (m^3/s) into a
named cell -- a river mouth entering the domain, a tributary, a gauged boundary.
We model it as a **cell source** (not an edge-flux BC): each step adds
``Q(t)*dt`` cubic metres to the target cell, i.e. ``h += Q(t)*dt / cell_area``.

Why a cell source and not a boundary flux:
  * mass bookkeeping is trivial and exact -- the injected volume *is* the ledger
    inflow, with no divergence/orientation subtleties at the edge;
  * inflow can enter anywhere (mid-domain river mouth), which is what real
    hydrographs need;
  * it composes cleanly with the closed/open boundary handling.

Determinism (§8/§12): the discharge is sampled at the step **midpoint**
(``t + dt/2``) and the *same* volume is used for both the injection and the ledger
accumulation, so the two agree bit-for-bit regardless of interpolation. Clamping
steps to hydrograph breakpoints (done by the caller via :meth:`breakpoints`) keeps
the sampled curve faithful to sharp peaks; it is not required for mass balance.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from solver.core.grid import Grid
from solver.core.state import State
from solver.io.config import Inflow


@wp.kernel
def _inject_point_sources(
    h: wp.array2d(dtype=wp.float32),
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    add_depth: wp.array(dtype=wp.float32),
):
    """Add ``add_depth[k]`` metres to cell ``(rows[k], cols[k])`` for each source."""
    k = wp.tid()
    h[rows[k], cols[k]] = h[rows[k], cols[k]] + add_depth[k]


class InflowInjector:
    """Applies a set of :class:`~solver.io.config.Inflow` hydrographs each step."""

    def __init__(self, inflows: list[Inflow], grid: Grid, device: str):
        self.inflows = list(inflows)
        self.grid = grid
        self.device = device
        self.cell_area = grid.cell_area
        ny, nx = grid.shape
        rows, cols, seen = [], [], set()
        for inf in self.inflows:
            i, j = inf.cell
            if not (0 <= i < ny and 0 <= j < nx):
                raise ValueError(f"inflow cell {inf.cell} is outside the {ny}x{nx} grid")
            # One thread per entry does a non-atomic h[cell] += ...; two entries on
            # the same cell would race (lost updates, nondeterministic -- breaks the
            # §12 determinism invariant). Reject; merge the hydrographs upstream.
            if inf.cell in seen:
                raise ValueError(
                    f"duplicate inflow cell {inf.cell}; merge their hydrographs into "
                    "one [[inflow]] entry (concurrent same-cell sources would race)"
                )
            seen.add(inf.cell)
            rows.append(i)
            cols.append(j)
        self._rows = wp.array(np.asarray(rows, dtype=np.int32), dtype=wp.int32, device=device)
        self._cols = wp.array(np.asarray(cols, dtype=np.int32), dtype=wp.int32, device=device)
        self._add = wp.zeros(len(self.inflows), dtype=wp.float32, device=device)

    def breakpoints(self) -> list[float]:
        """Sorted unique hydrograph knot times (for step-clamping by the caller)."""
        pts = {t for inf in self.inflows for t in inf.breakpoints}
        return sorted(pts)

    def apply(self, state: State, t: float, dt: float) -> float:
        """Inject the midpoint discharge over ``[t, t+dt]``; return the volume (m^3).

        The returned volume is exactly what was added to the field, so the caller
        can hand it straight to the mass ledger as inflow.
        """
        if not self.inflows:
            return 0.0
        t_mid = t + 0.5 * dt
        vols = np.array([inf.discharge_at(t_mid) * dt for inf in self.inflows], dtype=np.float64)
        add_depth = (vols / self.cell_area).astype(np.float32)
        self._add.assign(add_depth)
        wp.launch(
            _inject_point_sources,
            dim=len(self.inflows),
            inputs=[state.h, self._rows, self._cols, self._add],
            device=self.device,
        )
        # Sum the float32 depths actually applied (times area) so the ledger matches
        # the field to float32 precision, not the float64 request.
        return float(add_depth.astype(np.float64).sum()) * self.cell_area
