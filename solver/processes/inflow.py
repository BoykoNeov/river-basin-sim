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

**The add is compensated (2026-08-17).** ``h += Q*dt/area`` is a float32 accumulator
run once per source cell per step, exactly like rain. The precision pass scoped point
sources out on evidence that inflow was ~1.3 % of the residual, but that was measured
on a **rain-driven** run and does not transfer to a flood-driven one: on
``reach_alluvial`` these four cells add ``Q*dt/A ~ 0.03`` m onto ~1 m of water while
``outflow_cum`` is exactly 0.0 all run, so every cubic metre of residual is stored
float32 volume weighed against the float64 inflow ledger. Probing the target cells in
float64 around each launch, ``applied - requested`` over the whole run was
**+1.215 m^3** uncompensated -- the field got that much more water than the ledger
banked, of 4.446 M m^3 over 5630 steps -- and **-0.000093 m^3** compensated, four
orders smaller and on the other side of zero, which is a term carrying an
unrepresentable remainder rather than one still losing bits.
Each source entry now carries its own float32 Kahan compensation
term through :func:`solver.core.sources.kahan_add` -- that module's stated requirement
that an accumulator into a float32 field has one definition of "how".

It is worth knowing what this does *not* buy, because the number that motivated it
does not halve twice: the run's total residual goes 4.79e-07 -> 2.66e-07 and stops
there. The rest is flux-divergence and limiter round-off, untouched. And on scenarios
where the source term is not what sets the residual the total moves either way
(``reach_basin`` 5.95e-08 -> 6.00e-08) -- the drift is systematic only while nothing
else writes ``h``, and once continuity rewrites it every step the low-order bits
decorrelate into a random walk. See ``docs/plans/point-source-compensation.md``.

Two things that deliberately did **not** change:

* **The ledger still banks the float32 *request*** (:meth:`InflowInjector.apply`'s
  return), not the field's actual delta. Banking the delta would make the residual
  read ~0 by construction and blind the mass gate to this whole defect class. The
  ledger is the independent witness; compensation makes the field catch up to it.
* **Compensation is not armed off the scenario.** The injector only exists when a
  scenario has ``[[inflow]]`` at all, so "armed" and "has a point source" are the
  same condition and no rain-free, inflow-free run can notice. ``compensated=False``
  exists to reproduce the pre-fix arithmetic for the A/B gates in
  ``test_inflow.py``; it is not a production switch.

**Sub-ULP caveat**, as in :mod:`solver.core.sources`: the compensation term is exact
under Fast2Sum's ``|h| >= |increment|``. The first injection onto a dry cell violates
that and its ``comp`` is then only approximate -- but a dry cell is also where the
add loses nothing, so the term it fails to capture is below the drift being removed.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from solver.core.grid import Grid
from solver.core.sources import kahan_add
from solver.core.state import State
from solver.io.config import Inflow


@wp.kernel
def _inject_point_sources(
    h: wp.array2d(dtype=wp.float32),
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    add_depth: wp.array(dtype=wp.float32),
):
    """Add ``add_depth[k]`` metres to cell ``(rows[k], cols[k])`` for each source.

    The uncompensated add, kept as the control arm of the A/B gates in
    ``test_inflow.py`` (``InflowInjector(..., compensated=False)``) -- a ratio test
    needs the pre-fix arithmetic to measure the fix against. Nothing ships on it.
    """
    k = wp.tid()
    h[rows[k], cols[k]] = h[rows[k], cols[k]] + add_depth[k]


@wp.kernel
def _inject_point_sources_c(
    h: wp.array2d(dtype=wp.float32),
    comp: wp.array(dtype=wp.float32),
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    add_depth: wp.array(dtype=wp.float32),
):
    """Kahan-add ``add_depth[k]`` metres to cell ``(rows[k], cols[k])``.

    ``comp`` is indexed by *source entry*, not by cell, which is what makes one
    thread per entry safe: duplicate cells are rejected in the constructor, so no
    two threads touch the same ``h`` or the same ``comp`` slot. It persists across
    steps -- it is the running debt of low-order bits -- and the kernel keeps being
    launched when the hydrograph reads zero, which is exactly when the outstanding
    debt gets repaid into the field instead of stranded.
    """
    k = wp.tid()
    i = rows[k]
    j = cols[k]
    out = kahan_add(h[i, j], comp[k], add_depth[k])
    h[i, j] = out[0]
    comp[k] = out[1]


class InflowInjector:
    """Applies a set of :class:`~solver.io.config.Inflow` hydrographs each step."""

    def __init__(self, inflows: list[Inflow], grid: Grid, device: str, compensated: bool = True):
        self.inflows = list(inflows)
        self.grid = grid
        self.device = device
        self.cell_area = grid.cell_area
        self.compensated = compensated
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
        # One float32 of Kahan debt per source entry (a handful of floats, whatever
        # the grid size). Owned here rather than armed on `State.h_comp`, which both
        # schemes dispatch their *areal* source kernels on -- arming that for inflow
        # would move rain-free scenarios onto a different code path for no reason.
        self._comp = (
            wp.zeros(len(self.inflows), dtype=wp.float32, device=device) if compensated else None
        )

    def compensation_numpy(self) -> np.ndarray | None:
        """The per-entry Kahan debt (m), or ``None`` when uncompensated.

        Public for the fast-math canary: if ``(t - h) - y`` is ever reassociated to
        zero this array stays all-zero and every other assertion about compensation
        would be measuring an uncompensated add against itself.
        """
        return None if self._comp is None else self._comp.numpy()

    def breakpoints(self) -> list[float]:
        """Sorted unique hydrograph knot times (for step-clamping by the caller)."""
        pts = {t for inf in self.inflows for t in inf.breakpoints}
        return sorted(pts)

    def apply(self, state: State, t: float, dt: float) -> float:
        """Inject the midpoint discharge over ``[t, t+dt]``; return the volume (m^3).

        The returned volume is the float32 depth *requested* of the field, times
        area, so the ledger matches the injection to float32 precision rather than
        to the float64 discharge curve. It is deliberately **not** the field's
        actual delta: with compensation armed the difference between the two is the
        outstanding Kahan debt, and that difference is the quantity the mass gate
        exists to see. Banking the delta instead would zero the residual by
        construction and make the gate blind here.
        """
        if not self.inflows:
            return 0.0
        t_mid = t + 0.5 * dt
        vols = np.array([inf.discharge_at(t_mid) * dt for inf in self.inflows], dtype=np.float64)
        add_depth = (vols / self.cell_area).astype(np.float32)
        self._add.assign(add_depth)
        # Launched every step, including where the hydrograph reads zero: a zero
        # increment repays the outstanding compensation instead of stranding it.
        if self._comp is None:
            wp.launch(
                _inject_point_sources,
                dim=len(self.inflows),
                inputs=[state.h, self._rows, self._cols, self._add],
                device=self.device,
            )
        else:
            wp.launch(
                _inject_point_sources_c,
                dim=len(self.inflows),
                inputs=[state.h, self._comp, self._rows, self._cols, self._add],
                device=self.device,
            )
        return float(add_depth.astype(np.float64).sum()) * self.cell_area
