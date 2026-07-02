"""Solver state fields (M1, HANDOFF §6, §8).

Container for the mutable simulation fields on the staggered grid defined in
:mod:`solver.core.grid`. Fields are **float32** Warp arrays on the target device
(HANDOFF §2: float32 GPU fields; the float64 protection is confined to the
mass-balance accumulator, which lives host-side in :mod:`solver.core.massbalance`).

State carried between steps:
  * ``h``   -- water depth at cell centres, ``(ny, nx)``
  * ``z``   -- static bed elevation at cell centres, ``(ny, nx)``
  * ``n``   -- static Manning roughness at cell centres, ``(ny, nx)`` (M3: may be
    spatially varying; a scalar broadcasts to a uniform field, which is bitwise
    identical to the old scalar path since the face average ``0.5*(n+n) == n``)
  * ``qx``  -- discharge per unit width on x-faces, ``(ny, nx+1)``
  * ``qy``  -- discharge per unit width on y-faces, ``(ny+1, nx)``

Scratch:
  * ``eta`` -- water-surface elevation ``h + z`` at cell centres, recomputed each
    step (kept as a field so the flux kernels read neighbours cheaply)
  * ``h_max`` -- single-element array for the atomic-max depth reduction that
    feeds the deterministic adaptive timestep (order-independent, so atomics stay
    reproducible -- HANDOFF §8, §12).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from solver.core.grid import Grid

# Domain-edge names (row-major (y, x) raster; see solver.core.grid docstring).
_EDGES = ("north", "south", "east", "west")


@dataclass
class State:
    """Mutable float32 fields for one simulation, on ``device``."""

    grid: Grid
    device: str
    h: wp.array
    z: wp.array
    n: wp.array  # (ny, nx) Manning roughness at cell centres
    qx: wp.array
    qy: wp.array
    eta: wp.array
    beta: wp.array  # (ny, nx) per-cell outflow-limiter factor in [0, 1]
    h_max: wp.array  # (1,) float32 scratch for the timestep reduction
    # M4 optional cell-centred conservative momentum (armed only by the HLLC FV
    # scheme; the local-inertial scheme leaves these None and uses qx/qy). See
    # solver.core.hllc.arm_hllc, which also attaches scheme scratch to ``hllc``.
    hu: wp.array | None = None  # (ny, nx) x-momentum h*u
    hv: wp.array | None = None  # (ny, nx) y-momentum h*v
    hllc: object | None = None  # HLLC scheme scratch (solver.core.hllc._HllcScratch)
    # M3 optional source/sink fields (None unless the scenario needs them):
    infil: wp.array | None = None  # (ny, nx) infiltration rate, m/s
    rain: wp.array | None = None  # (ny, nx) spatial rainfall rate, m/s (temporally scaled)
    # (ny, nx) float64 cumulative depth of water removed by local sinks
    # (infiltration, open-boundary outflow); summed at output cadence -> outflow.
    loss_cum: wp.array | None = None
    # Names of open (free-outflow) domain edges, subset of {north,south,east,west}.
    # The local-inertial scheme reads this (its post-interior sink is open-only).
    open_edges: frozenset[str] = frozenset()
    # Full per-edge boundary map {north,south,east,west} -> "closed"|"open". The
    # HLLC scheme needs *both* types: a closed edge is a reflective ghost-cell wall
    # (no through-flux), an open edge is transmissive + mass-banked. Defaults to an
    # all-closed box, so a State built directly by `from_bed` (dam-break, the
    # validation harness) is walled without any config call. `set_open_boundaries`
    # overwrites it from the resolved scenario.
    boundaries: dict[str, str] = field(default_factory=lambda: {e: "closed" for e in _EDGES})

    def set_infiltration(self, infil: np.ndarray) -> None:
        """Attach an infiltration-rate field (m/s) and arm the loss accumulator."""
        ny, nx = self.grid.shape
        if infil.shape != (ny, nx):
            raise ValueError(f"infiltration shape {infil.shape} != grid {(ny, nx)}")
        self.infil = wp.array(np.ascontiguousarray(infil, dtype=np.float32), device=self.device)
        self._ensure_loss_cum()

    def set_rain_field(self, rain: np.ndarray) -> None:
        """Attach a spatial rainfall-rate field (m/s); scaled on/off over time."""
        ny, nx = self.grid.shape
        if rain.shape != (ny, nx):
            raise ValueError(f"rain-field shape {rain.shape} != grid {(ny, nx)}")
        self.rain = wp.array(np.ascontiguousarray(rain, dtype=np.float32), device=self.device)

    def set_open_boundaries(self, boundaries: dict[str, str]) -> None:
        """Record the per-edge boundary map and arm the loss accumulator if needed.

        ``boundaries`` maps edge name -> "closed"|"open" (per :func:`solver.io.config`).
        The full map is stored on ``self.boundaries`` (the HLLC scheme walls closed
        edges and banks open ones); ``open_edges`` is the open subset the LI sink
        reads. Arming ``loss_cum`` is triggered by any open edge (both schemes bank
        their outflow there).
        """
        self.boundaries = {e: boundaries.get(e, "closed") for e in _EDGES}
        self.open_edges = frozenset(e for e, v in self.boundaries.items() if v == "open")
        if self.open_edges:
            self._ensure_loss_cum()

    def _ensure_loss_cum(self) -> None:
        # float64: sink outflow can concentrate at a single edge cell and grow far
        # larger than any per-step increment; a float32 accumulator there would
        # drift (HANDOFF §2/§12 -- the accumulator that judges mass stays float64).
        if self.loss_cum is None:
            self.loss_cum = wp.zeros(self.grid.shape, dtype=wp.float64, device=self.device)

    def loss_volume(self, cell_area: float) -> float:
        """Cumulative sink volume (m^3) so far, float64-summed (0 if unarmed)."""
        if self.loss_cum is None:
            return 0.0
        return float(self.loss_cum.numpy().astype(np.float64).sum()) * cell_area

    @classmethod
    def from_bed(
        cls,
        bed: np.ndarray,
        dx: float,
        *,
        depth: np.ndarray | float = 0.0,
        manning: np.ndarray | float = 0.035,
        device: str = "cpu",
    ) -> State:
        """Build a state from a bed-elevation array (metres, row-major ``(y, x)``).

        ``depth`` seeds the initial water depth ``h`` -- either a scalar (uniform)
        or a full ``(ny, nx)`` array (e.g. a dam-break step). ``manning`` seeds the
        roughness field the same way (scalar broadcasts to uniform). Discharges
        start at zero (fluid at rest).
        """
        bed = np.ascontiguousarray(bed, dtype=np.float32)
        ny, nx = bed.shape
        grid = Grid(ny=ny, nx=nx, dx=float(dx))

        if np.isscalar(depth):
            h0 = np.full((ny, nx), float(depth), dtype=np.float32)
        else:
            h0 = np.ascontiguousarray(depth, dtype=np.float32)
            if h0.shape != (ny, nx):
                raise ValueError(f"depth shape {h0.shape} != bed shape {(ny, nx)}")

        if np.isscalar(manning):
            n0 = np.full((ny, nx), float(manning), dtype=np.float32)
        else:
            n0 = np.ascontiguousarray(manning, dtype=np.float32)
            if n0.shape != (ny, nx):
                raise ValueError(f"manning shape {n0.shape} != bed shape {(ny, nx)}")

        eta0 = (h0 + bed).astype(np.float32)
        return cls(
            grid=grid,
            device=device,
            h=wp.array(h0, dtype=wp.float32, device=device),
            z=wp.array(bed, dtype=wp.float32, device=device),
            n=wp.array(n0, dtype=wp.float32, device=device),
            qx=wp.zeros(grid.qx_shape, dtype=wp.float32, device=device),
            qy=wp.zeros(grid.qy_shape, dtype=wp.float32, device=device),
            eta=wp.array(eta0, dtype=wp.float32, device=device),
            beta=wp.zeros(grid.shape, dtype=wp.float32, device=device),
            h_max=wp.zeros(1, dtype=wp.float32, device=device),
        )

    def depth_numpy(self) -> np.ndarray:
        """Copy the depth field back to host as ``(ny, nx)`` float32."""
        return self.h.numpy()

    def velocities_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centred ``(u, v)`` for output, guarded to 0 where ``h < H_DRY``.

        Scheme-aware (HANDOFF §7.2 emits cell-centred ``u, v`` regardless of the
        interior layout): the HLLC scheme stores cell-centred momentum, so
        ``u = hu/h``, ``v = hv/h``; the local-inertial scheme reconstructs from the
        two bounding face discharges. Never divides by an unguarded depth.
        """
        from solver.core.grid import H_DRY

        h = self.h.numpy()
        wet = h >= H_DRY
        u = np.zeros_like(h)
        v = np.zeros_like(h)
        if self.hu is not None:  # HLLC FV: cell-centred conservative momentum
            np.divide(self.hu.numpy(), h, out=u, where=wet)
            np.divide(self.hv.numpy(), h, out=v, where=wet)
        else:  # local-inertial: average the two bounding face discharges
            qx = self.qx.numpy()
            qy = self.qy.numpy()
            qx_c = 0.5 * (qx[:, :-1] + qx[:, 1:])  # -> (ny, nx)
            qy_c = 0.5 * (qy[:-1, :] + qy[1:, :])  # -> (ny, nx)
            np.divide(qx_c, h, out=u, where=wet)
            np.divide(qy_c, h, out=v, where=wet)
        return u.astype(np.float32), v.astype(np.float32)
