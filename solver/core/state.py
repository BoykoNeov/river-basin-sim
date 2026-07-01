"""Solver state fields (M1, HANDOFF §6, §8).

Container for the mutable simulation fields on the staggered grid defined in
:mod:`solver.core.grid`. Fields are **float32** Warp arrays on the target device
(HANDOFF §2: float32 GPU fields; the float64 protection is confined to the
mass-balance accumulator, which lives host-side in :mod:`solver.core.massbalance`).

State carried between steps:
  * ``h``   -- water depth at cell centres, ``(ny, nx)``
  * ``z``   -- static bed elevation at cell centres, ``(ny, nx)``
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

from dataclasses import dataclass

import numpy as np
import warp as wp

from solver.core.grid import Grid


@dataclass
class State:
    """Mutable float32 fields for one simulation, on ``device``."""

    grid: Grid
    device: str
    h: wp.array
    z: wp.array
    qx: wp.array
    qy: wp.array
    eta: wp.array
    beta: wp.array  # (ny, nx) per-cell outflow-limiter factor in [0, 1]
    h_max: wp.array  # (1,) float32 scratch for the timestep reduction

    @classmethod
    def from_bed(
        cls,
        bed: np.ndarray,
        dx: float,
        *,
        depth: np.ndarray | float = 0.0,
        device: str = "cpu",
    ) -> State:
        """Build a state from a bed-elevation array (metres, row-major ``(y, x)``).

        ``depth`` seeds the initial water depth ``h`` -- either a scalar (uniform)
        or a full ``(ny, nx)`` array (e.g. a dam-break step). Discharges start at
        zero (fluid at rest).
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

        eta0 = (h0 + bed).astype(np.float32)
        return cls(
            grid=grid,
            device=device,
            h=wp.array(h0, dtype=wp.float32, device=device),
            z=wp.array(bed, dtype=wp.float32, device=device),
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

        Reconstructs velocity by averaging the two bounding face discharges and
        dividing by depth (HANDOFF §7.2 emits cell-centred ``u, v``). Never
        divides by an unguarded depth.
        """
        from solver.core.grid import H_DRY

        h = self.h.numpy()
        qx = self.qx.numpy()
        qy = self.qy.numpy()
        qx_c = 0.5 * (qx[:, :-1] + qx[:, 1:])  # -> (ny, nx)
        qy_c = 0.5 * (qy[:-1, :] + qy[1:, :])  # -> (ny, nx)
        wet = h >= H_DRY
        u = np.zeros_like(h)
        v = np.zeros_like(h)
        np.divide(qx_c, h, out=u, where=wet)
        np.divide(qy_c, h, out=v, where=wet)
        return u.astype(np.float32), v.astype(np.float32)
