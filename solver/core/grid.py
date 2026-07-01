"""Staggered uniform-raster grid geometry (M1, HANDOFF §6, §8).

The local-inertial scheme (Bates et al. 2010) lives on a **staggered** (Arakawa
C) grid: water depth at cell centres, discharge on cell faces. Pinning the
layout and the sign convention here once -- and referencing this docstring from
the kernels -- is deliberate: staggered index/offset errors are *the* classic
shallow-water bug.

Layout (a ``ny`` x ``nx`` grid of square cells, side ``dx`` metres)::

    cell-centred, shape (ny, nx):   h  (depth)     z  (bed)     eta = h + z
    x-faces,      shape (ny, nx+1): qx  (discharge per unit width, m^2/s)
    y-faces,      shape (ny+1, nx): qy

Face indexing:
  * ``qx[i, j]`` is the flux on the vertical face between cells ``(i, j-1)`` and
    ``(i, j)``.  So cell ``(i, j)`` has a *left* face ``qx[i, j]`` and a *right*
    face ``qx[i, j+1]``.  Columns ``j = 0`` and ``j = nx`` are the domain's
    left/right boundary faces.
  * ``qy[i, j]`` is the flux on the horizontal face between cells ``(i-1, j)``
    and ``(i, j)``.  Cell ``(i, j)`` has a *top* face ``qy[i, j]`` and a *bottom*
    face ``qy[i+1, j]``.  Rows ``i = 0`` and ``i = ny`` are the top/bottom
    boundary faces.

Sign convention (fixed for the whole solver):
  * **positive ``qx`` flows +x** (left -> right, increasing column ``j``)
  * **positive ``qy`` flows +y** (top -> bottom, increasing row ``i``)

Continuity for an interior cell ``(i, j)`` is then the net inflow (flux in
through the left/top face minus flux out through the right/bottom face)::

    h[i,j] += dt/dx * ( qx[i,j] - qx[i,j+1] + qy[i,j] - qy[i+1,j] ) + R*dt

Row ``i`` increases downward (north -> south) to match the row-major
``(y, x)`` raster the pipeline writes; this is a bookkeeping choice, not physics
(the scheme is isotropic).
"""

from __future__ import annotations

from dataclasses import dataclass

# Standard gravity (m/s^2). Single source of truth for the solver numerics.
GRAVITY = 9.81

# Dry-cell threshold (m): below this a cell/face is treated as dry -- zero
# velocity, zero face flux -- guarding the friction denominator and velocity
# reconstruction against divide-by-near-zero (HANDOFF §8 wetting/drying).
H_DRY = 1.0e-3


@dataclass(frozen=True)
class Grid:
    """Geometry of a uniform square-cell raster.

    ``ny`` rows x ``nx`` columns of cells, each ``dx`` metres square. Holds only
    shape + spacing; the mutable fields live in :class:`solver.core.state.State`.
    """

    ny: int
    nx: int
    dx: float

    @property
    def shape(self) -> tuple[int, int]:
        """Cell-centred field shape ``(ny, nx)``."""
        return (self.ny, self.nx)

    @property
    def qx_shape(self) -> tuple[int, int]:
        """x-face field shape ``(ny, nx + 1)``."""
        return (self.ny, self.nx + 1)

    @property
    def qy_shape(self) -> tuple[int, int]:
        """y-face field shape ``(ny + 1, nx)``."""
        return (self.ny + 1, self.nx)

    @property
    def cell_area(self) -> float:
        """Plan area of one cell (m^2)."""
        return self.dx * self.dx

    @property
    def n_cells(self) -> int:
        return self.ny * self.nx
