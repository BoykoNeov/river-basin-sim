"""Boundary conditions (M1: closed only, HANDOFF §8).

M1 ships the **closed / reflective** boundary: zero normal flux on every
domain-edge face. On the staggered grid that means the left/right ``qx`` boundary
columns (``j = 0`` and ``j = nx``) and the top/bottom ``qy`` boundary rows
(``i = 0`` and ``i = ny``) are held at zero.

The local-inertial kernels (:mod:`solver.core.local_inertial`) only ever write
*interior* faces, so these edge faces already stay at their initialised zero --
``apply_closed_bc`` re-asserts it explicitly so the invariant is enforced in one
named place and so future BC types (open / inflow / fixed-stage, M3+) have a hook
to override per edge.
"""

from __future__ import annotations

import warp as wp

from solver.core.state import State


@wp.kernel
def _zero_qx_edges(qx: wp.array2d(dtype=wp.float32), nx: wp.int32):
    i = wp.tid()
    qx[i, 0] = 0.0
    qx[i, nx] = 0.0


@wp.kernel
def _zero_qy_edges(qy: wp.array2d(dtype=wp.float32), ny: wp.int32):
    j = wp.tid()
    qy[0, j] = 0.0
    qy[ny, j] = 0.0


def apply_closed_bc(state: State) -> None:
    """Zero the domain-edge face discharges (closed / reflective BC)."""
    g = state.grid
    wp.launch(_zero_qx_edges, dim=g.ny, inputs=[state.qx, g.nx], device=state.device)
    wp.launch(_zero_qy_edges, dim=g.nx, inputs=[state.qy, g.ny], device=state.device)
