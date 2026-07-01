"""Boundary conditions (M1 closed; M3 open / transmissive, HANDOFF §8).

**Closed / reflective** (M1): zero normal flux on every domain-edge face. On the
staggered grid the left/right ``qx`` boundary columns (``j = 0`` and ``j = nx``)
and the top/bottom ``qy`` boundary rows (``i = 0`` and ``i = ny``) are held at
zero. The local-inertial kernels only ever write *interior* faces, so these edge
faces already stay at their initialised zero -- ``apply_closed_bc`` re-asserts it
explicitly, in one named place.

**Open / transmissive (free-outflow)** (M3): water may leave the domain. This is
implemented as a **post-hoc, self-capping sink** rather than a live boundary
face, for a specific safety reason: the M1 donor flux limiter
(:func:`solver.core.local_inertial.limit_qx`) scales only *interior* faces, so a
non-zero boundary face would drain edge cells **unlimited** and go negative on
steep terrain. Instead, after the (fully closed) interior update:

  * extrapolate the discharge at the nearest interior face to the edge
    (zero-gradient on ``q``), keeping only the *outflow* component;
  * remove ``min(dt/dx * q_out, h_edge)`` from the edge cell -- **capped** at the
    available depth, so depth can never go negative;
  * bank the removed depth in ``loss_cum`` (one writer per cell -> deterministic;
    corner cells get two additive edge launches).

Because the closed BC already zeroed the boundary face, ``update_h`` never counted
this water, so banking it in ``loss_cum`` (the ledger's outflow, §8) is exact and
not double-counted. Closed-only runs never launch these kernels, so they are
bitwise-unchanged.
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


@wp.func
def _drain(h_edge: wp.float32, q_out: wp.float32, dxdt: wp.float32) -> wp.float32:
    """New edge depth after removing an (outflow-only, capped) face flux.

    ``removed = min(dt/dx * q_out, h_edge)`` so the depth never goes negative; the
    caller banks the exact loss ``f64(h_edge) - f64(h_new)`` in float64.
    """
    removed = dxdt * q_out  # dt/dx * outward discharge per unit width
    if removed > h_edge:
        removed = h_edge
    if removed < 0.0:
        removed = 0.0
    return h_edge - removed


@wp.kernel
def _open_out_west(
    h: wp.array2d(dtype=wp.float32),
    qx: wp.array2d(dtype=wp.float32),
    loss_cum: wp.array2d(dtype=wp.float64),
    dxdt: wp.float32,
):
    """West edge (col 0): water leaving flows -x; extrapolate face ``qx[i, 1]``."""
    i = wp.tid()
    avail = h[i, 0]
    h_new = _drain(avail, wp.max(-qx[i, 1], 0.0), dxdt)
    h[i, 0] = h_new
    loss_cum[i, 0] = loss_cum[i, 0] + (wp.float64(avail) - wp.float64(h_new))


@wp.kernel
def _open_out_east(
    h: wp.array2d(dtype=wp.float32),
    qx: wp.array2d(dtype=wp.float32),
    loss_cum: wp.array2d(dtype=wp.float64),
    dxdt: wp.float32,
    nx: wp.int32,
):
    """East edge (col nx-1): water leaving flows +x; extrapolate face ``qx[i, nx-1]``."""
    i = wp.tid()
    avail = h[i, nx - 1]
    h_new = _drain(avail, wp.max(qx[i, nx - 1], 0.0), dxdt)
    h[i, nx - 1] = h_new
    loss_cum[i, nx - 1] = loss_cum[i, nx - 1] + (wp.float64(avail) - wp.float64(h_new))


@wp.kernel
def _open_out_north(
    h: wp.array2d(dtype=wp.float32),
    qy: wp.array2d(dtype=wp.float32),
    loss_cum: wp.array2d(dtype=wp.float64),
    dxdt: wp.float32,
):
    """North edge (row 0): water leaving flows -y; extrapolate face ``qy[1, j]``."""
    j = wp.tid()
    avail = h[0, j]
    h_new = _drain(avail, wp.max(-qy[1, j], 0.0), dxdt)
    h[0, j] = h_new
    loss_cum[0, j] = loss_cum[0, j] + (wp.float64(avail) - wp.float64(h_new))


@wp.kernel
def _open_out_south(
    h: wp.array2d(dtype=wp.float32),
    qy: wp.array2d(dtype=wp.float32),
    loss_cum: wp.array2d(dtype=wp.float64),
    dxdt: wp.float32,
    ny: wp.int32,
):
    """South edge (row ny-1): water leaving flows +y; extrapolate face ``qy[ny-1, j]``."""
    j = wp.tid()
    avail = h[ny - 1, j]
    h_new = _drain(avail, wp.max(qy[ny - 1, j], 0.0), dxdt)
    h[ny - 1, j] = h_new
    loss_cum[ny - 1, j] = loss_cum[ny - 1, j] + (wp.float64(avail) - wp.float64(h_new))


def apply_open_outflow(state: State, dt: float) -> None:
    """Drain water through the open edges (post-interior sink; see module docstring).

    Launched once per open edge; needs ``state.loss_cum`` armed (done by
    :meth:`State.set_open_boundaries`). Requires ``nx``/``ny`` >= 2 so a nearest
    interior face exists.
    """
    g = state.grid
    dxdt = float(dt) / float(g.dx)
    edges = state.open_edges
    if "west" in edges and g.nx > 1:
        wp.launch(
            _open_out_west,
            dim=g.ny,
            inputs=[state.h, state.qx, state.loss_cum, dxdt],
            device=state.device,
        )
    if "east" in edges and g.nx > 1:
        wp.launch(
            _open_out_east,
            dim=g.ny,
            inputs=[state.h, state.qx, state.loss_cum, dxdt, g.nx],
            device=state.device,
        )
    if "north" in edges and g.ny > 1:
        wp.launch(
            _open_out_north,
            dim=g.nx,
            inputs=[state.h, state.qy, state.loss_cum, dxdt],
            device=state.device,
        )
    if "south" in edges and g.ny > 1:
        wp.launch(
            _open_out_south,
            dim=g.nx,
            inputs=[state.h, state.qy, state.loss_cum, dxdt, g.ny],
            device=state.device,
        )
