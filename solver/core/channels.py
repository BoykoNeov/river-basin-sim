"""Sub-grid channels: geometry + storage curve (M6, HANDOFF §2 scale path).

At reach scale a cell is 50-200 m and a river is 10-40 m wide. Resolving the
channel is what makes a basin unaffordable; erasing it is what makes a coarse run
wrong -- the river's conveyance and its storage both vanish into a wide flat cell.
A **sub-grid channel** keeps both, parametrically: each cell may carry a channel of

``width`` w (m, ``0 < w <= dx``)   narrower than the cell,
``depth`` d (m)                    bank-full depth **below** the floodplain bed,
``manning`` n_ch                   its own roughness (a channel is smoother than
                                   its floodplain).

so the cell's bed is two elevations: the floodplain bed ``z`` (the DEM value,
unchanged) and the channel bed ``z_ch = z - d``.

**The state variable does not change.** ``h`` stays what it is everywhere else in
the solver -- water volume per unit plan area, ``V = h·dx²`` -- so continuity, the
donor limiter and the float64 mass ledger are untouched, and **mass conservation
stays exact to float round-off by construction** (the ledger cannot tell whether a
channel exists). The channel enters through the map ``h -> eta`` alone, which is
what the momentum kernels read::

    h_bf = w·d/dx                       cell-mean depth at bank full
    h <= h_bf :  eta = z - d + h·dx/w   all water in the channel (dx/w amplifies)
    h >  h_bf :  eta = z + (h - h_bf)   bank full, spread over the whole cell

Continuous at ``h = h_bf`` (both give ``eta = z``), strictly monotone, exactly
invertible, and ``w = 0`` collapses to ``eta = z + h`` -- the pre-M6 relation, bit
for bit. The water *column* (``eta - z_ch``) is what sets the wave speed, so the
timestep reduction uses :func:`column_depth`, not ``h``.

**Honesty (M6 plan §4).** This is a parameterization, not resolved physics: it
restores conveyance and storage, not planform, meander routing or overbank
velocity structure. And the geometry is data -- hydraulic-geometry coefficients are
regional calibration inputs, so a coarse run is only as good as the channel fields
it is fed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import warp as wp

from solver.core.grid import Grid
from solver.core.state import State

# A channel narrower than dx/WIDTH_WARN_RATIO turns a millimetre of cell-mean depth
# into a deep column and collapses the timestep; almost always a data artifact.
WIDTH_WARN_RATIO = 1000.0


class ChannelError(ValueError):
    """Sub-grid channel geometry is inconsistent with the grid or with itself."""


@wp.func
def eta_subgrid(
    h: wp.float32, z: wp.float32, w: wp.float32, d: wp.float32, dx: wp.float32
) -> wp.float32:
    """Water-surface elevation from cell-mean depth through the storage curve."""
    eta = h + z
    if w > 0.0 and d > 0.0:
        h_bf = w * d / dx
        if h <= h_bf:
            eta = z - d + h * dx / w
        else:
            eta = z + (h - h_bf)
    return eta


@wp.func
def column_depth(h: wp.float32, w: wp.float32, d: wp.float32, dx: wp.float32) -> wp.float32:
    """Depth of the actual water column (``eta - z_ch``), which sets the wave speed."""
    col = h
    if w > 0.0 and d > 0.0:
        h_bf = w * d / dx
        if h <= h_bf:
            col = h * dx / w
        else:
            col = d + (h - h_bf)
    return col


def eta_from_h(
    h: np.ndarray | float,
    z: np.ndarray | float,
    w: np.ndarray | float,
    d: np.ndarray | float,
    dx: float,
) -> np.ndarray:
    """Host-side storage curve (the reference :func:`eta_subgrid` is tested against)."""
    h, z, w, d = (np.asarray(v, dtype=np.float64) for v in (h, z, w, d))
    has = (w > 0.0) & (d > 0.0)
    h_bf = np.where(has, w * d / dx, 0.0)
    in_channel = has & (h <= h_bf)
    return np.where(
        has,
        np.where(in_channel, z - d + h * dx / np.where(has, w, 1.0), z + (h - h_bf)),
        z + h,
    )


def h_from_eta(
    eta: np.ndarray | float,
    z: np.ndarray | float,
    w: np.ndarray | float,
    d: np.ndarray | float,
    dx: float,
) -> np.ndarray:
    """Inverse storage curve: cell-mean depth holding a given water surface."""
    eta, z, w, d = (np.asarray(v, dtype=np.float64) for v in (eta, z, w, d))
    has = (w > 0.0) & (d > 0.0)
    h_bf = np.where(has, w * d / dx, 0.0)
    below = has & (eta <= z)
    return np.clip(
        np.where(
            has,
            np.where(below, (eta - (z - d)) * w / dx, h_bf + (eta - z)),
            eta - z,
        ),
        0.0,
        None,
    )


@dataclass
class ChannelState:
    """Device-side sub-grid channel geometry + the split face discharges.

    The face arrays are the reason this is state and not a lookup: momentum has
    memory, and the channel and floodplain flows carry their own. ``qx_ch`` is
    discharge per unit **channel** width (m²/s), ``qx_fp`` per unit floodplain
    width; the scheme recombines them into the total per-cell-width flux ``qx``
    that continuity, the limiter and the boundaries already speak.
    """

    w: wp.array  # (ny, nx) channel width, m
    d: wp.array  # (ny, nx) bank-full depth below the floodplain bed, m
    n: wp.array  # (ny, nx) channel Manning roughness
    qx_ch: wp.array  # (ny, nx+1)
    qx_fp: wp.array  # (ny, nx+1)
    qy_ch: wp.array  # (ny+1, nx)
    qy_fp: wp.array  # (ny+1, nx)
    width_max: float  # host-side diagnostics (validated at arm time)
    depth_max: float
    channel_cells: int

    def summary(self) -> str:
        return (
            f"{self.channel_cells} channel cells, width <= {self.width_max:.1f} m, "
            f"bank-full depth <= {self.depth_max:.2f} m"
        )

    def as_attrs(self) -> dict:
        return {
            "channel_cells": self.channel_cells,
            "width_max_m": self.width_max,
            "depth_max_m": self.depth_max,
        }


def validate_geometry(
    width: np.ndarray,
    depth: np.ndarray,
    manning: np.ndarray,
    grid: Grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check and normalise channel fields against the grid (M6 plan §4).

    Rejects what cannot mean anything (a channel wider than its cell, a negative
    width or depth, a channel cell with no roughness), normalises "no channel" to
    ``w = d = 0`` in both fields so the kernels have a single test, and warns about
    a width so small it will collapse the timestep.
    """
    ny, nx = grid.shape
    for name, arr in (("width", width), ("depth", depth), ("manning", manning)):
        if arr.shape != (ny, nx):
            raise ChannelError(f"channel {name} shape {arr.shape} != grid {(ny, nx)}")
    w = np.ascontiguousarray(width, dtype=np.float32).copy()
    d = np.ascontiguousarray(depth, dtype=np.float32).copy()
    n = np.ascontiguousarray(manning, dtype=np.float32).copy()

    if not np.isfinite(w).all() or not np.isfinite(d).all():
        raise ChannelError("channel width/depth fields contain non-finite values")
    if (w < 0).any() or (d < 0).any():
        raise ChannelError("channel width and depth must be >= 0 everywhere")
    too_wide = w > grid.dx
    if too_wide.any():
        worst = float(w[too_wide].max())
        raise ChannelError(
            f"{int(too_wide.sum())} cells have a channel wider than the cell "
            f"(max {worst:.1f} m > dx {grid.dx:.1f} m); a sub-grid channel must fit "
            "inside its cell -- resolve the river or coarsen less"
        )
    has = (w > 0) & (d > 0)
    half = (w > 0) != (d > 0)
    if half.any():
        raise ChannelError(
            f"{int(half.sum())} cells give a channel width without a bank-full depth "
            "(or the reverse); a channel needs both, and 'no channel' is width = depth = 0"
        )
    if has.any() and (n[has] <= 0).any():
        raise ChannelError("channel cells need manning > 0")
    # "No channel" is a single test in the kernels: force both fields to 0.
    w[~has] = 0.0
    d[~has] = 0.0
    if has.any():
        w_min = float(w[has].min())
        if w_min < grid.dx / WIDTH_WARN_RATIO:
            warnings.warn(
                f"narrowest sub-grid channel is {w_min:g} m in a {grid.dx:g} m cell "
                f"(1:{grid.dx / max(w_min, 1e-12):.0f}); the water column -- and so the "
                "timestep -- is set by dx/w, so this will run very slowly",
                stacklevel=2,
            )
    return w, d, n


def arm_channels(
    state: State,
    width: np.ndarray,
    depth: np.ndarray,
    manning: np.ndarray | None = None,
) -> ChannelState:
    """Attach validated channel geometry (and its face arrays) to ``state``.

    ``manning`` defaults to the cell's floodplain roughness, which makes an armed
    channel with no roughness field a pure geometry change. Arming is what switches
    the local-inertial scheme onto its sub-grid path; a state that is never armed
    runs the M1 kernels untouched.
    """
    g = state.grid
    fp_n = state.n.numpy()
    w, d, n = validate_geometry(
        np.asarray(width, dtype=np.float32),
        np.asarray(depth, dtype=np.float32),
        fp_n if manning is None else np.asarray(manning, dtype=np.float32),
        g,
    )
    dev = state.device
    has = w > 0
    chan = ChannelState(
        w=wp.array(w, dtype=wp.float32, device=dev),
        d=wp.array(d, dtype=wp.float32, device=dev),
        n=wp.array(n, dtype=wp.float32, device=dev),
        qx_ch=wp.zeros(g.qx_shape, dtype=wp.float32, device=dev),
        qx_fp=wp.zeros(g.qx_shape, dtype=wp.float32, device=dev),
        qy_ch=wp.zeros(g.qy_shape, dtype=wp.float32, device=dev),
        qy_fp=wp.zeros(g.qy_shape, dtype=wp.float32, device=dev),
        width_max=float(w.max()),
        depth_max=float(d.max()),
        channel_cells=int(has.sum()),
    )
    state.channels = chan
    return chan
