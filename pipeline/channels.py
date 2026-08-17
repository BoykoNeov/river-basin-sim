"""Sub-grid channel geometry from flow accumulation (M6, plan §1.4).

The solver's sub-grid channels (:mod:`solver.core.channels`) are **data**: each
cell carries a channel width and a bank-full depth. This module produces those two
fields for real terrain, from the flow accumulation the M0 conditioning step
already computes, using **downstream hydraulic geometry** -- the empirical
observation that a river's width and depth scale with a power of its drainage
area::

    w = a_w · A^b_w        d = a_d · A^b_d        (A = upstream area, km²)

applied only where ``A >= min_area_km2``; elsewhere there is no channel
(``w = d = 0``) and the cell is pure floodplain.

**These coefficients are regional calibration inputs, not constants of nature.**
The defaults below are a humid-temperate starting point and will be wrong -- by a
factor, not a percent -- for an arid basin, a bedrock gorge or a managed lowland
river. A coarse run is only as good as the channel geometry it is fed (M6 plan §3),
so calibrate them against surveyed cross-sections or a width product before
reporting numbers, and record what you used: the CLI writes the coefficients beside
the fields for exactly that reason.

Outputs are raw little-endian float32 ``.r32`` aligned to the **tile mosaic** (the
run domain, :mod:`solver.io.mosaic`), which is the same field contract M3 uses for
roughness and infiltration.

**Two things here are not what a first reading suggests**, both measured in
``docs/plans/real-dem-reach.md``:

*A D8 network is 8-connected and the solver is not.* A D8 flow path steps to
whichever of eight neighbours is steepest, and on real terrain **48 %** of channel
cells take a diagonal step. The solver's channel faces are N/S/E/W only and face
width is ``min(w_L, w_R)``, so a diagonal step is a **wall**: derived as-is, 40 448
real channel cells form 19 008 rook-connected fragments (largest 37 cells) where the
same mask is 61 components under 8-connectivity. That is a chain of pools, not a
river, and it conserves mass perfectly while failing to convey -- so no gate in this
repo would catch it. :func:`rook_connect` fixes it by carrying the channel through a
corner cell at each diagonal step, and it is on by default.

*The inserted corner cell must inherit the through-path's drainage area, not keep its
own.* A corner cell is beside the river, not on it: its own area is ~450x smaller, so
sizing it normally leaves a **3.6 %** aperture -- a pinhole instead of a wall. So a
cell's channel width stops being a pure function of its own drainage area and becomes
the width of the river passing through it.

CLI::

    uv run python -m pipeline.channels --src data/dem/conditioned \\
        --tiles data/tiles/demo --out data/fields/demo --coarsen 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Humid-temperate downstream hydraulic geometry (see the module docstring: these
# are calibration inputs). w ~ 8 A^0.5 gives a 25 m channel at 10 km2 and an 80 m
# channel at 100 km2; d ~ 0.27 A^0.3 gives 1.1 m and 2.1 m at the same areas.
DEFAULT_WIDTH_COEF = 8.0
DEFAULT_WIDTH_EXP = 0.5
DEFAULT_DEPTH_COEF = 0.27
DEFAULT_DEPTH_EXP = 0.3
# Below this upstream area a cell is hillslope, not river.
DEFAULT_MIN_AREA_KM2 = 1.0

# D8 direction codes, in the order :data:`pipeline.condition.DIRMAP` declares them
# (N, NE, E, SE, S, SW, W, NW). Duplicated rather than imported because
# ``condition`` pulls in the geo extra at import time and this module must stay
# importable without it; ``condition.json`` records the dirmap actually used and
# :func:`channel_fields` prefers that over this default.
DEFAULT_DIRMAP: tuple[int, ...] = (64, 128, 1, 2, 4, 8, 16, 32)

#: ``(row, col)`` deltas matching :data:`DEFAULT_DIRMAP`'s order. Rows increase
#: southward, so "N" is ``-1`` in row.
DIRMAP_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # N
    (-1, 1),  # NE
    (0, 1),  # E
    (1, 1),  # SE
    (1, 0),  # S
    (1, -1),  # SW
    (0, -1),  # W
    (-1, -1),  # NW
)


def d8_offsets(dirmap: tuple[int, ...] | list[int] = DEFAULT_DIRMAP) -> dict[int, tuple[int, int]]:
    """Map D8 direction codes to ``(dr, dc)`` steps, in :data:`DIRMAP_OFFSETS` order."""
    codes = tuple(int(c) for c in dirmap)
    if len(codes) != len(DIRMAP_OFFSETS):
        raise ValueError(f"dirmap must have {len(DIRMAP_OFFSETS)} codes, got {len(codes)}")
    return dict(zip(codes, DIRMAP_OFFSETS, strict=True))


def rook_connect(
    area_km2: np.ndarray,
    flowdir: np.ndarray,
    *,
    dirmap: tuple[int, ...] | list[int] = DEFAULT_DIRMAP,
    min_area_km2: float = DEFAULT_MIN_AREA_KM2,
) -> tuple[np.ndarray, int]:
    """Make the D8 channel network 4-connected, returning ``(area, n_inserted)``.

    At every diagonal step of the flow path, the channel is carried through one of
    the two corner cells -- the one with the larger drainage area, so the inserted
    cell is the more river-like of the pair. **The inserted cell takes the drainage
    area of the river making the step**, not its own: a corner cell sits beside the
    river rather than on it, and its own area is smaller by a median factor of ~450,
    which under ``w_face = min(w_L, w_R)`` would leave a 3.6 % aperture. Inserting
    the cell fixes the topology; inheriting the area is what makes it convey.

    The returned array is the input with those cells raised -- feed it to
    :func:`hydraulic_geometry` exactly as you would the raw area. Cells already at or
    above the through-area are untouched, so the operation is idempotent.
    """
    area = np.array(area_km2, dtype=np.float64, copy=True)
    fdir = np.asarray(flowdir)
    if fdir.shape != area.shape:
        raise ValueError(f"flowdir shape {fdir.shape} != area shape {area.shape}")
    offsets = d8_offsets(dirmap)
    ny, nx = area.shape
    river = area >= float(min_area_km2)
    before = int(np.count_nonzero(river))

    for code, (dr, dc) in offsets.items():
        if dr == 0 or dc == 0:
            continue  # a cardinal step already has a face
        rr, cc = np.nonzero(river & (fdir == code))
        if rr.size == 0:
            continue
        tr, tc = rr + dr, cc + dc
        keep = (tr >= 0) & (tr < ny) & (tc >= 0) & (tc < nx)
        rr, cc, tr, tc = rr[keep], cc[keep], tr[keep], tc[keep]
        if rr.size == 0:
            continue
        # The two cells sharing a face with both ends of the diagonal step.
        ar, ac = tr, cc  # step vertically first
        br, bc = rr, tc  # step horizontally first
        take_a = area[ar, ac] >= area[br, bc]
        pr = np.where(take_a, ar, br)
        pc = np.where(take_a, ac, bc)
        # Flow accumulation grows downstream, so this is the upstream end's area.
        through = np.minimum(area[rr, cc], area[tr, tc])
        # `.at` so several steps routed through one corner take the largest river.
        np.maximum.at(area, (pr, pc), through)

    inserted = int(np.count_nonzero(area >= float(min_area_km2))) - before
    return area, inserted


def _neighbour_count(mask: np.ndarray) -> np.ndarray:
    """Number of 4-connected neighbours each cell has inside ``mask``."""
    n = np.zeros(mask.shape, dtype=np.int8)
    n[:, :-1] += mask[:, 1:]
    n[:, 1:] += mask[:, :-1]
    n[:-1, :] += mask[1:, :]
    n[1:, :] += mask[:-1, :]
    return n


def isolated_cells(mask: np.ndarray, *, interior_only: bool = False) -> int:
    """Channel cells with **no** 4-connected neighbour -- each one is a sealed pool.

    This is the binary form of the connectivity gate: it needs no tolerance, so it
    cannot be calibrated by its own answer.

    ``interior_only`` drops cells on the array border, and the gate wants it on. A
    field is a *window* cut from a larger raster, so a river crossing the border has
    its continuation outside the domain -- on the real DEM's M0 tile exactly one cell
    is isolated and it sits on row 0. That is the window's edge, not a broken network,
    and no amount of corner insertion can join it to anything.
    """
    mask = np.asarray(mask, dtype=bool)
    bad = mask & (_neighbour_count(mask) == 0)
    if interior_only and bad.any():
        bad[0, :] = False
        bad[-1, :] = False
        bad[:, 0] = False
        bad[:, -1] = False
    return int(np.count_nonzero(bad))


def components(mask: np.ndarray, *, diagonal: bool = False) -> tuple[int, int]:
    """``(n_components, largest)`` for a boolean mask, 4- or 8-connected.

    Union-find over the mask's edges -- numpy only, because ``scipy`` is not a
    declared dependency of this project (it arrives transitively with the geo extra
    and must not be relied on).
    """
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return 0, 0
    idx = np.full(mask.shape, -1, dtype=np.int64)
    idx[mask] = np.arange(n)

    edges = []

    def _pair(a_mask, b_mask, a_idx, b_idx):
        both = a_mask & b_mask
        if both.any():
            edges.append((a_idx[both], b_idx[both]))

    _pair(mask[:, :-1], mask[:, 1:], idx[:, :-1], idx[:, 1:])
    _pair(mask[:-1, :], mask[1:, :], idx[:-1, :], idx[1:, :])
    if diagonal:
        _pair(mask[:-1, :-1], mask[1:, 1:], idx[:-1, :-1], idx[1:, 1:])
        _pair(mask[:-1, 1:], mask[1:, :-1], idx[:-1, 1:], idx[1:, :-1])

    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a_arr, b_arr in edges:
        for a, b in zip(a_arr.tolist(), b_arr.tolist(), strict=True):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    counts = np.unique(roots, return_counts=True)[1]
    return int(counts.size), int(counts.max())


def connectivity_report(width: np.ndarray) -> dict:
    """Diagnostics for a channel-width field: is this network able to convey?

    ``components_4`` equal to ``components_8`` with ``isolated_interior`` zero is the
    shipped state; ``components_4`` far larger is the defect this module's docstring
    describes, and it is invisible to every other gate in the repo. ``isolated``
    counts border cells too, which a window legitimately has.
    """
    mask = np.asarray(width) > 0.0
    n4, big4 = components(mask, diagonal=False)
    n8, big8 = components(mask, diagonal=True)
    return {
        "channel_cells": int(mask.sum()),
        "isolated": isolated_cells(mask),
        "isolated_interior": isolated_cells(mask, interior_only=True),
        "components_4": n4,
        "largest_4": big4,
        "components_8": n8,
        "largest_8": big8,
    }


def hydraulic_geometry(
    area_km2: np.ndarray,
    *,
    dx: float,
    width_coef: float = DEFAULT_WIDTH_COEF,
    width_exp: float = DEFAULT_WIDTH_EXP,
    depth_coef: float = DEFAULT_DEPTH_COEF,
    depth_exp: float = DEFAULT_DEPTH_EXP,
    min_area_km2: float = DEFAULT_MIN_AREA_KM2,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Channel ``(width, depth, n_clipped)`` fields from upstream drainage area.

    Widths are clipped to ``dx``: a channel wider than its cell is not sub-grid,
    and the solver refuses it (:func:`solver.core.channels.validate_geometry`). The
    clip count is returned rather than swallowed -- a large one means the grid is
    too coarse for the river it is carrying, which is a modelling decision, not a
    detail.

    ``dx`` is **the resolution the run will step at**, not necessarily the tile
    resolution: ``[grid] coarsen = k`` means cells of ``k*dx``, and
    :mod:`solver.io.coarsen` aggregates channel width by block *max*, which cannot
    recover a width already clipped away. Clipping at the tile resolution therefore
    understates the river -- by up to 10.5x on the real DEM at ``coarsen = 4`` -- and
    reports the clip count against the wrong denominator (30 % where the run's own
    figure is 5.4 %). :func:`channel_fields` takes ``coarsen`` for this reason.
    """
    area = np.asarray(area_km2, dtype=np.float64)
    river = area >= float(min_area_km2)
    width = np.where(river, width_coef * np.power(np.maximum(area, 0.0), width_exp), 0.0)
    depth = np.where(river, depth_coef * np.power(np.maximum(area, 0.0), depth_exp), 0.0)
    clipped = int(np.count_nonzero(width > dx))
    # Clip so the value still satisfies `w <= dx` *after* the float32 cast:
    # solver.core.channels.validate_geometry rejects on a strict `w > grid.dx`, and
    # float32(dx) can round up past dx -- at the real DEM's coarsen-4 cell that is
    # 112.58551788 against 112.58551546. Today `Grid.dx` is a Python float, so NEP 50
    # makes that comparison float32 and it passes by luck; a float64 dx from a
    # manifest would turn the luck into a rejected field with no obvious cause.
    cap = np.float32(dx)
    if float(cap) > float(dx):
        cap = np.nextafter(cap, np.float32(0.0))
    width = np.minimum(width, float(cap))
    # "No channel" must be zero in *both* fields (the solver's single test).
    depth = np.where(width > 0.0, depth, 0.0)
    return width.astype(np.float32), depth.astype(np.float32), clipped


def area_km2_from_accumulation(acc_cells: np.ndarray, dx: float) -> np.ndarray:
    """Upstream drainage area (km²) from a D8 flow-accumulation raster (cells)."""
    return np.asarray(acc_cells, dtype=np.float64) * (dx * dx) / 1.0e6


def _mosaic_window(tiles: list[dict]) -> tuple[int, int, int, int]:
    """Half-open ``(r0, c0, r1, c1)`` source-raster window covering a tile set."""
    r0 = min(int(t["row"]) for t in tiles)
    c0 = min(int(t["col"]) for t in tiles)
    r1 = max(int(t["row"]) + int(t["height"]) for t in tiles)
    c1 = max(int(t["col"]) + int(t["width"]) for t in tiles)
    return r0, c0, r1, c1


def _read_flowdir(cond_dir: str | Path) -> np.ndarray:
    """The conditioned D8 direction raster (local import: needs the geo extra)."""
    import rasterio

    with rasterio.open(Path(cond_dir) / "flow_direction.tif") as ds:
        return ds.read(1)


def channel_fields(
    cond_dir: str | Path,
    tiles_dir: str | Path,
    out_dir: str | Path,
    *,
    coarsen: int = 1,
    connect: bool = True,
    **coeffs: float,
) -> dict:
    """Write ``channel_width.r32`` / ``channel_depth.r32`` for a tile mosaic.

    Reads the conditioned flow accumulation (``pipeline.condition`` output), cuts
    the window the tile set covers, applies :func:`hydraulic_geometry`, and writes
    the two fields plus a ``channels.json`` note recording the coefficients used.

    ``coarsen`` must match the scenario's ``[grid] coarsen``: widths are clipped to
    the resolution the run steps at, ``coarsen * dx``, not the tile resolution.
    ``connect`` applies :func:`rook_connect` first -- leave it on unless you are
    deliberately reproducing the unconnected network, which does not convey.
    """
    from pipeline.tile import read_conditioned  # local: needs rasterio (geo extra)

    if int(coarsen) < 1:
        raise ValueError(f"coarsen must be >= 1, got {coarsen}")

    _, acc, meta, _ = read_conditioned(cond_dir)
    manifest = json.loads((Path(tiles_dir) / "tiles.json").read_text())
    dx = float(manifest.get("dx_m", meta["dx_m"]))
    run_dx = dx * int(coarsen)
    r0, c0, r1, c1 = _mosaic_window(manifest["tiles"])
    area = area_km2_from_accumulation(acc[r0:r1, c0:c1], dx)

    min_area = float(coeffs.get("min_area_km2", DEFAULT_MIN_AREA_KM2))
    inserted = 0
    if connect:
        dirmap = meta.get("dirmap") or DEFAULT_DIRMAP
        fdir = _read_flowdir(cond_dir)[r0:r1, c0:c1]
        area, inserted = rook_connect(area, fdir, dirmap=dirmap, min_area_km2=min_area)

    width, depth, clipped = hydraulic_geometry(area, dx=run_dx, **coeffs)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    width.astype("<f4").tofile(out / "channel_width.r32")
    depth.astype("<f4").tofile(out / "channel_depth.r32")
    record = {
        "source": str(Path(cond_dir)),
        "tiles": str(Path(tiles_dir)),
        "shape": [int(r1 - r0), int(c1 - c0)],
        "dx_m": dx,
        "coarsen": int(coarsen),
        "run_dx_m": run_dx,
        "coefficients": {
            "width_coef": coeffs.get("width_coef", DEFAULT_WIDTH_COEF),
            "width_exp": coeffs.get("width_exp", DEFAULT_WIDTH_EXP),
            "depth_coef": coeffs.get("depth_coef", DEFAULT_DEPTH_COEF),
            "depth_exp": coeffs.get("depth_exp", DEFAULT_DEPTH_EXP),
            "min_area_km2": min_area,
        },
        "note": "downstream hydraulic geometry -- REGIONAL CALIBRATION INPUTS, not constants",
        "channel_cells": int(np.count_nonzero(width)),
        "width_clipped_to_run_dx": clipped,
        "width_max_m": float(width.max()),
        "depth_max_m": float(depth.max()),
        "rook_connected": bool(connect),
        "cells_inserted_for_connectivity": inserted,
        "connectivity": connectivity_report(width),
    }
    (out / "channels.json").write_text(json.dumps(record, indent=2))
    return record


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sub-grid channel fields from flow accumulation.")
    p.add_argument("--src", required=True, help="conditioned dir (pipeline.condition output)")
    p.add_argument("--tiles", required=True, help="tiles dir (tiles.json) the fields align to")
    p.add_argument("--out", required=True, help="output dir for the .r32 fields")
    p.add_argument(
        "--coarsen",
        type=int,
        default=1,
        help="the scenario's [grid] coarsen: widths are clipped to coarsen*dx, the "
        "resolution the run actually steps at (default 1)",
    )
    p.add_argument(
        "--no-connect",
        action="store_true",
        help="skip the 4-connectivity fix -- the derived network then does not convey, "
        "and no gate in this repo will say so (see the module docstring)",
    )
    p.add_argument("--width-coef", type=float, default=DEFAULT_WIDTH_COEF)
    p.add_argument("--width-exp", type=float, default=DEFAULT_WIDTH_EXP)
    p.add_argument("--depth-coef", type=float, default=DEFAULT_DEPTH_COEF)
    p.add_argument("--depth-exp", type=float, default=DEFAULT_DEPTH_EXP)
    p.add_argument("--min-area-km2", type=float, default=DEFAULT_MIN_AREA_KM2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    a = _parse_args(argv)
    rec = channel_fields(
        a.src,
        a.tiles,
        a.out,
        coarsen=a.coarsen,
        connect=not a.no_connect,
        width_coef=a.width_coef,
        width_exp=a.width_exp,
        depth_coef=a.depth_coef,
        depth_exp=a.depth_exp,
        min_area_km2=a.min_area_km2,
    )
    con = rec["connectivity"]
    print(f"channel fields -> {a.out}")
    print(
        f"  grid        : {rec['shape'][0]}x{rec['shape'][1]} @ tile dx={rec['dx_m']:.2f} m,"
        f" run dx={rec['run_dx_m']:.2f} m (coarsen {rec['coarsen']})"
    )
    print(f"  channel     : {rec['channel_cells']} cells, width <= {rec['width_max_m']:.1f} m")
    if rec["rook_connected"]:
        print(
            f"  connectivity: +{rec['cells_inserted_for_connectivity']} cells inserted at"
            f" diagonal steps; {con['components_4']} components (4-connected) vs"
            f" {con['components_8']} (8-connected), {con['isolated_interior']} isolated"
            f" ({con['isolated'] - con['isolated_interior']} more on the window edge, where"
            " the river continues outside the domain)"
        )
    if con["isolated_interior"] or con["components_4"] > con["components_8"]:
        print(
            f"  WARNING: {con['isolated_interior']} interior channel cells have no 4-connected"
            f" neighbour and the network is {con['components_4']} rook-connected fragments"
            f" against {con['components_8']} under 8-connectivity. The solver has no diagonal"
            " face, so this network fills rather than conveys -- and the mass gate cannot"
            " see it."
        )
    if rec["width_clipped_to_run_dx"]:
        print(
            f"  NOTE: {rec['width_clipped_to_run_dx']} cells had a channel wider than the run's"
            f" {rec['run_dx_m']:.1f} m cell and were clipped -- the grid is coarse for this"
            " river. Raise --min-area-km2 to leave the main stem on the grid instead."
        )
    print("  coefficients are regional calibration inputs; see channels.json")


if __name__ == "__main__":
    main()
