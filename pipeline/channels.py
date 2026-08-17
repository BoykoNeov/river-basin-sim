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

**And connectivity is not conveyance.** Two further things decide whether a derived
network can carry water from where it is put in to where it is meant to leave, and
neither is visible in the width field:

*A channel wider than its cell is not sub-grid, and clipping it to the cell is a
modelling decision, not a rounding.* :func:`subgrid_cutoff_km2` gives the drainage area
at which ``w = dx``; passing it as ``max_area_km2`` drops the rivers above it from the
channel mask so the main stem is resolved on the grid like any other terrain. That is
plan §4's choice and it has a price, measured on the real DEM's M0 tile at
``coarsen = 4``: it takes the clip count from 2740 cells to **0**, and it shatters the
network from **27** rook-connected pieces into **64**, because the trunk those
tributaries hung from is no longer a channel. Off by default -- the cutoff is a choice
and this module will not make it silently.

*A D8 network can dead-end inside the domain.* The conditioning leaves cells with no
flow direction at all -- pysheds writes ``-1`` for a **flat** it could not resolve,
``-2`` for a **pit**, and ``0`` for nodata (its ``flowdir(..., flats=-1, pits=-2)``
defaults; the codes are easy to get backwards and were, in this module, until the full
mosaic was measured). Across the whole conditioned raster there are 14 482 such cells,
of which **100 are interior and in valid data**: 91 pits and 9 flats. The largest
carries **1262.5 km²** and sits at ``[1052, 1125]`` at 530.5 m. Accumulation restarts
at 1 below it, so the derived river simply stops there, and **39.1 % of the domain's
valid cells drain to one of those 100** rather than out of the raster.

They are **artefacts of the conditioning chain, not closed basins in the terrain**: on
the *raw* bed all 100 have a lower neighbour, and on the filled surface 91 are
single-cell pits whose rim stands as little as 1 mm above them. Nor are they cheaply
fixable -- pysheds' own ``fill_pits`` (which raises those 91 cells by up to 1.28 m)
takes the stranded share to 23.6 %, and a further ``resolve_flats`` pass to 20.1 %, so
iterating the primitives converges slowly and not to zero.

:func:`drainage_check` finds those components and :func:`route_report` answers whether
a scenario's inflow and outflow cells are in the same piece of river.

CLI::

    uv run python -m pipeline.channels --src data/dem/conditioned \\
        --tiles data/tiles/demo --out data/fields/smoky --coarsen 4 \\
        --max-area-km2 auto --inlet 266,327 --outlet 0,172
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


def component_labels(mask: np.ndarray, *, diagonal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``(labels, sizes)`` for a boolean mask -- ``labels`` is ``-1`` off the mask.

    Union-find over the mask's edges -- numpy only, because ``scipy`` is not a
    declared dependency of this project (it arrives transitively with the geo extra
    and must not be relied on). Label ids are arbitrary but stable for a given mask;
    ``sizes[k]`` is the cell count of label ``k``.
    """
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return np.full(mask.shape, -1, dtype=np.int64), np.zeros(0, dtype=np.int64)
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
    _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    labels = np.full(mask.shape, -1, dtype=np.int64)
    labels[mask] = inverse
    return labels, counts


def components(mask: np.ndarray, *, diagonal: bool = False) -> tuple[int, int]:
    """``(n_components, largest)`` for a boolean mask, 4- or 8-connected."""
    _, counts = component_labels(mask, diagonal=diagonal)
    if counts.size == 0:
        return 0, 0
    return int(counts.size), int(counts.max())


def _block_max(field: np.ndarray, k: int) -> np.ndarray:
    """Block-max a field over ``k x k``, trailing partial blocks cropped.

    Mirrors :func:`solver.io.coarsen.block_reduce` for channel width -- duplicated
    rather than imported because ``solver.io.coarsen`` pulls in the solver package,
    and this module must stay importable with nothing but numpy.
    """
    if int(k) <= 1:
        return field
    ny, nx = field.shape
    cropped = field[: ny - ny % k, : nx - nx % k]
    ny, nx = cropped.shape
    return cropped.reshape(ny // k, k, nx // k, k).max(axis=(1, 3))


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


def isolation_cause(connectivity: dict, uncut: dict | None) -> str:
    """Why does this network fail the 4-connectivity gate: ``"clean"``, ``"cutoff"``, ``"d8"``?

    Two entirely different things break the gate and they need different words. The D8
    defect (:func:`rook_connect`'s reason for existing) is a *bug* -- a river with a wall
    across it that nothing else can see. The ``max_area_km2`` cutoff shatters the network
    too, but that is its **known, chosen price** (a tributary joined to the trunk is cut
    loose when the trunk stops being a channel), and reporting it in the D8 defect's words
    sends the reader hunting for a connectivity bug that is not there.

    :func:`rook_connect` must run *before* the cutoff -- it needs the uncut network to
    know what the through-river is -- so the two cannot simply be reordered. The
    attribution is therefore **measured**, not assumed: ``uncut`` is the connectivity of
    the same network with the cutoff not applied, and if that one satisfies the gate then
    the cutoff is what broke it. Measured on the full mosaic at ``coarsen = 4``: cutoff
    off is 163 components 4-connected against 163 8-connected with 0 isolated cells, and
    cutoff on is 458 against 456 with 9 isolated.
    """

    def broken(rep: dict) -> bool:
        return bool(rep["isolated_interior"] or rep["components_4"] > rep["components_8"])

    if not broken(connectivity):
        return "clean"
    if uncut is None:
        return "d8"
    return "d8" if broken(uncut) else "cutoff"


def subgrid_cutoff_km2(
    run_dx: float,
    *,
    width_coef: float = DEFAULT_WIDTH_COEF,
    width_exp: float = DEFAULT_WIDTH_EXP,
) -> float:
    """The drainage area whose channel is exactly one cell wide: ``(dx / a_w)^(1/b_w)``.

    Above this the hydraulic geometry asks for a channel wider than the cell carrying
    it, which is not a sub-grid feature at all -- :func:`hydraulic_geometry` clips it to
    ``dx`` and the model quietly degenerates to "the river is one cell across", precisely
    where the flood is. Pass this as ``max_area_km2`` to drop those rivers from the
    channel mask instead and let the main stem be ordinary on-grid terrain.

    It is a function of the **run** resolution and of the width coefficients, so it moves
    when either does: 198.1 km² at the real DEM's ``coarsen = 4`` cell (112.59 m) under
    the humid-temperate defaults, 12.4 km² at the native 28.15 m.
    """
    if float(run_dx) <= 0.0:
        raise ValueError(f"run_dx must be > 0, got {run_dx}")
    return float((float(run_dx) / float(width_coef)) ** (1.0 / float(width_exp)))


def trace_downstream(
    flowdir: np.ndarray,
    cell: tuple[int, int],
    *,
    dirmap: tuple[int, ...] | list[int] = DEFAULT_DIRMAP,
    stop: np.ndarray | None = None,
    max_steps: int = 1_000_000,
) -> dict:
    """Walk the D8 flow path downstream from ``cell`` and say how it ends.

    Returns ``{"start", "end", "steps", "reason", "code", "path"}``. ``reason`` is one
    of ``left_domain`` (the path leaves the array -- the ordinary, healthy ending for a
    windowed domain), ``reached_stop`` (entered a cell of the ``stop`` mask),
    ``no_direction`` (the raster has no D8 code here -- pysheds writes ``-1`` for a
    **flat** it could not resolve, ``-2`` for a **pit** and ``0`` for nodata, per its
    ``flowdir(..., flats=-1, pits=-2)`` defaults, and ``code`` carries which),
    ``loop`` or ``max_steps``.

    **What a trace does and does not prove.** The direction raster comes from the
    *filled* elevation, while ``pipeline.tile`` writes the *raw* bed into the tiles the
    solver steps on -- so this is a check on the derived network's routing, not a
    prediction about the shallow-water run. A path that leaves the domain says the
    conditioning found a way out from here; whether water actually reaches the outlet is
    a run-time gate (plan §6) and nothing here can stand in for it. What a trace *can*
    settle is the negative: a path ending ``no_direction`` well inside the window is a
    stretch of river with nowhere to go, and there are 100 such cells in valid data in
    this raster (91 pits, 9 flats) draining 39.1 % of it.
    """
    fdir = np.asarray(flowdir)
    ny, nx = fdir.shape
    r, c = int(cell[0]), int(cell[1])
    if not (0 <= r < ny and 0 <= c < nx):
        raise ValueError(f"cell {(r, c)} is outside the {ny}x{nx} grid")
    offsets = d8_offsets(dirmap)
    stop_mask = None if stop is None else np.asarray(stop, dtype=bool)
    path: list[tuple[int, int]] = [(r, c)]
    if stop_mask is not None and stop_mask[r, c]:
        return {
            "start": [r, c],
            "end": [r, c],
            "steps": 0,
            "reason": "reached_stop",
            "code": None,
            "path": path,
        }
    seen = {(r, c)}
    for step in range(1, int(max_steps) + 1):
        code = int(fdir[r, c])
        if code not in offsets:
            return {
                "start": list(path[0]),
                "end": [r, c],
                "steps": step - 1,
                "reason": "no_direction",
                "code": code,
                "path": path,
            }
        dr, dc = offsets[code]
        r, c = r + dr, c + dc
        if not (0 <= r < ny and 0 <= c < nx):
            return {
                "start": list(path[0]),
                "end": [r, c],
                "steps": step,
                "reason": "left_domain",
                "code": None,
                "path": path,
            }
        path.append((r, c))
        if stop_mask is not None and stop_mask[r, c]:
            return {
                "start": list(path[0]),
                "end": [r, c],
                "steps": step,
                "reason": "reached_stop",
                "code": None,
                "path": path,
            }
        if (r, c) in seen:
            return {
                "start": list(path[0]),
                "end": [r, c],
                "steps": step,
                "reason": "loop",
                "code": None,
                "path": path,
            }
        seen.add((r, c))
    return {
        "start": list(path[0]),
        "end": [r, c],
        "steps": int(max_steps),
        "reason": "max_steps",
        "code": None,
        "path": path,
    }


def _on_border(cell: tuple[int, int] | list[int], shape: tuple[int, int]) -> bool:
    r, c = int(cell[0]), int(cell[1])
    return r <= 0 or c <= 0 or r >= shape[0] - 1 or c >= shape[1] - 1


def route_report(
    width: np.ndarray,
    flowdir: np.ndarray,
    inlets: list[tuple[int, int]] | tuple = (),
    outlets: list[tuple[int, int]] | tuple = (),
    *,
    dirmap: tuple[int, ...] | list[int] = DEFAULT_DIRMAP,
) -> dict:
    """Is a scenario's water put in somewhere it can get out of?

    Cells are ``(row, col)`` in **field coordinates** -- the assembled, pre-coarsen
    domain, which is the same coordinate system a scenario's ``[[inflow]] cell`` uses
    (``solver.io.coarsen`` maps them by ``i // k``).

    It answers three separate claims, and they are separate on purpose:

    1. **Is the inlet in the channel at all?** Nothing else in this repo checks this. The
       shipped ``reach_basin`` scenario injects at ``[4, 768]``, which is floodplain two
       cells off the meander, and it went unnoticed for two milestones because rain wet
       the channel anyway.
    2. **Are inlet and outlet in the same rook-connected piece of channel?** Meaningful
       only when the main stem is *carried* as a channel: with ``max_area_km2`` set the
       trunk is deliberately floodplain, the network is a set of tributary stubs hanging
       off it, and a shared component is neither expected nor required. ``None`` when a
       cell is not in the channel to begin with.
    3. **Does the flow path from the inlet dead-end inside the domain?** See
       :func:`trace_downstream` for what that does and does not prove.
    """
    w = np.asarray(width)
    mask = w > 0.0
    labels, sizes = component_labels(mask, diagonal=False)
    shape = (int(mask.shape[0]), int(mask.shape[1]))

    def describe(cell) -> dict:
        r, c = int(cell[0]), int(cell[1])
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            raise ValueError(f"cell {(r, c)} is outside the {shape[0]}x{shape[1]} field")
        lab = int(labels[r, c])
        return {
            "cell": [r, c],
            "in_channel": bool(mask[r, c]),
            "width_m": float(w[r, c]),
            "component": lab if lab >= 0 else None,
            "component_cells": int(sizes[lab]) if lab >= 0 else 0,
        }

    ins = [describe(c) for c in inlets]
    outs = [describe(c) for c in outlets]
    stop = None
    if outs:
        stop = np.zeros(shape, dtype=bool)
        for o in outs:
            stop[o["cell"][0], o["cell"][1]] = True

    warnings: list[str] = []
    for entry in ins:
        tr = trace_downstream(flowdir, entry["cell"], dirmap=dirmap, stop=stop)
        cells = np.array(tr["path"], dtype=np.int64)
        entry["route"] = {
            "reason": tr["reason"],
            "code": tr["code"],
            "steps": tr["steps"],
            "end": tr["end"],
            "channel_steps": int(mask[cells[:, 0], cells[:, 1]].sum()),
            "dead_end_inside": bool(
                tr["reason"] in ("no_direction", "loop") and not _on_border(tr["end"], shape)
            ),
        }
        if not entry["in_channel"]:
            warnings.append(
                f"inlet {entry['cell']} is not a channel cell -- the discharge lands on"
                " floodplain, which is a modelling choice, not an error, but it is"
                " usually not the one intended"
            )
        if entry["route"]["dead_end_inside"]:
            warnings.append(
                f"the flow path from inlet {entry['cell']} ends at {tr['end']} after"
                f" {tr['steps']} cells with no D8 direction (code {tr['code']}) and that"
                " cell is not on the domain edge -- this stretch of river has nowhere to"
                " go in the derived network"
            )

    # "Same piece" is a question about a *pair*: with no outlet given there is nothing
    # to be in the same piece as, and a run of inlets all in one piece must not read as
    # a connected route.
    same = None
    labs = [e["component"] for e in ins + outs]
    if ins and outs and all(x is not None for x in labs):
        same = len(set(labs)) == 1
        if not same:
            warnings.append(
                f"inlet and outlet are in different pieces of channel {sorted(set(labs))}"
                " -- with the main stem carried as a channel that is a sealed stretch;"
                " with max_area_km2 set it is expected, because the trunk between them is"
                " floodplain by choice"
            )
    return {
        "inlets": ins,
        "outlets": outs,
        "same_component": same,
        "warnings": warnings,
    }


def drainage_check(
    width: np.ndarray,
    area_km2: np.ndarray,
    flowdir: np.ndarray,
    *,
    dirmap: tuple[int, ...] | list[int] = DEFAULT_DIRMAP,
    report_max: int = 10,
) -> dict:
    """Which pieces of the derived network drain out of the domain, and which do not.

    For each rook-connected component, the flow path is traced from its
    highest-accumulation cell -- the piece's own outlet. A component is **sealed** when
    that path dead-ends inside the window (``no_direction`` or ``loop`` away from the
    border) rather than leaving it.

    A bounding-box touch is not this test and must not be mistaken for it: on the real
    DEM's M0 tile all 27 pieces reach the window edge somewhere, while the second-largest
    of them (924 cells, 1262 km²) drains to a single-cell pit in the middle of the
    domain. Sprawling to the edge through a tributary says nothing about where the water
    goes.

    **Scale matters to what this finds.** The same census on the whole 3991x3283 mosaic
    reads 154 of 458 pieces sealed (114 939 of 298 147 channel cells, 38.6 %) with the
    §4 cutoff on, and 23 of 163 (123 469 of 316 070) with it off -- against 2 of 29
    (14.5 %) on the single M0 tile. A one-tile census understates this badly, because a
    window that happens to miss the pits looks clean.
    """
    mask = np.asarray(width) > 0.0
    area = np.asarray(area_km2, dtype=np.float64)
    labels, sizes = component_labels(mask, diagonal=False)
    shape = (int(mask.shape[0]), int(mask.shape[1]))
    sealed: list[dict] = []
    largest: dict | None = None
    for lab in range(int(sizes.size)):
        sel = labels == lab
        pos = np.unravel_index(int(np.argmax(np.where(sel, area, -np.inf))), shape)
        tr = trace_downstream(flowdir, pos, dirmap=dirmap)
        entry = {
            "component": lab,
            "cells": int(sizes[lab]),
            "outlet": [int(pos[0]), int(pos[1])],
            "outlet_area_km2": float(area[pos]),
            "outlet_on_edge": _on_border(pos, shape),
            "reason": tr["reason"],
            "code": tr["code"],
            "steps": tr["steps"],
        }
        if tr["reason"] in ("no_direction", "loop") and not _on_border(tr["end"], shape):
            sealed.append(entry)
        if largest is None or entry["cells"] > largest["cells"]:
            largest = entry
    sealed.sort(key=lambda e: -e["cells"])
    return {
        "components": int(sizes.size),
        "sealed_components": len(sealed),
        "sealed_cells": int(sum(e["cells"] for e in sealed)),
        "largest": largest,
        "sealed": sealed[: int(report_max)],
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
    max_area_km2: float | None = None,
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

    ``max_area_km2`` is the other end of the same argument: a cell whose river is
    *wider* than ``dx`` is not sub-grid, so carrying it clipped is a degenerate model
    rather than an approximate one. Setting the cutoff to :func:`subgrid_cutoff_km2`
    drops those cells from the mask -- the main stem then travels as ordinary on-grid
    flow and only the tributaries are sub-grid. ``None`` (the default) keeps them and
    clips, which is what every figure recorded before 2026-08-17 was measured under.
    """
    area = np.asarray(area_km2, dtype=np.float64)
    river = area >= float(min_area_km2)
    if max_area_km2 is not None:
        river &= area <= float(max_area_km2)
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
    max_area_km2: float | str | None = None,
    inlets: list[tuple[int, int]] | tuple = (),
    outlets: list[tuple[int, int]] | tuple = (),
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

    ``max_area_km2`` drops rivers too big to be sub-grid at the run resolution;
    ``"auto"`` resolves it to :func:`subgrid_cutoff_km2` from the coefficients actually
    in use, and ``None`` (the default) carries them clipped to the cell. ``inlets`` /
    ``outlets`` are ``(row, col)`` cells in this field's coordinates -- the same ones a
    scenario's ``[[inflow]]`` uses -- and their :func:`route_report` is written into
    ``channels.json`` beside the geometry, because where the water goes in is part of
    what these fields mean.
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
    dirmap = tuple(meta.get("dirmap") or DEFAULT_DIRMAP)
    fdir = _read_flowdir(cond_dir)[r0:r1, c0:c1]
    inserted = 0
    if connect:
        area, inserted = rook_connect(area, fdir, dirmap=dirmap, min_area_km2=min_area)

    if isinstance(max_area_km2, str):
        if max_area_km2 != "auto":
            raise ValueError(f"max_area_km2 must be a number, 'auto' or None, got {max_area_km2!r}")
        max_area = subgrid_cutoff_km2(
            run_dx,
            width_coef=coeffs.get("width_coef", DEFAULT_WIDTH_COEF),
            width_exp=coeffs.get("width_exp", DEFAULT_WIDTH_EXP),
        )
    else:
        max_area = None if max_area_km2 is None else float(max_area_km2)
    above = 0
    if max_area is not None:
        above = int(np.count_nonzero((area >= min_area) & (area > max_area)))

    width, depth, clipped = hydraulic_geometry(area, dx=run_dx, max_area_km2=max_area, **coeffs)

    # With a cutoff set, the network shatters and some cells are left isolated -- the
    # cutoff's known price, not the D8 defect. Measure the same network *without* the
    # cutoff so the attribution is a comparison rather than an assumption
    # (:func:`isolation_cause`); the mask is otherwise identical, so this isolates the
    # one variable.
    uncut_connectivity = None
    if max_area is not None:
        uncut_width, _, _ = hydraulic_geometry(area, dx=run_dx, max_area_km2=None, **coeffs)
        uncut_connectivity = connectivity_report(uncut_width)

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
            "max_area_km2": max_area,
        },
        "note": "downstream hydraulic geometry -- REGIONAL CALIBRATION INPUTS, not constants",
        "channel_cells": int(np.count_nonzero(width)),
        "width_clipped_to_run_dx": clipped,
        "width_max_m": float(width.max()),
        "depth_max_m": float(depth.max()),
        "subgrid_cutoff_km2": subgrid_cutoff_km2(
            run_dx,
            width_coef=coeffs.get("width_coef", DEFAULT_WIDTH_COEF),
            width_exp=coeffs.get("width_exp", DEFAULT_WIDTH_EXP),
        ),
        "cells_above_max_area": above,
        "rook_connected": bool(connect),
        "cells_inserted_for_connectivity": inserted,
        "connectivity": connectivity_report(width),
        # The fields are authored at the tile resolution and the solver block-maxes them
        # to its own; quoting only the authored figure invites a "did not reproduce".
        "connectivity_at_run_dx": connectivity_report(_block_max(width, int(coarsen))),
        # Only present when a cutoff is set: the same network without it, so a broken
        # connectivity gate can be attributed to the cutoff rather than to the D8 defect.
        "connectivity_without_cutoff": uncut_connectivity,
        "isolation_cause": isolation_cause(connectivity_report(width), uncut_connectivity),
        "drainage": drainage_check(width, area, fdir, dirmap=dirmap),
    }
    if inlets or outlets:
        record["route"] = route_report(width, fdir, inlets, outlets, dirmap=dirmap)
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
    p.add_argument(
        "--max-area-km2",
        default=None,
        help="drop rivers above this drainage area from the channel mask -- they are "
        "wider than a cell, so carrying them is a degenerate model, and the main stem "
        "travels on the grid instead. 'auto' uses the area whose channel is exactly one "
        "cell wide at the run resolution (default: keep them, clipped to the cell)",
    )
    p.add_argument(
        "--inlet",
        action="append",
        default=[],
        metavar="ROW,COL",
        help="a cell water is put into (a scenario's [[inflow]] cell, in this field's "
        "pre-coarsen coordinates); repeatable. Reported in channels.json",
    )
    p.add_argument(
        "--outlet",
        action="append",
        default=[],
        metavar="ROW,COL",
        help="a cell water is meant to leave by; repeatable",
    )
    p.add_argument("--width-coef", type=float, default=DEFAULT_WIDTH_COEF)
    p.add_argument("--width-exp", type=float, default=DEFAULT_WIDTH_EXP)
    p.add_argument("--depth-coef", type=float, default=DEFAULT_DEPTH_COEF)
    p.add_argument("--depth-exp", type=float, default=DEFAULT_DEPTH_EXP)
    p.add_argument("--min-area-km2", type=float, default=DEFAULT_MIN_AREA_KM2)
    return p.parse_args(argv)


def _parse_cell(text: str) -> tuple[int, int]:
    """``"row,col"`` -> ``(row, col)``."""
    parts = text.replace("(", "").replace(")", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"expected ROW,COL, got {text!r}")
    return int(parts[0]), int(parts[1])


def main(argv: list[str] | None = None) -> None:
    a = _parse_args(argv)
    max_area: float | str | None = a.max_area_km2
    if isinstance(max_area, str) and max_area != "auto":
        max_area = float(max_area)
    rec = channel_fields(
        a.src,
        a.tiles,
        a.out,
        coarsen=a.coarsen,
        connect=not a.no_connect,
        max_area_km2=max_area,
        inlets=[_parse_cell(t) for t in a.inlet],
        outlets=[_parse_cell(t) for t in a.outlet],
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
        run = rec["connectivity_at_run_dx"]
        print(
            f"  connectivity: +{rec['cells_inserted_for_connectivity']} cells inserted at"
            f" diagonal steps; {con['components_4']} components (4-connected) vs"
            f" {con['components_8']} (8-connected), {con['isolated_interior']} isolated"
            f" ({con['isolated'] - con['isolated_interior']} more on the window edge, where"
            " the river continues outside the domain)"
        )
        if rec["coarsen"] > 1:
            print(
                f"                as the solver runs it ({run['channel_cells']} cells after"
                f" block-max at coarsen {rec['coarsen']}): {run['components_4']} vs"
                f" {run['components_8']}, {run['isolated_interior']} isolated"
            )
    cause = rec["isolation_cause"]
    if cause == "d8":
        print(
            f"  WARNING: {con['isolated_interior']} interior channel cells have no 4-connected"
            f" neighbour and the network is {con['components_4']} rook-connected fragments"
            f" against {con['components_8']} under 8-connectivity. The solver has no diagonal"
            " face, so this network fills rather than conveys -- and the mass gate cannot"
            " see it."
        )
    elif cause == "cutoff":
        # Not the D8 defect: the same network without the cutoff satisfies the gate, so
        # saying "this network fills rather than conveys" would send the reader hunting
        # for a bug that is not there.
        unc = rec["connectivity_without_cutoff"]
        print(
            f"  cutoff cost : {con['components_4']} pieces (4-connected) vs"
            f" {con['components_8']} (8-connected) and {con['isolated_interior']} isolated"
            f" cells -- against {unc['components_4']}/{unc['components_8']} and"
            f" {unc['isolated_interior']} for the same network WITHOUT the cutoff. This is"
            " the cutoff's price, not the D8 defect: tributaries joined through the trunk"
            " are cut loose when the trunk stops being a channel. Read the flow paths, not"
            " the component count."
        )
    cut = rec["coefficients"]["max_area_km2"]
    if cut is not None:
        print(
            f"  cutoff      : A <= {cut:.1f} km2 ({rec['cells_above_max_area']} cells above it"
            " are floodplain, not channel -- the main stem is resolved on the grid)"
        )
    if rec["width_clipped_to_run_dx"]:
        print(
            f"  NOTE: {rec['width_clipped_to_run_dx']} cells had a channel wider than the run's"
            f" {rec['run_dx_m']:.1f} m cell and were clipped to it -- there the model is not"
            " sub-grid at all, it is 'the river is one cell across', and that is where the"
            f" flood is. Pass --max-area-km2 auto (= {rec['subgrid_cutoff_km2']:.1f} km2 here)"
            " to leave the main stem on the grid instead."
        )
    dr = rec["drainage"]
    if dr["sealed_components"]:
        worst = dr["sealed"][0]
        print(
            f"  WARNING: {dr['sealed_components']} of {dr['components']} pieces of channel"
            f" ({dr['sealed_cells']} cells) drain to a dead end inside the domain. The"
            f" largest is {worst['cells']} cells ending at {worst['outlet']} with"
            f" {worst['outlet_area_km2']:.1f} km2 upstream and no D8 direction (code"
            f" {worst['code']}) -- water put in there ponds, and the mass gate cannot see it."
        )
    route = rec.get("route")
    if route:
        for e in route["inlets"]:
            r = e["route"]
            print(
                f"  inlet {e['cell']}: {'channel' if e['in_channel'] else 'FLOODPLAIN'}"
                f" w={e['width_m']:.1f} m, piece {e['component']} ({e['component_cells']} cells);"
                f" flow path {r['reason']} after {r['steps']} cells, {r['channel_steps']} of them"
                " channel"
            )
        for e in route["outlets"]:
            print(
                f"  outlet {e['cell']}: {'channel' if e['in_channel'] else 'floodplain'},"
                f" piece {e['component']} ({e['component_cells']} cells)"
            )
        same = route["same_component"]
        off = [e["cell"] for e in route["inlets"] + route["outlets"] if not e["in_channel"]]
        if same is True:
            print("  route       : inlet and outlet are in the same piece of channel")
        elif same is False:
            print("  route       : inlet and outlet are in DIFFERENT pieces of channel")
        elif not route["outlets"]:
            pass  # no outlet given: there is no route to have a verdict about
        elif cut is not None:
            print(
                f"  route       : {off} not in the channel, so there is no channel-to-channel"
                " route -- with a cutoff set the main stem is floodplain on purpose, so read"
                " the flow path above instead"
            )
        else:
            print(f"  route       : {off} not in the channel, so 'same piece' has no answer")
    for msg in route["warnings"] if route else []:
        print(f"  WARNING: {msg}")
    print("  coefficients are regional calibration inputs; see channels.json")


if __name__ == "__main__":
    main()
