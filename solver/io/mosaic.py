"""Tile-mosaic domain assembly (M6, HANDOFF §9 M6 -- tiling at scale).

Through M5 the solver ran the **first tile** of an M0 ``tiles.json`` manifest: one
1024-square window, which is a demo, not a reach. M6 makes the run domain the
**tile set**: every tile is placed at its ``(row, col)`` origin in the source
raster and the covered bounding box becomes the grid.

The manifest contract is M0's (:mod:`pipeline.tile`) and is unchanged -- each tile
entry carries ``file``, ``row``, ``col``, ``width``, ``height`` in source-raster
cells, and the payload is raw little-endian float32, row-major (§7). Tiles are read
through ``np.memmap``, so a windowed run touches only the bytes it needs even when
the tile set on disk is far larger than the domain.

Two selections, both declared in ``[grid]``:

``tiles``
    ``"all"`` (default -- the domain is the mosaic) or ``"first"`` (the pre-M6
    single-tile behaviour, kept because it is what every demo scenario means and
    because it makes "one tile" an explicit choice rather than an accident).
``window``
    an inclusive ``[row0, col0, row1, col1]`` box in **mosaic coordinates**
    (``(0, 0)`` is the mosaic's top-left, not the source raster's), for running a
    reach out of a large tile set.

**Gaps are declared, never silent.** A tile set need not cover its own bounding box
(edge tiles are clipped, and a hand-assembled set may be L-shaped). Uncovered cells
are filled with the mosaic's minimum covered elevation -- the same policy M0 uses
for nodata inside a tile -- and the count is returned so the caller can report it.
A hole in the terrain is a hole in the hydrology; it should never be discovered by
reading the depth field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Selections accepted by [grid] tiles.
TILE_SELECTIONS = ("all", "first")


class MosaicError(ValueError):
    """A tile manifest is unusable, or the requested window is not in it."""


@dataclass(frozen=True)
class Mosaic:
    """An assembled bed array plus the provenance of how it was assembled."""

    bed: np.ndarray  # (ny, nx) float32 elevations, metres
    dx: float  # cell size (m) from the manifest
    crs: str
    origin: tuple[int, int]  # (row, col) of bed[0, 0] in source-raster cells
    tiles_used: int  # tiles that contributed at least one cell
    tiles_total: int  # tiles in the manifest
    gap_cells: int  # cells no tile covered (filled with fill_value)
    fill_value: float
    manifest: dict

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.bed.shape[0]), int(self.bed.shape[1]))

    def as_attrs(self) -> dict:
        """JSON-serializable record of the assembly (for ``.zattrs``/provenance)."""
        return {
            "origin_row": self.origin[0],
            "origin_col": self.origin[1],
            "shape": list(self.shape),
            "tiles_used": self.tiles_used,
            "tiles_total": self.tiles_total,
            "gap_cells": self.gap_cells,
            "fill_value": self.fill_value,
        }

    def summary(self) -> str:
        ny, nx = self.shape
        gaps = (
            f", {self.gap_cells} gap cells filled @ {self.fill_value:.1f} m"
            if self.gap_cells
            else ""
        )
        return (
            f"{ny}x{nx} cells from {self.tiles_used}/{self.tiles_total} tiles "
            f"@ dx={self.dx:.2f} m (origin {self.origin}){gaps}"
        )


def read_manifest(tiles_dir: str | Path) -> dict:
    """Load and sanity-check an M0 ``tiles.json``."""
    path = Path(tiles_dir) / "tiles.json"
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise MosaicError(f"no tile manifest at {path}") from e
    except json.JSONDecodeError as e:
        raise MosaicError(f"invalid tile manifest {path}: {e}") from e
    tiles = manifest.get("tiles")
    if not tiles:
        raise MosaicError(f"tile manifest {path} lists no tiles")
    for k, t in enumerate(tiles):
        missing = [key for key in ("file", "row", "col", "width", "height") if key not in t]
        if missing:
            raise MosaicError(f"tile #{k} in {path} is missing {missing}")
    return manifest


def _bbox(tiles: list[dict]) -> tuple[int, int, int, int]:
    """Half-open ``(r0, c0, r1, c1)`` bounding box of a tile set, source cells."""
    r0 = min(int(t["row"]) for t in tiles)
    c0 = min(int(t["col"]) for t in tiles)
    r1 = max(int(t["row"]) + int(t["height"]) for t in tiles)
    c1 = max(int(t["col"]) + int(t["width"]) for t in tiles)
    return r0, c0, r1, c1


def _resolve_window(
    bbox: tuple[int, int, int, int], window: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int]:
    """Half-open source-cell box for an inclusive mosaic-relative ``window``."""
    r0, c0, r1, c1 = bbox
    if window is None:
        return bbox
    wr0, wc0, wr1, wc1 = (int(v) for v in window)
    if wr1 < wr0 or wc1 < wc0:
        raise MosaicError(
            f"[grid] window {list(window)} must be [row0, col0, row1, col1] with row1 >= row0"
        )
    ar0, ac0 = r0 + wr0, c0 + wc0
    ar1, ac1 = r0 + wr1 + 1, c0 + wc1 + 1  # inclusive -> half-open
    if wr0 < 0 or wc0 < 0 or ar1 > r1 or ac1 > c1:
        raise MosaicError(
            f"[grid] window {list(window)} is outside the mosaic, which is "
            f"{r1 - r0}x{c1 - c0} cells (rows 0..{r1 - r0 - 1}, cols 0..{c1 - c0 - 1})"
        )
    return ar0, ac0, ar1, ac1


def assemble_mosaic(
    tiles_dir: str | Path,
    *,
    select: str = "all",
    window: tuple[int, int, int, int] | None = None,
) -> Mosaic:
    """Assemble the run domain from an M0 tile set.

    ``select`` is ``"all"`` (the whole tile set) or ``"first"`` (tile 0 only, the
    pre-M6 behaviour). ``window`` further clips to an inclusive
    ``[row0, col0, row1, col1]`` box in mosaic coordinates; only the tiles that
    intersect it are read.
    """
    if select not in TILE_SELECTIONS:
        raise MosaicError(f"[grid] tiles={select!r} must be one of {list(TILE_SELECTIONS)}")
    tiles_dir = Path(tiles_dir)
    manifest = read_manifest(tiles_dir)
    tiles = list(manifest["tiles"])
    selected = tiles[:1] if select == "first" else tiles

    r0, c0, r1, c1 = _resolve_window(_bbox(selected), window)
    ny, nx = r1 - r0, c1 - c0
    bed = np.zeros((ny, nx), dtype=np.float32)
    covered = np.zeros((ny, nx), dtype=bool)

    used = 0
    for t in selected:
        tr0, tc0 = int(t["row"]), int(t["col"])
        th, tw = int(t["height"]), int(t["width"])
        # Intersection of this tile with the requested window, in source cells.
        sr0, sc0 = max(tr0, r0), max(tc0, c0)
        sr1, sc1 = min(tr0 + th, r1), min(tc0 + tw, c1)
        if sr1 <= sr0 or sc1 <= sc0:
            continue
        path = tiles_dir / str(t["file"])
        try:
            data = np.memmap(path, dtype="<f4", mode="r", shape=(th, tw))
        except (FileNotFoundError, ValueError) as e:
            raise MosaicError(f"tile {path.name}: {e}") from e
        block = np.asarray(data[sr0 - tr0 : sr1 - tr0, sc0 - tc0 : sc1 - tc0], dtype=np.float32)
        bed[sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0] = block
        covered[sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0] = True
        used += 1
        del data

    gap_cells = int((~covered).sum())
    if gap_cells == ny * nx:
        raise MosaicError(f"no tile covers the requested domain in {tiles_dir}")
    fill_value = float(bed[covered].min())
    if gap_cells:
        bed[~covered] = fill_value

    return Mosaic(
        bed=np.ascontiguousarray(bed, dtype=np.float32),
        dx=float(manifest.get("dx_m", 0.0)),
        crs=str(manifest.get("crs", "")),
        origin=(r0, c0),
        tiles_used=used,
        tiles_total=len(tiles),
        gap_cells=gap_cells,
        fill_value=fill_value,
        manifest=manifest,
    )
