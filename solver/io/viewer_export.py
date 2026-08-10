"""Per-frame viewer export (M2, HANDOFF §7.3 -- the lean stream Godot reads).

Godot cannot read Zarr; the canonical store (§7.2) is for analysis. So after a run
we make a **parallel lean stream**: one raw little-endian float32 tile per frame
per exported field, plus a ``manifest.json`` carrying times and colormap ranges so
the viewer never has to scan the data to colour it.

This is deliberately a **post-process over the finished Zarr**, not inline in the
solver hot loop: it can be re-run to regenerate ``frames/`` without re-simulating,
and it keeps the solver's only *live* output the canonical store (decoupling, §4).

Layout -- one file per frame while the frame fits in a tile (the demo case), a
row-major tile grid beyond that (M6, reach scale)::

    frames/
      manifest.json
      f0000_depth.raw               small domain: the whole frame, one tile
      f0001_depth.raw
      ...
      f0000_depth_r00_c00.raw       reach scale: `tile_grid` tiles, row-major
      f0000_depth_r00_c01.raw
      ...

``manifest["tile_grid"]`` carries ``rows``/``cols``/``size`` and the geometry of
every tile (``x``, ``y``, ``width``, ``height``) **once**; each frame lists its tile
files in that same order (``frames[i]["tiles"]["depth"]``). A 1x1 grid keeps the M2
per-frame shape exactly -- one ``.raw`` of identical bytes, ``frames[i]["files"]``,
no ``tiles`` key -- so existing readers keep working; only the manifest's
``tile_grid`` gains the (now non-trivial) geometry fields.

**The bed ships with the frames.** ``manifest["static"]`` carries the run's own bed
field, exported once through the same tile layout (``bed.raw``, or
``bed_r00_c00.raw`` ... when tiled) and with the same entry shape as a frame, so one
reader decodes both. It is not a convenience copy of the M0 tile: through M5 the run
domain *was* the first terrain tile, but a reach-scale run is a **mosaic**, possibly
windowed and coarsened, so the only surface that registers with the depth field cell
for cell is the one the solver actually stepped on. The canonical store already holds
it (§7.2 ``bed``, in true elevations -- the datum shift is undone on the way out), so
exporting it here means the viewer's terrain and its water share extent, origin and
cell size **by construction** rather than by two implementations agreeing.
``manifest["domain"]`` carries the mosaic assembly record beside it, because
``assemble_mosaic`` fills uncovered cells at the minimum covered elevation and
rendering that is a flat plateau no one can tell from a bug.

**A morphological run's bed is the one it started on.** ``bed`` in the canonical
store is always the initial bed and M7's ``bed_change`` is not exported as frames
(plan §1.7 -- terrain animation is a viewer milestone, and the shader's
``bed + depth`` lift debt would move with it). So when the store carries
``bed_change``, ``manifest["morphology"]`` declares that the terrain is the
``t = 0`` bed and by how much the run moved it -- same species as the ``domain``
note, and for the same reason: a picture has to be able to say what it is not.

Byte layout matches the M0 ``.r32`` convention (raw LE f32, row-major) so the
viewer loads a frame into a Godot ``FORMAT_RF`` image with the *same* orientation
and origin the M0 terrain loader established -- no transpose, water registers with
terrain.

**Colormap ranges.** ``manifest["global"]["depth"]`` carries robust stats over all
*wet* cells across all frames (``min``/``max``/``p50``/``p99``); each frame also
carries its own ``min``/``max``. The viewer's default colormap clamps to
``[0, global.p99]`` so a thin floodplain sheet stays visible and a rare deep
channel does not wash it out -- and using a **global** (not per-frame) range keeps
colours stable while scrubbing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from solver.core.grid import H_DRY

# Frames larger than this in either dimension are split into a tile grid (M6). 512
# matches the §7.2 chunk hint and Godot's comfortable texture-update size; below it,
# one file per frame stays byte-identical to the M2 export.
DEFAULT_TILE_SIZE = 512

# Cap on wet samples kept for global percentiles; beyond this a frame is strided
# (deterministically) so memory stays bounded on large runs. Percentiles are
# robust to this uniform thinning.
_PCTL_SAMPLE_CAP = 4_000_000


def _robust_stats(wet: np.ndarray) -> dict:
    """min / max / p50 / p99 of a 1-D array of wet depths (zeros if empty)."""
    if wet.size == 0:
        return {"min": 0.0, "max": 0.0, "p50": 0.0, "p99": 0.0}
    return {
        "min": float(wet.min()),
        "max": float(wet.max()),
        "p50": float(np.percentile(wet, 50.0)),
        "p99": float(np.percentile(wet, 99.0)),
    }


def _tile_layout(ny: int, nx: int, tile_size: int) -> list[dict]:
    """Row-major tile geometry covering a ``(ny, nx)`` frame (edge tiles clipped)."""
    tiles = []
    for r, y0 in enumerate(range(0, ny, tile_size)):
        for c, x0 in enumerate(range(0, nx, tile_size)):
            tiles.append(
                {
                    "row": r,
                    "col": c,
                    "x": x0,
                    "y": y0,
                    "width": min(tile_size, nx - x0),
                    "height": min(tile_size, ny - y0),
                }
            )
    return tiles


def _write_field(
    arr: np.ndarray,
    out_dir: Path,
    layout: list[dict],
    *,
    field: str,
    stem: str,
    tiled: bool,
) -> dict:
    """Write one ``(ny, nx)`` field as a whole ``.raw`` or as ``layout``'s tiles.

    ``stem`` names the payload on disk (``f0007_depth``, ``bed``); ``field`` is the
    manifest key it is filed under. Returns the fragment describing where the bytes
    went -- ``files`` untiled, ``files``/``tiles`` tiled. Frames and the static bed
    share this so a reader has exactly one payload shape to decode.
    """
    if not tiled:
        name = f"{stem}.raw"
        (out_dir / name).write_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())
        return {"files": {field: name}}
    names: list[str] = []
    for t in layout:
        block = np.ascontiguousarray(
            arr[t["y"] : t["y"] + t["height"], t["x"] : t["x"] + t["width"]], dtype="<f4"
        )
        tname = f"{stem}_r{t['row']:02d}_c{t['col']:02d}.raw"
        (out_dir / tname).write_bytes(block.tobytes())
        names.append(tname)
    return {"files": {}, "tiles": {field: names}}


def _export_bed(ds, out_dir: Path, layout: list[dict], tiled: bool) -> dict | None:
    """Export the store's static bed through the frame tile layout (viewer terrain).

    Returns the ``manifest["static"]`` entry, or ``None`` for a store written before
    the bed was part of §7.2 (the viewer then falls back to the M0 terrain tile).
    """
    if "bed" not in ds:
        return None
    bed = np.ascontiguousarray(ds["bed"].values, dtype="<f4")
    entry = _write_field(bed, out_dir, layout, field="bed", stem="bed", tiled=tiled)
    entry["fields"] = ["bed"]
    entry["bed"] = {"min": float(bed.min()), "max": float(bed.max())}
    return entry


def _morphology_note(ds, last_frame: int) -> dict | None:
    """Declare that the exported bed is the *initial* one, when the run moved it.

    M7 ships ``bed_change (T, Y, X)`` in the canonical store but deliberately does
    **not** animate the viewer's terrain (M7 plan §1.7): re-fitting the height map
    per frame is a viewer milestone, and the carried debt makes it worse rather than
    better -- the shader still lifts water as ``bed + depth`` instead of through the
    sub-grid storage curve (§7.3 *Known gap*), so a moving bed would move that
    mis-lift with it. Fix the lift first, then animate.

    What is *not* acceptable is a picture that quietly implies the terrain is
    current, which is the same failure the mosaic terrain fix was about. So a store
    that morphed says so in the manifest, **quantitatively** -- the extremes of the
    final bed change are how wrong the rendered terrain is -- and the viewer prints
    it. Returns ``None`` for a run without morphology, so an unarmed run's manifest
    is byte-identical to M6's.
    """
    if "bed_change" not in ds or last_frame < 0:
        return None
    dz = np.asarray(ds["bed_change"].isel(time=last_frame).values, dtype=np.float64)
    return {
        "bed_change": {
            "min": float(dz.min()),
            "max": float(dz.max()),
            "frame": last_frame,
            "time": float(ds["time"].values[last_frame]),
        },
        "static_bed": "initial",
        "note": (
            "terrain is the bed at t=0; this run moved it (see bed_change in the "
            "canonical store). The viewer does not animate terrain in M7."
        ),
    }


def export_frames(
    zarr_path: str | Path,
    out_dir: str | Path,
    *,
    field: str = "depth",
    h_dry: float = H_DRY,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> Path:
    """Export ``field`` from a canonical Zarr store as §7.3 per-frame tiles.

    A frame that fits inside ``tile_size`` is written as **one** ``.raw`` per frame
    -- byte-identical to the M2 export, which is what every demo-scale run gets.
    Beyond that the frame is split into a row-major tile grid (M6): a 4096² frame is
    64 MB, too big to hand a viewer as one buffer per scrub step, and §7.3 always
    specified a ``tile_grid``. The geometry is listed **once** in the manifest and
    each frame lists its tile files in the same row-major order.

    The store's static ``bed`` rides along through the same layout as
    ``manifest["static"]`` -- the surface the run actually stepped on, which after M6
    is a mosaic and need not be any tile on disk. Returns the written
    ``manifest.json`` path.
    """
    zarr_path = Path(zarr_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(zarr_path, consolidated=False)
    # Honour the ledger's recorded frame count: on an early stop the time axis is
    # preallocated larger than what was written (zarr_writer.finalize records
    # n_frames but does not resize), so trailing zero frames must not be exported.
    n_frames = min(int(ds.attrs.get("n_frames", ds.sizes["time"])), int(ds.sizes["time"]))
    ny, nx = int(ds.sizes["y"]), int(ds.sizes["x"])
    times = [float(t) for t in ds["time"].values]

    layout = _tile_layout(ny, nx, max(int(tile_size), 1))
    tiled = len(layout) > 1
    rows = max(t["row"] for t in layout) + 1
    cols = max(t["col"] for t in layout) + 1

    frames: list[dict] = []
    wet_samples: list[np.ndarray] = []
    global_max = 0.0

    for i in range(n_frames):
        arr = np.ascontiguousarray(ds[field].isel(time=i).values, dtype="<f4")
        payload = _write_field(
            arr, out_dir, layout, field=field, stem=f"f{i:04d}_{field}", tiled=tiled
        )

        fmin, fmax = float(arr.min()), float(arr.max())
        global_max = max(global_max, fmax)
        wet = arr[arr >= h_dry].astype(np.float64, copy=False)
        if wet.size > _PCTL_SAMPLE_CAP:
            stride = int(np.ceil(wet.size / _PCTL_SAMPLE_CAP))
            wet = wet[::stride]
        wet_samples.append(wet)

        # When tiled, the tile list *is* the frame and `files` stays empty: keeping
        # both keys would invite a reader to load a file that does not exist.
        entry = {"index": i, "time": times[i], **payload, field: {"min": fmin, "max": fmax}}
        frames.append(entry)

    all_wet = np.concatenate(wet_samples) if wet_samples else np.empty(0)
    global_stats = _robust_stats(all_wet)
    global_stats["max"] = global_max  # true max (percentile sampling never clips it)

    manifest = {
        "dx": float(ds.attrs.get("dx", 1.0)),
        "crs": str(ds.attrs.get("crs", "")),
        "scheme": str(ds.attrs.get("scheme", "")),
        "coarsen": int(ds.attrs.get("coarsen", 1)),
        "grid": {"width": nx, "height": ny},
        # Row-major tile geometry, listed once (frames reference it by order).
        "tile_grid": {"cols": cols, "rows": rows, "size": int(tile_size), "tiles": layout},
        "fields": [field],
        "h_dry": float(h_dry),
        "n_frames": n_frames,
        "global": {field: global_stats},
        "frames": frames,
    }
    static = _export_bed(ds, out_dir, layout, tiled)
    if static is not None:
        manifest["static"] = static
    morphology = _morphology_note(ds, n_frames - 1)
    if morphology is not None:
        # The bed the viewer renders is the run's *first* bed when the run morphed;
        # the note is what keeps that from reading as the current one.
        manifest["morphology"] = morphology
    domain = ds.attrs.get("domain")
    if isinstance(domain, dict):
        # How the mosaic was assembled (M6): uncovered cells are filled at the
        # minimum covered elevation, which renders as a flat plateau -- the viewer
        # has to be able to say so rather than let it read as a rendering bug.
        manifest["domain"] = dict(domain)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export §7.3 per-frame viewer tiles from a Zarr store.")
    p.add_argument("zarr", help="canonical results.zarr store")
    p.add_argument("out_dir", help="output frames/ directory")
    p.add_argument("--field", default="depth", help="field to export (default: depth)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = export_frames(args.zarr, args.out_dir, field=args.field)
    print(f"viewer export: {manifest}")


if __name__ == "__main__":
    main()
