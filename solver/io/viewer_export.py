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

**So does the sub-grid channel geometry, and it does not buy the storage curve.**
When the store carries ``channel_width``/``channel_depth`` (§7.2, M6) they ride along
as static fields through the same mechanism, because without them a viewer cannot tell
a river cell from a floodplain cell and has to lift every cell as ``bed + depth`` --
which draws the surface along a river up to the bank-full depth `d` too high (measured:
2.74 m on the M6 demo). What the viewer does with them is **not** the physical storage
curve, and that is a measured decision rather than a shortcut: below bank full the true
surface ``z - d + h·dx/w`` is *under* the floodplain bed -- up to 2.46 m under it on the
same demo, on 1030 of 2232 channel cells -- and the rendered terrain has no trench to
put it in, because the channel is sub-grid. Drawing it there hides the river inside the
ground. So :func:`render_eta` is what the viewer draws: **above** bank full the exact
curve (``z + (h - h_bf)``, which is where the old lift was wrong by up to ``h_bf``), and
**below** it the bank itself, flat -- never above the terrain, never buried under it.
The residual is then one-sided and measurable, so the export measures it on the run's
own frames and ``manifest["static"]["channel"]`` declares it (``in_bank_offset_m``), the
same way ``domain`` declares gap fill and ``morphology`` declares a stale bed: a picture
has to be able to say what it is not. See ``docs/plans/viewer-channel-surface.md``.

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

from solver.core.channels import eta_from_h
from solver.core.grid import H_DRY

# Frames larger than this in either dimension are split into a tile grid (M6). 512
# matches the §7.2 chunk hint and Godot's comfortable texture-update size; below it,
# one file per frame stays byte-identical to the M2 export.
DEFAULT_TILE_SIZE = 512

# Cap on wet samples kept for global percentiles; beyond this a frame is strided
# (deterministically) so memory stays bounded on large runs. Percentiles are
# robust to this uniform thinning.
_PCTL_SAMPLE_CAP = 4_000_000


def render_eta(
    h: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    d: np.ndarray,
    dx: float,
) -> np.ndarray:
    """The water-surface elevation the **viewer draws** -- not the physical one.

    The physical relation is :func:`solver.core.channels.eta_from_h`; this is the
    renderable projection of it onto a terrain that has no sub-grid trench, and the
    two differ *only* below bank full::

        h >  h_bf :  z + (h - h_bf)     the exact storage curve
        h <= h_bf :  z                  the bank, flat (true surface is below it)
        no channel:  z + h              the M1 relation, bit for bit

    Continuous at ``h = h_bf`` (both branches give ``z``) and monotone in ``h``, so
    scrubbing a rising flood never steps. Kept here in host numpy because it is the
    reference the GLSL in ``viewer/shaders/water_surface.gdshader`` is written from and
    what :func:`_channel_note` measures the residual with -- one formula, two consumers,
    which is the same reason M6's face geometry lives in one place.
    """
    h, z, w, d = (np.asarray(v, dtype=np.float64) for v in (h, z, w, d))
    has = (w > 0.0) & (d > 0.0)
    h_bf = np.where(has, w * d / dx, 0.0)
    return np.where(has, z + np.maximum(h - h_bf, 0.0), z + h)


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


def _channel_geometry(ds) -> tuple[np.ndarray, np.ndarray] | None:
    """The store's sub-grid channel width/depth (§7.2), or ``None`` if it has none.

    Both or neither: ``run.py`` writes the pair together, and half of it cannot select
    a branch of the storage curve.
    """
    if "channel_width" not in ds or "channel_depth" not in ds:
        return None
    return (
        np.ascontiguousarray(ds["channel_width"].values, dtype="<f4"),
        np.ascontiguousarray(ds["channel_depth"].values, dtype="<f4"),
    )


def _export_static(
    ds,
    out_dir: Path,
    layout: list[dict],
    tiled: bool,
    channels: tuple[np.ndarray, np.ndarray] | None,
) -> dict | None:
    """Export the run's static fields through the frame tile layout (viewer terrain).

    Always the bed -- the surface the run stepped on, which is what makes water and
    terrain register (module docstring). Plus ``channel_width``/``channel_depth`` when
    the run carried sub-grid channels, because a viewer that cannot tell a river cell
    from a floodplain cell has to draw every cell the same way and gets the river up to
    ``d`` too high. Every field goes through :func:`_write_field`, so the static entry
    keeps exactly one payload shape and a channel-free run's manifest is unchanged, key
    for key.

    Returns the ``manifest["static"]`` entry, or ``None`` for a store written before
    the bed was part of §7.2 (the viewer then falls back to the M0 terrain tile).
    """
    if "bed" not in ds:
        return None
    bed = np.ascontiguousarray(ds["bed"].values, dtype="<f4")
    entry = _write_field(bed, out_dir, layout, field="bed", stem="bed", tiled=tiled)
    fields = ["bed"]
    if channels is not None:
        for name, arr in zip(("channel_width", "channel_depth"), channels, strict=True):
            payload = _write_field(arr, out_dir, layout, field=name, stem=name, tiled=tiled)
            entry["files"].update(payload["files"])
            if "tiles" in payload:
                entry.setdefault("tiles", {}).update(payload["tiles"])
            fields.append(name)
    entry["fields"] = fields
    entry["bed"] = {"min": float(bed.min()), "max": float(bed.max())}
    return entry


def _channel_note(
    channels: tuple[np.ndarray, np.ndarray],
    offset: dict,
    dx: float,
) -> dict:
    """Declare the channel geometry the picture uses **and** where it approximates.

    ``in_bank_offset_m`` is how far above its true surface the drawn sheet sits on the
    worst wet in-channel cell of the run -- the one place :func:`render_eta` knowingly
    departs from the physics, because the true surface there is below a floodplain bed
    the rendered terrain has no trench in. Measured on the run's own frames rather than
    bounded by ``d``: a bound is a property of the geometry, this is a property of the
    picture. Same species as ``manifest["morphology"]``, and for the same reason.
    """
    w, d = channels
    has = (w > 0) & (d > 0)
    return {
        "cells": int(has.sum()),
        "width_max_m": float(w.max()),
        "depth_max_m": float(d.max()),
        "dx_m": float(dx),
        "in_bank_offset_m": offset["offset"],
        "in_bank_cells": offset["cells"],
        "frame": offset["frame"],
        "note": (
            "water below bank full is drawn at the bank: its true surface is under the "
            "floodplain bed and the rendered terrain carries no sub-grid trench"
        ),
    }


def _morphology_note(ds, last_frame: int) -> dict | None:
    """Declare that the exported bed is the *initial* one, when the run moved it.

    M7 ships ``bed_change (T, Y, X)`` in the canonical store but deliberately does
    **not** animate the viewer's terrain (M7 plan §1.7): re-fitting the height map per
    frame -- and with it the water plane, the bed texture and the registration check --
    is a viewer milestone of its own. The lift that used to be the *other* reason is
    fixed (:func:`render_eta`), so what remains is the animation, not a debt riding on
    it: the curve's other input is ``channel_depth``, and M7 freezes the section (Exner
    moves ``z`` and the invert ``z - d`` translates with it), so a moving bed leaves
    ``d`` valid and only the terrain stale.

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
    is a mosaic and need not be any tile on disk -- and so does the sub-grid channel
    geometry when the run had any, because the viewer needs it to know which cells are
    river (:func:`render_eta`). Returns the written ``manifest.json`` path.
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

    # Sub-grid channels change what the viewer draws, so the export both ships the
    # geometry and measures where the drawing departs from the physics -- over every
    # frame, because the worst in-bank cell is a barely-wet one and the last frame is
    # not where the flood is (`_channel_note`).
    dx = float(ds.attrs.get("dx", 1.0))
    channels = _channel_geometry(ds)
    # The geometry ships regardless of which field was exported -- it describes the run,
    # not the frames. The *residual* is a property of the water surface, so it is only
    # measurable on a depth export.
    measure_channels = channels is not None and field == "depth"
    chan_offset = {"offset": 0.0, "cells": 0, "frame": -1}
    if measure_channels:
        chan_w, chan_d = channels
        chan_has = (chan_w > 0) & (chan_d > 0)
        chan_h_bf = np.where(chan_has, chan_w * chan_d / dx, 0.0)
        chan_bed = np.asarray(ds["bed"].values, dtype=np.float64)

    for i in range(n_frames):
        arr = np.ascontiguousarray(ds[field].isel(time=i).values, dtype="<f4")
        payload = _write_field(
            arr, out_dir, layout, field=field, stem=f"f{i:04d}_{field}", tiled=tiled
        )

        if measure_channels:
            # Only below bank full does `render_eta` depart from `eta_from_h`, and it
            # departs upward -- so the residual is the drawn surface minus the true one
            # over wet in-channel cells, and it is one-sided by construction.
            in_bank = chan_has & (arr >= h_dry) & (arr <= chan_h_bf)
            if in_bank.any():
                gap = (
                    render_eta(arr, chan_bed, chan_w, chan_d, dx)
                    - eta_from_h(arr, chan_bed, chan_w, chan_d, dx)
                )[in_bank]
                worst = float(gap.max())
                if worst > chan_offset["offset"]:
                    chan_offset = {
                        "offset": worst,
                        "cells": int(in_bank.sum()),
                        "frame": i,
                    }

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
        "dx": dx,
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
    static = _export_static(ds, out_dir, layout, tiled, channels)
    if static is not None:
        if measure_channels:
            # What the viewer draws over a river, and where that drawing is knowingly
            # an approximation -- beside the geometry it is derived from, not in a log.
            static["channel"] = _channel_note(channels, chan_offset, dx)
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
