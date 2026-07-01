"""Per-frame viewer export (M2, HANDOFF §7.3 -- the lean stream Godot reads).

Godot cannot read Zarr; the canonical store (§7.2) is for analysis. So after a run
we make a **parallel lean stream**: one raw little-endian float32 tile per frame
per exported field, plus a ``manifest.json`` carrying times and colormap ranges so
the viewer never has to scan the data to colour it.

This is deliberately a **post-process over the finished Zarr**, not inline in the
solver hot loop: it can be re-run to regenerate ``frames/`` without re-simulating,
and it keeps the solver's only *live* output the canonical store (decoupling, §4).

Layout (single demo tile -> a 1x1 tile grid; multi-tile splitting is M6)::

    frames/
      manifest.json
      f0000_depth.raw   raw LE f32, row-major (Y, X), metres -- one tile
      f0001_depth.raw
      ...

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


def export_frames(
    zarr_path: str | Path,
    out_dir: str | Path,
    *,
    field: str = "depth",
    h_dry: float = H_DRY,
) -> Path:
    """Export ``field`` from a canonical Zarr store as §7.3 per-frame tiles.

    Returns the written ``manifest.json`` path.
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

    frames: list[dict] = []
    wet_samples: list[np.ndarray] = []
    global_max = 0.0

    for i in range(n_frames):
        arr = np.ascontiguousarray(ds[field].isel(time=i).values, dtype="<f4")
        fname = f"f{i:04d}_{field}.raw"
        (out_dir / fname).write_bytes(arr.tobytes())

        fmin, fmax = float(arr.min()), float(arr.max())
        global_max = max(global_max, fmax)
        wet = arr[arr >= h_dry].astype(np.float64, copy=False)
        if wet.size > _PCTL_SAMPLE_CAP:
            stride = int(np.ceil(wet.size / _PCTL_SAMPLE_CAP))
            wet = wet[::stride]
        wet_samples.append(wet)

        frames.append(
            {
                "index": i,
                "time": times[i],
                "files": {field: fname},
                field: {"min": fmin, "max": fmax},
            }
        )

    all_wet = np.concatenate(wet_samples) if wet_samples else np.empty(0)
    global_stats = _robust_stats(all_wet)
    global_stats["max"] = global_max  # true max (percentile sampling never clips it)

    manifest = {
        "dx": float(ds.attrs.get("dx", 1.0)),
        "crs": str(ds.attrs.get("crs", "")),
        "scheme": str(ds.attrs.get("scheme", "")),
        "grid": {"width": nx, "height": ny},
        "tile_grid": {"cols": 1, "rows": 1},
        "fields": [field],
        "h_dry": float(h_dry),
        "n_frames": n_frames,
        "global": {field: global_stats},
        "frames": frames,
    }
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
