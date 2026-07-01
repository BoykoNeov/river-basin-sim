"""Per-frame viewer export tests (M2, HANDOFF §7.3).

The core guarantee: a frame ``.raw`` reloaded byte-for-byte equals the canonical
Zarr ``depth[i]`` -- same shape, orientation, values -- so water registers with
terrain in Godot. Plus the manifest's colormap ranges are correct.
"""

from __future__ import annotations

import json

import numpy as np
import warp as wp
import xarray as xr

from solver.io.viewer_export import export_frames
from solver.run import Scenario, run_simulation

wp.init()


def _make_run(tmp_path):
    ny, nx = 16, 20  # non-square -> catches a Y/X transpose
    yy, xx = np.mgrid[0:ny, 0:nx]
    bed = (((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) * 0.02).astype(np.float32)
    scn = Scenario(
        name="export_test",
        dx=20.0,
        end_time=600.0,
        output_every=200.0,
        dt_max=10.0,
        rain_mm_hr=100.0,
        rain_duration=300.0,
    )
    zarr_path = tmp_path / "r.zarr"
    run_simulation(scn, bed, zarr_path, device="cpu", verbose=False)
    return zarr_path


def test_frames_roundtrip_zarr_bytes(tmp_path):
    zarr_path = _make_run(tmp_path)
    out = tmp_path / "frames"
    export_frames(zarr_path, out)

    ds = xr.open_zarr(zarr_path, consolidated=False)
    n_frames = int(ds.sizes["time"])
    ny, nx = int(ds.sizes["y"]), int(ds.sizes["x"])
    manifest = json.loads((out / "manifest.json").read_text())

    assert manifest["n_frames"] == n_frames
    assert manifest["fields"] == ["depth"]
    assert manifest["grid"] == {"width": nx, "height": ny}
    assert manifest["tile_grid"] == {"cols": 1, "rows": 1}

    # Every frame's raw bytes reload to exactly the Zarr depth slice (orientation!).
    for fr in manifest["frames"]:
        raw = np.fromfile(out / fr["files"]["depth"], dtype="<f4").reshape(ny, nx)
        expected = ds["depth"].isel(time=fr["index"]).values.astype(np.float32)
        assert np.array_equal(raw, expected)
        assert fr["depth"]["min"] == float(expected.min())
        assert fr["depth"]["max"] == float(expected.max())


def test_global_stats_are_robust_and_bounded(tmp_path):
    zarr_path = _make_run(tmp_path)
    out = tmp_path / "frames"
    export_frames(zarr_path, out)
    ds = xr.open_zarr(zarr_path, consolidated=False)
    g = json.loads((out / "manifest.json").read_text())["global"]["depth"]

    true_max = float(ds["depth"].max())
    assert g["max"] == true_max  # global max is exact, never clipped by sampling
    assert 0.0 <= g["p50"] <= g["p99"] <= g["max"]
    assert g["min"] >= 0.0


def test_export_via_cli_entrypoint(tmp_path):
    from solver.io import viewer_export

    zarr_path = _make_run(tmp_path)
    out = tmp_path / "frames_cli"
    viewer_export.main([str(zarr_path), str(out)])
    assert (out / "manifest.json").exists()
