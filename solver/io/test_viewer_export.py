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
import zarr

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
    assert manifest["tile_grid"]["cols"] == 1 and manifest["tile_grid"]["rows"] == 1
    assert "tiles" not in manifest["frames"][0]  # a 1x1 grid keeps the M2 per-frame shape

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


def test_large_frames_split_into_a_tile_grid_that_reassembles(tmp_path):
    """M6: past `tile_size` a frame is a tile grid -- and it must reassemble exactly.

    Uses a small `tile_size` so the test stays cheap; the mechanism is the same one
    a 4096-square reach run hits at the 512 default.
    """
    zarr_path = _make_run(tmp_path)  # 16 x 20 frames
    out = tmp_path / "frames_tiled"
    export_frames(zarr_path, out, tile_size=8)

    ds = xr.open_zarr(zarr_path, consolidated=False)
    ny, nx = int(ds.sizes["y"]), int(ds.sizes["x"])
    manifest = json.loads((out / "manifest.json").read_text())

    grid = manifest["tile_grid"]
    assert (grid["rows"], grid["cols"], grid["size"]) == (2, 3, 8)
    assert len(grid["tiles"]) == 6
    # Edge tiles are clipped, not padded: 20 = 8 + 8 + 4.
    assert [t["width"] for t in grid["tiles"][:3]] == [8, 8, 4]
    # The geometry tiles the frame exactly once.
    assert sum(t["width"] * t["height"] for t in grid["tiles"]) == ny * nx

    for fr in manifest["frames"]:
        assert fr["files"] == {}
        names = fr["tiles"]["depth"]
        assert len(names) == len(grid["tiles"])
        rebuilt = np.zeros((ny, nx), dtype=np.float32)
        for t, name in zip(grid["tiles"], names, strict=True):
            block = np.fromfile(out / name, dtype="<f4").reshape(t["height"], t["width"])
            rebuilt[t["y"] : t["y"] + t["height"], t["x"] : t["x"] + t["width"]] = block
        expected = ds["depth"].isel(time=fr["index"]).values.astype(np.float32)
        assert np.array_equal(rebuilt, expected)


def test_static_bed_ships_with_the_frames_and_matches_the_store(tmp_path):
    """The viewer's terrain is the run's own bed, byte-for-byte from the store.

    This is what makes terrain and water register on a mosaic/coarsened domain: the
    surface the viewer renders is the one the solver stepped on, not a tile that
    happens to be on disk.
    """
    zarr_path = _make_run(tmp_path)
    out = tmp_path / "frames"
    export_frames(zarr_path, out)

    ds = xr.open_zarr(zarr_path, consolidated=False)
    bed = ds["bed"].values.astype("<f4")
    manifest = json.loads((out / "manifest.json").read_text())

    static = manifest["static"]
    assert static["fields"] == ["bed"]
    got = np.fromfile(out / static["files"]["bed"], dtype="<f4").reshape(bed.shape)
    assert np.array_equal(got, bed)
    assert static["bed"]["min"] == float(bed.min())
    assert static["bed"]["max"] == float(bed.max())
    # The bed shares the frames' grid -- that identity is the whole point.
    assert bed.shape == (manifest["grid"]["height"], manifest["grid"]["width"])


def test_static_bed_is_tiled_with_the_frame_layout(tmp_path):
    """Tiled export: the bed uses the *same* tile geometry, so one reader decodes both."""
    zarr_path = _make_run(tmp_path)
    out = tmp_path / "frames_tiled"
    export_frames(zarr_path, out, tile_size=8)

    ds = xr.open_zarr(zarr_path, consolidated=False)
    bed = ds["bed"].values.astype(np.float32)
    manifest = json.loads((out / "manifest.json").read_text())
    tiles = manifest["tile_grid"]["tiles"]

    static = manifest["static"]
    assert static["files"] == {}
    names = static["tiles"]["bed"]
    assert len(names) == len(tiles)
    rebuilt = np.zeros(bed.shape, dtype=np.float32)
    for t, name in zip(tiles, names, strict=True):
        block = np.fromfile(out / name, dtype="<f4").reshape(t["height"], t["width"])
        rebuilt[t["y"] : t["y"] + t["height"], t["x"] : t["x"] + t["width"]] = block
    assert np.array_equal(rebuilt, bed)


def test_manifest_carries_the_mosaic_assembly_record(tmp_path):
    """Gap cells are declared to the viewer, not just to the solver's log.

    `assemble_mosaic` fills cells no tile covered at the minimum covered elevation.
    Rendered, that is a flat plateau no one can tell from a rendering bug, so the
    manifest has to carry the count and the fill value through to the picture.
    """
    zarr_path = _make_run(tmp_path)
    root = zarr.open_group(zarr_path, mode="r+")
    root.attrs.update(
        {
            "coarsen": 2,
            "domain": {
                "origin_row": 0,
                "origin_col": 0,
                "shape": [16, 20],
                "gap_cells": 7,
                "tiles_used": 3,
                "tiles_total": 4,
                "fill_value": 12.5,
            },
        }
    )
    out = tmp_path / "frames"
    export_frames(zarr_path, out)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["coarsen"] == 2
    assert manifest["domain"]["gap_cells"] == 7
    assert manifest["domain"]["fill_value"] == 12.5


def _make_sediment_run(tmp_path):
    """A morphological run over the same bowl (the step-5 scenario, see solver/test_run)."""
    ny, nx = 16, 20
    yy, xx = np.mgrid[0:ny, 0:nx]
    bed = (((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) * 0.02).astype(np.float32)
    scn = Scenario(
        name="export_sediment",
        dx=20.0,
        end_time=600.0,
        output_every=300.0,
        dt_max=5.0,
        rain_mm_hr=120.0,
        rain_duration=600.0,
        sediment_law="mpm",
        sediment_d50_m=0.002,
        sediment_interval_s=150.0,
    )
    zarr_path = tmp_path / "sed.zarr"
    run_simulation(scn, bed, zarr_path, device="cpu", verbose=False)
    return zarr_path


def test_a_morphological_run_declares_that_its_terrain_is_the_starting_bed(tmp_path):
    """M7 §1.7: the viewer does not animate terrain, so the picture must say so.

    The exported bed is the run's bed at ``t = 0`` and stays that way while the run
    scours and fills underneath it. The note is **quantitative** on purpose -- "the
    terrain is a little out of date" and "it scoured a metre" are different pictures,
    and only the number tells them apart.

    ``bed_change`` itself is deliberately *not* exported as frame tiles: the shader
    still lifts water as ``bed + depth`` rather than through the sub-grid storage
    curve (§7.3 *Known gap*), so animating the bed would animate that mis-lift too.
    """
    zarr_path = _make_sediment_run(tmp_path)
    out = tmp_path / "frames"
    export_frames(zarr_path, out)

    ds = xr.open_zarr(zarr_path, consolidated=False)
    manifest = json.loads((out / "manifest.json").read_text())
    morph = manifest["morphology"]
    assert morph["static_bed"] == "initial"

    final = ds["bed_change"].isel(time=manifest["n_frames"] - 1).values
    assert morph["bed_change"]["min"] == float(final.min()) < 0.0
    assert morph["bed_change"]["max"] == float(final.max())
    assert morph["bed_change"]["time"] == float(ds["time"].values[-1])

    # The terrain that shipped is the *initial* bed, byte for byte -- which is the
    # claim the note exists to qualify.
    bed_raw = np.fromfile(out / manifest["static"]["files"]["bed"], dtype="<f4")
    assert np.array_equal(bed_raw.reshape(ds["bed"].shape), ds["bed"].values)
    assert not any("bed_change" in name for name in [p.name for p in out.iterdir()])


def test_a_still_bed_leaves_the_manifest_exactly_as_it_was(tmp_path):
    """No morphology, no note: an unarmed run's manifest is M6's, key for key."""
    manifest = json.loads((export_frames(_make_run(tmp_path), tmp_path / "f")).read_text())
    assert "morphology" not in manifest


def test_untiled_export_bytes_are_unchanged_by_the_m6_export(tmp_path):
    """The demo-scale path must be byte-for-byte what M2 wrote."""
    zarr_path = _make_run(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    export_frames(zarr_path, a)  # default tile_size (512) -> one tile
    export_frames(zarr_path, b, tile_size=4096)

    for fr in json.loads((a / "manifest.json").read_text())["frames"]:
        name = fr["files"]["depth"]
        assert (a / name).read_bytes() == (b / name).read_bytes()
