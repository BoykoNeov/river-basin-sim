"""Per-frame viewer export tests (M2, HANDOFF §7.3).

The core guarantee: a frame ``.raw`` reloaded byte-for-byte equals the canonical
Zarr ``depth[i]`` -- same shape, orientation, values -- so water registers with
terrain in Godot. Plus the manifest's colormap ranges are correct.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import warp as wp
import xarray as xr
import zarr

from solver.core.channels import eta_from_h
from solver.io.viewer_export import export_frames, render_eta
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


def _make_channel_run(tmp_path):
    """A run with a sub-grid channel band down one column (M6 geometry, tiny grid).

    A band rather than a uniform width, so "channel cell" and "floodplain cell" are
    both present and the counts in the manifest mean something.
    """
    ny, nx = 16, 20
    yy, xx = np.mgrid[0:ny, 0:nx]
    # A valley sloping north so water runs, with the channel down its floor.
    bed = (0.4 * yy + 0.05 * (xx - nx / 2) ** 2).astype(np.float32)
    width = np.zeros((ny, nx), dtype=np.float32)
    depth = np.zeros((ny, nx), dtype=np.float32)
    width[:, nx // 2] = 8.0  # dx = 20 -> a channel narrower than its cell
    depth[:, nx // 2] = 1.0  # h_bf = 8*1/20 = 0.4 m of cell-mean depth
    (tmp_path / "w.r32").write_bytes(width.astype("<f4").tobytes())
    (tmp_path / "d.r32").write_bytes(depth.astype("<f4").tobytes())
    scn = Scenario(
        name="export_channels",
        dx=20.0,
        end_time=600.0,
        output_every=150.0,
        dt_max=5.0,
        rain_mm_hr=150.0,
        rain_duration=600.0,
        channel_width_field=str(tmp_path / "w.r32"),
        channel_depth_field=str(tmp_path / "d.r32"),
    )
    zarr_path = tmp_path / "chan.zarr"
    run_simulation(scn, bed, zarr_path, device="cpu", verbose=False)
    return zarr_path


def test_the_drawn_surface_is_the_storage_curve_overbank_and_the_bank_below_it(tmp_path):
    """`render_eta` is the renderable projection of the physical curve, and only that.

    Above bank full it *is* `channels.eta_from_h`; below it, it draws the bank instead
    of the true surface, which is under the floodplain bed on a terrain that carries no
    sub-grid trench. The three properties that make it safe to render: it agrees with
    the physics exactly overbank, it never draws *below* the true surface (the residual
    is one-sided, so the manifest's single number bounds it), and it is continuous at
    bank full so scrubbing a rising flood does not step.
    """
    dx = 20.0
    z = np.full(9, 100.0)
    w = np.full(9, 8.0)
    d = np.full(9, 1.0)
    h_bf = 8.0 * 1.0 / dx  # 0.4
    h = np.array([0.0, 0.05, 0.2, 0.399, h_bf, 0.401, 0.6, 1.0, 3.0])

    drawn = render_eta(h, z, w, d, dx)
    true = eta_from_h(h, z, w, d, dx)
    over = h > h_bf

    assert np.allclose(drawn[over], true[over], rtol=0, atol=0)  # exact overbank
    assert np.all(drawn[~over] == z[~over])  # the bank, flat
    assert np.all(drawn >= true - 1e-12)  # one-sided: never below the true surface
    # Continuous at bank full, from both sides.
    assert drawn[3] == pytest.approx(drawn[5], abs=2e-3)
    assert drawn[4] == z[4]
    # The worst case is a barely-wet channel: drawn at the bank, truly a depth `d` down.
    assert drawn[1] - true[1] == pytest.approx(d[1] - h[1] * dx / w[1], rel=1e-12)


def test_a_cell_without_a_channel_is_drawn_exactly_as_before(tmp_path):
    """No channel, no change: the M1 relation `z + h`, bit for bit.

    This is what keeps every pre-M6 run's picture identical -- the same argument the
    solver makes for its unarmed paths.
    """
    z = np.array([1.0, 250.0, -3.5])
    h = np.array([0.0, 0.001, 2.25])
    zeros = np.zeros(3)
    assert np.array_equal(render_eta(h, z, zeros, zeros, 20.0), z + h)
    # Half a channel is no channel (the solver normalises to w = d = 0, but a field pair
    # read off an older store must not select the curve on width alone).
    assert np.array_equal(render_eta(h, z, np.full(3, 8.0), zeros, 20.0), z + h)
    assert np.array_equal(render_eta(h, z, zeros, np.full(3, 1.0), 20.0), z + h)


def test_static_ships_the_channel_geometry_beside_the_bed(tmp_path):
    """The viewer cannot tell a river cell from a floodplain cell without these.

    Same static mechanism as the bed, same tile layout, same entry shape -- which is
    what "anything the composite depends on travels through the manifest" means.
    """
    zarr_path = _make_channel_run(tmp_path)
    out = tmp_path / "frames"
    export_frames(zarr_path, out)

    ds = xr.open_zarr(zarr_path, consolidated=False)
    manifest = json.loads((out / "manifest.json").read_text())
    static = manifest["static"]

    assert static["fields"] == ["bed", "channel_width", "channel_depth"]
    for name in ("channel_width", "channel_depth"):
        got = np.fromfile(out / static["files"][name], dtype="<f4")
        expected = ds[name].values.astype("<f4")
        assert np.array_equal(got.reshape(expected.shape), expected)

    chan = static["channel"]
    w, d = ds["channel_width"].values, ds["channel_depth"].values
    assert chan["cells"] == int(((w > 0) & (d > 0)).sum()) == 16  # one column of 16
    assert chan["width_max_m"] == float(w.max()) == 8.0
    assert chan["depth_max_m"] == float(d.max()) == 1.0
    assert chan["dx_m"] == manifest["dx"]


def test_channel_geometry_is_tiled_with_the_frame_layout(tmp_path):
    """Tiled export: the channel fields ride the frames' tile grid, like the bed."""
    zarr_path = _make_channel_run(tmp_path)
    out = tmp_path / "frames_tiled"
    export_frames(zarr_path, out, tile_size=8)

    ds = xr.open_zarr(zarr_path, consolidated=False)
    manifest = json.loads((out / "manifest.json").read_text())
    tiles = manifest["tile_grid"]["tiles"]
    static = manifest["static"]

    assert static["files"] == {}
    for name in ("bed", "channel_width", "channel_depth"):
        names = static["tiles"][name]
        assert len(names) == len(tiles)
        expected = ds[name].values.astype(np.float32)
        rebuilt = np.zeros(expected.shape, dtype=np.float32)
        for t, fname in zip(tiles, names, strict=True):
            block = np.fromfile(out / fname, dtype="<f4").reshape(t["height"], t["width"])
            rebuilt[t["y"] : t["y"] + t["height"], t["x"] : t["x"] + t["width"]] = block
        assert np.array_equal(rebuilt, expected)


def test_the_manifest_measures_how_far_above_the_true_surface_the_river_is_drawn(tmp_path):
    """The picture declares its own approximation, measured on the run's own frames.

    Not bounded by `d`: a bound is a property of the geometry, this is a property of
    this run's frames -- and it is taken over *all* of them, because the worst in-bank
    cell is a barely-wet one and the last frame is not where the flood is.
    """
    zarr_path = _make_channel_run(tmp_path)
    out = tmp_path / "frames"
    export_frames(zarr_path, out)

    ds = xr.open_zarr(zarr_path, consolidated=False)
    manifest = json.loads((out / "manifest.json").read_text())
    chan = manifest["static"]["channel"]

    dx = float(manifest["dx"])
    h_dry = float(manifest["h_dry"])
    z = ds["bed"].values
    w, d = ds["channel_width"].values, ds["channel_depth"].values
    has = (w > 0) & (d > 0)
    h_bf = np.where(has, w * d / dx, 0.0)

    best = (0.0, 0, -1)
    for i in range(manifest["n_frames"]):
        h = ds["depth"].isel(time=i).values
        in_bank = has & (h >= h_dry) & (h <= h_bf)
        if not in_bank.any():
            continue
        gap = float((render_eta(h, z, w, d, dx) - eta_from_h(h, z, w, d, dx))[in_bank].max())
        if gap > best[0]:
            best = (gap, int(in_bank.sum()), i)

    assert best[1] > 0, "fixture must wet the channel below bank full somewhere"
    assert chan["in_bank_offset_m"] == pytest.approx(best[0], rel=1e-12)
    assert chan["in_bank_cells"] == best[1]
    assert chan["frame"] == best[2]
    # The offset is real and bounded by the bank-full depth, which is what makes it a
    # declaration rather than an alarm.
    assert 0.0 < chan["in_bank_offset_m"] <= chan["depth_max_m"]


def test_a_channel_free_run_ships_no_channel_geometry(tmp_path):
    """No channels, no extra keys and no extra files: the M6 manifest, key for key."""
    zarr_path = _make_run(tmp_path)
    out = tmp_path / "frames"
    export_frames(zarr_path, out)

    manifest = json.loads((out / "manifest.json").read_text())
    static = manifest["static"]
    assert static["fields"] == ["bed"]
    assert "channel" not in static
    assert list(static["files"]) == ["bed"]
    assert not any("channel" in p.name for p in out.iterdir())


def test_untiled_export_bytes_are_unchanged_by_the_m6_export(tmp_path):
    """The demo-scale path must be byte-for-byte what M2 wrote."""
    zarr_path = _make_run(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    export_frames(zarr_path, a)  # default tile_size (512) -> one tile
    export_frames(zarr_path, b, tile_size=4096)

    for fr in json.loads((a / "manifest.json").read_text())["frames"]:
        name = fr["files"]["depth"]
        assert (a / name).read_bytes() == (b / name).read_bytes()
