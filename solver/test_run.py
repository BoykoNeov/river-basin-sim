"""Integration test: run_simulation -> Zarr store -> xarray read-back (M1, CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import warp as wp
import xarray as xr

from solver.core.massbalance import MASS_GATE
from solver.io.config import ConfigError
from solver.run import Scenario, main, run_simulation

wp.init()


def test_run_writes_valid_zarr_and_conserves_mass(tmp_path):
    # A shallow bowl so rain collects and water moves toward the centre.
    ny = nx = 24
    yy, xx = np.mgrid[0:ny, 0:nx]
    bed = (((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) * 0.02).astype(np.float32)

    scn = Scenario(
        name="test_bowl_rain",
        dx=20.0,
        end_time=600.0,
        output_every=150.0,
        dt_max=10.0,
        rain_mm_hr=100.0,
        rain_duration=300.0,
        initial_depth=0.0,
    )
    out = tmp_path / "r.zarr"
    ledger = run_simulation(scn, bed, out, device="cpu", verbose=False)

    # Mass gate holds.
    assert ledger.max_rel_error < MASS_GATE

    # Store opens with xarray and has the §7.2 shape.
    ds = xr.open_zarr(out, consolidated=False)
    n_frames = int(round(scn.end_time / scn.output_every)) + 1  # incl. t=0
    assert ds["depth"].shape == (n_frames, ny, nx)
    assert set(ds.data_vars) >= {"depth", "u", "v", "bed"}
    assert ds.attrs["scheme"] == "local_inertial"
    assert float(ds["time"][0]) == 0.0
    assert float(ds["time"][-1]) == 600.0

    # Bed round-trips exactly; rain produced standing water; no NaNs anywhere.
    assert np.allclose(ds["bed"].values, bed)
    assert float(ds["depth"].isel(time=-1).sum()) > 0.0
    assert np.isfinite(ds["depth"].values).all()

    # Mass series recorded to attrs for the viewer.
    assert ds.attrs["mass_max_rel_error"] < MASS_GATE
    assert len(ds.attrs["mass_balance_series"]) >= 1


def test_run_is_bitwise_deterministic(tmp_path):
    """Determinism is a locked invariant (HANDOFF §8/§12): two runs must be
    bitwise identical. Δt derives only from state (atomic-max, order-independent)
    and the mass sum is host-side float64 -- no nondeterministic float atomics."""
    ny = nx = 20
    yy, xx = np.mgrid[0:ny, 0:nx]
    bed = (((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) * 0.03).astype(np.float32)
    scn = Scenario(
        name="det",
        dx=15.0,
        end_time=450.0,
        output_every=150.0,
        dt_max=8.0,
        rain_mm_hr=80.0,
        rain_duration=300.0,
    )
    a = run_simulation(scn, bed, tmp_path / "a.zarr", device="cpu", verbose=False)
    b = run_simulation(scn, bed, tmp_path / "b.zarr", device="cpu", verbose=False)

    da = xr.open_zarr(tmp_path / "a.zarr", consolidated=False)
    db = xr.open_zarr(tmp_path / "b.zarr", consolidated=False)
    assert np.array_equal(da["depth"].values, db["depth"].values)
    assert np.array_equal(da["u"].values, db["u"].values)
    assert np.array_equal(da["v"].values, db["v"].values)
    assert a.max_rel_error == b.max_rel_error


def test_main_writes_error_status_on_bad_config(tmp_path):
    """A scope-gate ConfigError must be reported via status.json, not a silent exit
    (else the viewer polls forever). The error is written *and* re-raised."""
    cfg = tmp_path / "bad.toml"
    cfg.write_text('[meta]\nscheme = "hllc_fv"\n', encoding="utf-8")  # M4, rejected in M2
    status_path = tmp_path / "status.json"

    with pytest.raises(ConfigError):
        main(
            [
                "--config",
                str(cfg),
                "--out",
                str(tmp_path / "out.zarr"),
                "--status",
                str(status_path),
            ]
        )

    rec = json.loads(status_path.read_text(encoding="utf-8"))
    assert rec["state"] == "error"
    assert "hllc_fv" in rec["message"]
