"""Integration test: run_simulation -> Zarr store -> xarray read-back (M1, CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import warp as wp
import xarray as xr

from solver.core.massbalance import MASS_GATE
from solver.io.config import ConfigError, Inflow
from solver.run import Scenario, main, run_simulation

wp.init()


def _bowl(ny: int, nx: int) -> np.ndarray:
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) * 0.02).astype(np.float32)


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


def test_m3_paths_are_deterministic(tmp_path):
    """Determinism (§12) must hold for the new state-mutating M3 kernels too:
    an infiltration + inflow + open-boundary run repeated must be bitwise identical
    (single-writer kernels, no float atomics -- but assert it, don't assume it)."""
    bed = _bowl(20, 20)
    scn = Scenario(
        name="m3det",
        dx=20.0,
        end_time=400.0,
        output_every=200.0,
        dt_max=10.0,
        rain_mm_hr=60.0,
        rain_duration=200.0,
        infiltration_mm_hr=5.0,
        inflows=[Inflow(cell=(4, 4), hydrograph=[(0.0, 0.0), (400.0, 3.0)])],
        boundaries={"north": "closed", "south": "open", "east": "closed", "west": "closed"},
    )
    run_simulation(scn, bed, tmp_path / "a.zarr", device="cpu", verbose=False)
    run_simulation(scn, bed, tmp_path / "b.zarr", device="cpu", verbose=False)
    da = xr.open_zarr(tmp_path / "a.zarr", consolidated=False)
    db = xr.open_zarr(tmp_path / "b.zarr", consolidated=False)
    assert np.array_equal(da["depth"].values, db["depth"].values)


def test_infiltration_mm_hr_conversion(tmp_path):
    """Guard the run.py mm/hr -> m/s conversion (untested by the m/s kernel tests):
    a shallow, still, non-raining basin loses ~ rate_m_s * area * end_time. Catches
    a gross conversion slip (a missing /1000 or /3600 is orders of magnitude); the
    1% tolerance absorbs float32 field quantization (shallow h keeps it small)."""
    ny = nx = 12
    scn = Scenario(
        name="infil_conv",
        dx=10.0,
        end_time=200.0,
        output_every=200.0,
        dt_max=10.0,
        rain_mm_hr=0.0,
        rain_duration=0.0,
        infiltration_mm_hr=30.0,
        initial_depth=0.5,  # shallow -> fine ULP; removes ~1.7e-3 m << 0.5 (uncapped)
    )
    ledger = run_simulation(
        scn, np.zeros((ny, nx), np.float32), tmp_path / "c.zarr", device="cpu", verbose=False
    )
    rate_m_s = 30.0 / 1000.0 / 3600.0
    expected = rate_m_s * scn.end_time * (scn.dx**2) * ny * nx
    assert ledger.series[-1].outflow_cum == pytest.approx(expected, rel=1e-2)
    assert ledger.max_rel_error < MASS_GATE


def test_main_writes_error_status_on_bad_config(tmp_path):
    """A scope-gate ConfigError must be reported via status.json, not a silent exit
    (else the viewer polls forever). The error is written *and* re-raised.

    Uses ``[[structures]]`` (deferred to M5) as the rejected feature: since M4
    wired up ``scheme='hllc_fv'`` (no longer a config scope-gate error), the
    structures gate is the stable stand-in for "config asks for a deferred feature".
    """
    cfg = tmp_path / "bad.toml"
    cfg.write_text("[[structures]]\nkind = 'dam'\n", encoding="utf-8")  # M5, rejected now
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
    assert "structures" in rec["message"]
