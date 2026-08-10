"""Integration test: run_simulation -> Zarr store -> xarray read-back (M1, CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import warp as wp
import xarray as xr

from solver.core.massbalance import MASS_GATE, SEDIMENT_GATE
from solver.io.config import ConfigError, Inflow
from solver.run import Scenario, field_memory_mb, main, run_simulation

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

    Uses temporal rainfall as the rejected feature. (It has outlived two earlier
    stand-ins: ``scheme='hllc_fv'`` opened at M4 and ``[[structures]]`` at M5 --
    pick a gate that is *still* shut when this test needs updating again.)
    """
    cfg = tmp_path / "bad.toml"
    cfg.write_text("[rainfall]\ntype = 'storm_cells'\n", encoding="utf-8")  # deferred
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
    assert "storm_cells" in rec["message"]


# --- M5: fixed_stage through the config -> run path -------------------------- #
def test_fixed_stage_scenario_runs_end_to_end(tmp_path):
    """A stage edge drives a run from the §7.1 config path, ledger and store intact.

    Covers the wiring the unit tests cannot: stage curves surviving load_config ->
    the datum shift -> State -> the HLLC ghost, plus the curve knots joining the
    scheduler's sync points.
    """
    ny = nx = 16
    bed = np.tile(np.linspace(1.0, 0.0, nx, dtype=np.float32), (ny, 1))
    scn = Scenario(
        name="stage_run",
        scheme="hllc_fv",
        dx=10.0,
        end_time=600.0,
        output_every=300.0,
        dt_max=2.0,
        alpha=0.45,
        rain_mm_hr=0.0,
        rain_duration=0.0,
        boundaries={"north": "closed", "south": "closed", "east": "fixed_stage", "west": "closed"},
        stage_curves={"east": [(0.0, 0.2), (600.0, 0.6)]},
    )
    ledger = run_simulation(scn, bed, tmp_path / "stage.zarr", device="cpu", verbose=False)
    assert ledger.max_rel_error < MASS_GATE

    ds = xr.open_zarr(tmp_path / "stage.zarr", consolidated=False)
    final = ds["depth"].isel(time=-1).values
    assert np.isfinite(final).all() and final.min() >= 0.0
    assert float(final.sum()) > 0.0, "the stage edge never let water in"
    # Ledger inflow arrives as a *negative* banked loss (the signed banking path).
    assert ledger.series[-1].outflow_cum < 0.0


def test_fixed_stage_is_rejected_on_the_local_inertial_scheme():
    """HLLC-only (M5 plan §1.4): a loud error on the in-code path too, not silence."""
    with pytest.raises(ValueError, match="hllc_fv"):
        Scenario(
            scheme="local_inertial",
            boundaries={
                "north": "fixed_stage",
                "south": "closed",
                "east": "closed",
                "west": "closed",
            },
            stage_curves={"north": [(0.0, 1.0)]},
        )


def test_fixed_stage_without_a_curve_is_rejected():
    with pytest.raises(ValueError, match="no stage curve"):
        Scenario(
            scheme="hllc_fv",
            boundaries={
                "north": "fixed_stage",
                "south": "closed",
                "east": "closed",
                "west": "closed",
            },
        )


def test_datum_shift_is_recorded_and_the_store_keeps_true_elevations(tmp_path):
    """[grid] datum steps in shifted coordinates but stores the real bed (§7.2)."""
    ny = nx = 12
    bed = (np.zeros((ny, nx), np.float32) + 9.5).astype(np.float32)
    scn = Scenario(
        name="datum_run",
        dx=10.0,
        datum="auto",
        end_time=200.0,
        output_every=200.0,
        dt_max=5.0,
        rain_mm_hr=20.0,
        rain_duration=200.0,
    )
    run_simulation(scn, bed, tmp_path / "d.zarr", device="cpu", verbose=False)
    ds = xr.open_zarr(tmp_path / "d.zarr", consolidated=False)
    assert ds.attrs["datum_shift_m"] == 9.0  # floor(min(bed))
    assert np.allclose(ds["bed"].values, bed)  # stored un-shifted


def test_a_sediment_scenario_runs_and_records_what_its_slow_clock_did(tmp_path):
    """M7 step 5: [sediment] now *runs* -- the inverse of the refusal it replaces.

    The run-loop half of the milestone in one assertion: the scenario is accepted,
    the morphology process is built and scheduled on its own cadence, and the store
    is self-describing about it -- the static configuration in ``sediment`` and the
    activation history in ``morphology``, beside the reservoir's release series for
    the same reason. The ``bed_change`` *field* is build step 7; this is the record
    of the activations that will produce it.

    Rain over a bowl, so there is real flow to transport with, and the water mass
    gate must stay green **with the bed moving** -- "the bed update quietly ate
    water" is precisely the failure this milestone can produce.
    """
    scn = Scenario(
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
    assert scn.has_sediment
    ledger = run_simulation(scn, _bowl(16, 16), tmp_path / "s.zarr", device="cpu", verbose=False)
    assert ledger.max_rel_error < MASS_GATE

    ds = xr.open_zarr(tmp_path / "s.zarr", consolidated=False)
    assert ds.attrs["sediment"]["law"] == "mpm"
    assert ds.attrs["sediment"]["interval_s"] == 150.0
    series = ds.attrs["morphology"]
    assert [r["time"] for r in series] == [150.0, 300.0, 450.0, 600.0]
    assert any(r["applied_m3"] != 0.0 for r in series), "no sediment moved at all"
    # Unbounded bed: nothing was refused, so the ledger is owed nothing (step 6).
    assert all(r["banked_m3"] == 0.0 for r in series)

    # Step 6: the second conserved substance has its own gauge, and it ships with the
    # run rather than living in the process object -- the same reason the water series
    # does. It is sampled at the *output* cadence (the morphology series above is the
    # activation one), so a reader can line it up frame for frame with the mass series.
    assert ds.attrs["sediment_max_rel_error"] < SEDIMENT_GATE
    sed_series = ds.attrs["sediment_balance_series"]
    assert [r["time"] for r in sed_series] == [0.0, 300.0, 600.0]
    assert sed_series[-1]["gross_volume"] > 0.0, "gross must say the bed actually moved"
    assert all(r["banked_volume"] == 0.0 for r in sed_series)


def test_field_memory_counts_the_two_widths_separately_for_morphology():
    """The f64 pair is the *largest* contributor, and a `count x 4` would hide it.

    M7's working set is six float32 arrays (``d50``, ``z0`` and the four face
    accumulators) plus **two float64** ones (``dz_cum``, ``dz_unapplied``) -- the
    ledger arrays a sub-millimetre bed increment needs (M7 plan §1.1). At 768^2 that
    pair alone is ~9 MB -- **two** arrays weighing exactly what the **four** face
    accumulators do -- so the print that exists to warn before a CUDA out-of-memory
    has to weigh the widths separately rather than count arrays at 4 bytes.
    """
    shape = (768, 768)
    cells = shape[0] * shape[1]
    mb = lambda b: b * cells / (1024.0 * 1024.0)  # noqa: E731
    assert field_memory_mb(shape) == pytest.approx(mb(7 * 4))
    assert field_memory_mb(shape, sediment=False) == field_memory_mb(shape)
    assert field_memory_mb(shape, sediment=True) == pytest.approx(mb(13 * 4 + 2 * 8))
    added = field_memory_mb(shape, sediment=True) - field_memory_mb(shape)
    assert added == pytest.approx(mb(6 * 4 + 2 * 8))
    # Two f64 arrays weigh what four f32 face accumulators do: counting arrays at a
    # single 4-byte width would under-report morphology by a third (~9 MB here).
    assert mb(2 * 8) == pytest.approx(mb(4 * 4)) == pytest.approx(9.0)
    assert field_memory_mb(shape, sediment=True) == pytest.approx(
        field_memory_mb(shape) + mb(6 * 4) + 9.0
    )
