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
    the same reason. This is the record of the activations that produce the
    ``bed_change`` field; that field is asserted two tests below.

    Rain over a bowl, so there is real flow to transport with, and the water mass
    gate must stay green **with the bed moving** -- "the bed update quietly ate
    water" is precisely the failure this milestone can produce.

    **What transports here is the thin sheet at the wet/dry guard, and that regime is
    on notice.** MPM's shear diverges as ``h -> H_DRY``, so this bowl scours 5.6 cm in
    its first 150 s activation and almost nothing afterwards (M7 plan §4, measured at
    build step 6). Nothing below depends on the *size* of that -- only that the bed
    moved at all, so the plumbing is exercised -- but if step 8 gives the law a depth
    guard, this scenario stops transporting and the two "something moved" assertions
    fail here rather than where the law changed. Re-home it to channel flow then; do
    not weaken the assertions.
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


def _sediment_bowl(**over) -> Scenario:
    """The step-5 bowl scenario, parameterised.

    **What transports here is the thin sheet at the wet/dry guard, and that regime is
    outside MPM** -- the shear diverges as ``h -> H_DRY`` (M7 plan §4, and the step-5
    test above says it at length). Relative submergence here is ``h/d50 = 0.5``: the
    sheet is shallower than one grain.

    **Build step 8 decided not to guard the law**, and this helper is one of the
    reasons. A relative-submergence cut-off would keep every test below green while
    silently changing what they test -- ``test_a_grain_size_the_flow_cannot_move_...``
    passes today *because* ``theta < theta_c``, and a ``h >= k*d50`` guard would make
    it pass for a second, unrelated reason and stop exercising the threshold at all.
    The gates keep clear of the regime instead, and assert that they do
    (``validation.test_morphology_gates.test_the_gate_scenarios_transport_inside_the_laws_range``).
    So the numbers this helper produces are **plumbing evidence, not physics**: they
    show the bed moves, the store records it and the ledger balances, and nothing
    here should be quoted as a transport result.

    The contingency stands if that decision is ever revisited: every test built on
    this helper asserts the bed *moved*, so a depth guard would fail them here rather
    than where the law changed -- re-home them to channel flow then, do not weaken the
    assertions.
    """
    kw = dict(
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
    kw.update(over)
    return Scenario(**kw)


def test_bed_and_bed_change_reconstruct_the_bed_the_run_finished_on(tmp_path):
    """M7 step 7, the keystone: ``bed`` is the initial bed and ``bed_change`` moves it.

    §7.2 stores the moving part *beside* ``bed`` rather than promoting ``bed`` to
    ``(T, Y, X)``, so the two only mean anything together -- and they are consistent
    by construction (M7 plan §1.1): ``z0`` is captured after barriers from the same
    array ``bed`` is written from, and ``z = float32(z0 + dz_cum)`` is rebuilt from
    it at every activation. This asserts that chain end to end through the store,
    which is what catches a barrier or datum mistake in one shot.

    The volume cross-check is against the **gross** displaced volume the sediment
    ledger reports, not the net: the net is identically zero in a domain closed to
    bedload (build step 6), so a net-against-net comparison would be two near-zeros
    agreeing about nothing. What the store must reproduce is the scale that says the
    bed moved -- and it reproduces it only to what a float32 *rendering* of the f64
    ``dz_cum`` costs, which is exactly what the tolerance measures.
    """
    scn = _sediment_bowl()
    run_simulation(scn, _bowl(16, 16), tmp_path / "s.zarr", device="cpu", verbose=False)
    ds = xr.open_zarr(tmp_path / "s.zarr", consolidated=False)

    dz = ds["bed_change"].values
    assert dz.shape == ds["depth"].shape
    assert np.isfinite(dz).all()
    # t = 0 is the pristine bed: nothing has moved, exactly.
    assert np.array_equal(dz[0], np.zeros_like(dz[0]))
    assert np.abs(dz[-1]).max() > 0.0, "the bed moved but the field says it did not"

    # The final output frame lands on the final activation, so the field and that
    # activation's record describe the same bed.
    last = ds.attrs["morphology"][-1]
    assert last["time"] == scn.end_time
    solid = scn.dx**2 * (1.0 - 0.4)  # the default [sediment] porosity
    gross = ds.attrs["sediment_balance_series"][-1]["gross_volume"]
    assert float(np.abs(dz[-1]).sum()) * solid == pytest.approx(gross, rel=1e-5)
    assert float(dz[-1].min()) == pytest.approx(last["dz_min_m"], rel=1e-6)
    assert float(dz[-1].max()) == pytest.approx(last["dz_max_m"], rel=1e-6)

    # And the pair adds up to a bed: `bed` never moves, so the terrain a reader
    # reconstructs for the last frame is bed + bed_change[-1].
    final_bed = ds["bed"].values + dz[-1]
    assert np.isfinite(final_bed).all()
    assert not np.array_equal(final_bed, ds["bed"].values)
    assert np.array_equal(ds["bed"].values, _bowl(16, 16))  # still the *initial* bed


def test_bed_change_is_datum_free(tmp_path):
    """A difference of elevations has no origin, so ``[grid] datum`` must not touch it.

    ``bed`` goes through :func:`~solver.core.datum.unshift_bed` on the way out and
    ``bed_change`` deliberately does not (M7 plan §1.7) -- which is what lets a reader
    add them without knowing the datum. **The discriminator is scale, not tolerance**:
    un-shifting the change by mistake would offset it by ``z_ref`` -- 9 m here --
    against a bed that moves by centimetres, three orders away from anything the two
    runs can differ by legitimately. And they do differ: a datum shift changes float32
    ``eta = h + z`` arithmetic, which this scenario is unusually sensitive to (MPM at
    the wet/dry guard, M7 plan §4), so the runs agree to ~5% of the largest change
    rather than bitwise. M5 measured the same shape of thing on EA Test 1 -- agreement
    to three decimals, not bit for bit.
    """
    bed = _bowl(16, 16) + 9.5
    plain = _sediment_bowl(name="sed_plain")
    shifted = _sediment_bowl(name="sed_datum", datum="auto")
    run_simulation(plain, bed, tmp_path / "p.zarr", device="cpu", verbose=False)
    run_simulation(shifted, bed, tmp_path / "d.zarr", device="cpu", verbose=False)

    dp = xr.open_zarr(tmp_path / "p.zarr", consolidated=False)
    dd = xr.open_zarr(tmp_path / "d.zarr", consolidated=False)
    assert dp.attrs["datum_shift_m"] == 0.0
    assert dd.attrs["datum_shift_m"] == 9.0  # the shift really was applied
    assert np.allclose(dp["bed"].values, dd["bed"].values)  # both store true elevations

    a, b = dp["bed_change"].values[-1], dd["bed_change"].values[-1]
    assert np.abs(a).max() < 0.1  # centimetres, three orders below z_ref = 9 m
    assert np.abs(a - b).max() < 0.01  # ... and the two runs are within one of those


def test_below_threshold_the_stored_bed_change_is_exactly_zero(tmp_path):
    """Boulders do not move: no transport must render as *bit-exact* zero, not small.

    MPM's threshold is the cheap sharp test the law was chosen for (M7 plan §1.2) --
    here it also says the store path invents nothing on its way through the f64 ->
    f32 cast. The array is still created, because the run *has* morphology: "armed
    and nothing moved" and "no morphology" are different statements about a run and
    the store makes both readable.
    """
    scn = _sediment_bowl(name="sed_boulders", sediment_d50_m=1.0)
    run_simulation(scn, _bowl(16, 16), tmp_path / "z.zarr", device="cpu", verbose=False)
    ds = xr.open_zarr(tmp_path / "z.zarr", consolidated=False)

    assert "bed_change" in ds
    assert np.array_equal(ds["bed_change"].values, np.zeros_like(ds["bed_change"].values))
    assert all(r["applied_m3"] == 0.0 for r in ds.attrs["morphology"])
    assert np.array_equal(ds["bed"].values, _bowl(16, 16))


def test_a_run_without_sediment_stores_no_bed_change(tmp_path):
    """The invariant every milestone holds: unarmed is the previous store, untouched."""
    scn = Scenario(dx=20.0, end_time=300.0, output_every=150.0, dt_max=5.0, rain_mm_hr=60.0)
    run_simulation(scn, _bowl(12, 12), tmp_path / "n.zarr", device="cpu", verbose=False)
    ds = xr.open_zarr(tmp_path / "n.zarr", consolidated=False)
    assert "bed_change" not in ds
    assert "morphology" not in ds.attrs


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


# --- M7 build step 8: the morphological-CFL gate at the scenario level ------------
# The unit-level pieces live in `solver.core.test_sediment` (the ratio) and
# `solver.processes.test_morphology` (the per-activation measurement); the physics
# gate lives in `validation.test_bed_wave`. What is left, and what M7 plan §3 means
# by *"asserted, not just printed"*, is that a whole scenario carrying an
# over-Courant interval says so out loud instead of finishing quietly.


def test_a_scenario_inside_the_morphological_courant_gate_says_nothing(tmp_path, capsys):
    """The step-5 bowl is under the gate, so the warning must stay silent.

    Half of a warning's value is that it is not always on. This also pins the bowl at
    Courant 0.07 -- comfortably inside -- so a future change that pushed the demo
    scenarios over would surface here rather than in a reviewer's eyeball.
    """
    scn = _sediment_bowl()
    run_simulation(scn, _bowl(16, 16), tmp_path / "ok.zarr", device="cpu", verbose=True)
    out = capsys.readouterr().out
    print(out)
    assert "morphological Courant" not in out
    assert "bed courant" in out  # ... but the number is still reported


def test_a_scenario_over_the_morphological_courant_gate_warns(tmp_path, capsys):
    """An over-Courant run finishes clean on every other gauge, and warns on this one.

    The same bowl with the rain held on for the whole run and a 2400 s bed interval:
    the bed wave then crosses ~3 cells per activation. Nothing else notices -- the
    mass balance and the sediment balance both stay far inside their gates, the store
    is written, `status.json` reaches `done` -- which is exactly why this warning is
    printed unconditionally rather than under `verbose`.
    """
    scn = _sediment_bowl(end_time=6000.0, output_every=3000.0, rain_duration=6000.0,
                         sediment_interval_s=2400.0)  # fmt: skip
    ledger = run_simulation(scn, _bowl(16, 16), tmp_path / "bad.zarr", device="cpu",
                            verbose=False)  # fmt: skip
    out = capsys.readouterr().out
    print(out)

    assert "WARNING: morphological Courant" in out
    assert "2400" in out, "the warning should name the interval it is complaining about"
    # Every other gauge is happy, which is the finding this test carries.
    assert ledger.max_rel_error < MASS_GATE

    ds = xr.open_zarr(tmp_path / "bad.zarr", consolidated=False)
    series = ds.attrs["morphology"]
    assert max(r["courant"] for r in series) > 1.0
    assert ds.attrs["sediment_max_rel_error"] < SEDIMENT_GATE


def test_the_courant_diagnostic_samples_the_flow_and_a_drained_run_reads_zero(tmp_path, capsys):
    """A carried limitation of the diagnostic, pinned rather than remembered.

    The transport integral is accumulated over the interval (M7 plan §1.3); the
    Courant number is **not**. :func:`solver.core.sediment.celerity_field` evaluates
    the flow the state is carrying *at the activation instant*, so an interval during
    which a flood arrived, moved the bed and drained away reports a celerity of zero
    -- and therefore a Courant number of zero -- having moved the bed by more than the
    over-Courant run above.

    Measured here: rain for 600 s of a 6000 s run with a 3000 s bed interval moves
    577 m^3 of sediment and reports Courant 0.000, while the same scenario with the
    rain held on reports 5.05. The warning is a **floor on what you can trust, not a
    certificate**: a silent run whose forcing is spiky has not been checked. Making it
    a true interval maximum means tracking the celerity every fast step, which is a
    full-field host reduction per step -- out of scope for a diagnostic, and named in
    M7 plan §4 instead.
    """
    scn = _sediment_bowl(end_time=6000.0, output_every=3000.0, sediment_interval_s=3000.0)
    run_simulation(scn, _bowl(16, 16), tmp_path / "spiky.zarr", device="cpu", verbose=False)
    out = capsys.readouterr().out
    print(out)

    ds = xr.open_zarr(tmp_path / "spiky.zarr", consolidated=False)
    series = ds.attrs["morphology"]
    # The bed moved a lot ...
    assert ds.attrs["sediment_balance_series"][-1]["gross_volume"] > 100.0
    # ... and the diagnostic still read zero, so nothing warned.
    assert max(r["courant"] for r in series) == 0.0
    assert "WARNING: morphological Courant" not in out
