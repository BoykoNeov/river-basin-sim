"""Solver entry point (M2): config/in-code scenario -> run -> results + viewer stream.

With M2 the loop closes: the run is driven by a §7.1 TOML config (``--config``) or
the in-code demo scenario, writes the canonical Zarr (§7.2), reports progress via
``status.json`` (§7.4), and post-processes the Zarr into the lean per-frame viewer
stream (§7.3) that Godot reads. The demo runs uniform rainfall over the real M0
terrain tile (``data/tiles/demo``) with closed boundaries.

CLI::

    uv run python -m solver.run --config scenarios/demo_basin_rain.toml
    uv run python -m solver.run                    # demo: M0 tile + uniform rain
    uv run python -m solver.run --tiles data/tiles/demo --out data/results/demo.zarr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from solver.core.grid import Grid
from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State
from solver.io.config import Scenario, load_config
from solver.io.fields import load_field
from solver.io.status import StatusWriter
from solver.io.viewer_export import export_frames
from solver.io.zarr_writer import ZarrWriter
from solver.processes.inflow import InflowInjector

# Scenario is defined in solver.io.config (the §7.1 contract); re-exported here so
# existing callers (`from solver.run import Scenario`) keep working.
__all__ = ["Scenario", "load_config", "run_simulation", "main"]

EPS_T = 1e-6  # time-comparison tolerance (seconds)


def load_r32_bed(tiles_dir: str | Path) -> tuple[np.ndarray, dict]:
    """Load the first tile of an M0 ``tiles.json`` manifest as a bed array.

    Returns the ``(ny, nx)`` float32 bed (metres) plus the manifest dict (for dx,
    CRS, bounds). The ``.r32`` is raw little-endian row-major float32 (HANDOFF §7).
    """
    tiles_dir = Path(tiles_dir)
    manifest = json.loads((tiles_dir / "tiles.json").read_text())
    t0 = manifest["tiles"][0]
    h, w = int(t0["height"]), int(t0["width"])
    raw = np.fromfile(tiles_dir / t0["file"], dtype="<f4", count=h * w)
    bed = raw.reshape(h, w).astype(np.float32)
    return bed, manifest


def pick_device(requested: str | None) -> str:
    """Resolve the Warp device: honour a request, else CUDA if present, else CPU."""
    wp.init()
    if requested:
        return requested
    return "cuda:0" if wp.get_cuda_devices() else "cpu"


def _next_event_time(t: float, events: list[float]) -> float:
    """Smallest event time strictly after ``t`` (or +inf if none)."""
    future = [e for e in events if e > t + EPS_T]
    return min(future) if future else float("inf")


def run_simulation(
    scenario: Scenario,
    bed: np.ndarray,
    out_path: str | Path,
    *,
    device: str = "cpu",
    verbose: bool = True,
    status: StatusWriter | None = None,
) -> MassLedger:
    """Run the local-inertial solver and stream results to a Zarr store.

    Timestep is adaptive and derived from state (determinism, §8/§12) but clamped
    so a step never crosses an output time or the rainfall on/off boundary -- so
    frames land exactly on ``output_every`` and each step is either fully raining
    or fully dry (exact source accounting).

    If ``status`` is given, a ``running`` record is written at each output frame
    (§7.4). ``status`` is a read-only progress observer -- it never touches Δt or
    the Zarr, so determinism is unaffected.
    """
    if scenario.dx is None:
        raise ValueError("scenario.dx is unresolved; fill it from the tile manifest first")
    grid = Grid(ny=bed.shape[0], nx=bed.shape[1], dx=scenario.dx)
    manning = load_field(scenario.manning_field, grid, scalar=scenario.manning_n)
    st = State.from_bed(
        bed, dx=scenario.dx, depth=scenario.initial_depth, manning=manning, device=device
    )

    # --- M3 sources/sinks -------------------------------------------------------
    # Infiltration (constant-rate sink, mm/hr -> m/s); armed only when nonzero.
    infil = load_field(scenario.infiltration_field, grid, scalar=scenario.infiltration_mm_hr)
    infil_m_s = infil / 1000.0 / 3600.0
    if scenario.infiltration_field is not None or float(infil_m_s.max()) > 0.0:
        st.set_infiltration(infil_m_s)
    # Spatial rainfall field (mm/hr -> m/s); uniform rain keeps the scalar path.
    rain_is_field = scenario.rain_type == "field"
    rain_field_sum_m_s = 0.0
    if rain_is_field:
        rain_field = load_field(scenario.rain_field, grid) / 1000.0 / 3600.0
        st.set_rain_field(rain_field)
        rain_field_sum_m_s = float(rain_field.astype(np.float64).sum())
    # Inflow hydrographs (prescribed discharge point sources).
    injector = InflowInjector(scenario.inflows, grid, device) if scenario.inflows else None

    ledger = MassLedger.from_state(st)

    n_frames = int(round(scenario.end_time / scenario.output_every)) + 1
    attrs = {
        "scheme": "local_inertial",
        "crs": scenario.crs,
        "dx": scenario.dx,
        "units": {"depth": "m", "u": "m/s", "v": "m/s", "time": "s", "bed": "m"},
        "scenario": scenario.name,
        "rain_type": scenario.rain_type,
        "rain_mm_hr": scenario.rain_mm_hr,
        "rain_duration_s": scenario.rain_duration,
        "manning_n": scenario.manning_n,
        "infiltration_mm_hr": scenario.infiltration_mm_hr,
        "end_time_s": scenario.end_time,
        "output_every_s": scenario.output_every,
    }
    writer = ZarrWriter(out_path, grid, n_frames, attrs)
    writer.write_bed(bed)

    # Frame at t = 0 (baseline).
    u0, v0 = st.velocities_numpy()
    writer.append(0.0, st.depth_numpy(), u0, v0)

    t = 0.0
    next_output = scenario.output_every
    # Event times a step must not cross: output cadence, rain end, end of run, and
    # each inflow-hydrograph breakpoint (so the sampled discharge stays faithful).
    output_times = [scenario.output_every * k for k in range(1, n_frames)]
    inflow_events = injector.breakpoints() if injector else []
    events = output_times + [scenario.rain_duration, scenario.end_time] + inflow_events

    while t < scenario.end_time - EPS_T:
        dt = compute_dt(st, alpha=scenario.alpha, dt_max=scenario.dt_max)
        dt = min(dt, _next_event_time(t, events) - t)

        # Inject inflow hydrographs for this step (midpoint discharge -> volume).
        if injector is not None:
            ledger.add_inflow(injector.apply(st, t, dt))

        raining = t < scenario.rain_duration - EPS_T

        if rain_is_field:
            step(st, dt=dt, rain_scale=(1.0 if raining else 0.0))
            if raining:
                ledger.add_inflow(rain_field_sum_m_s * dt * grid.cell_area)
        else:
            rain = scenario.rain_m_s if raining else 0.0
            step(st, dt=dt, rain=rain)
            if rain > 0.0:
                ledger.add_rain_step(rain, dt, grid.n_cells)
        t += dt

        if t >= next_output - EPS_T and next_output <= scenario.end_time + EPS_T:
            rec = ledger.record(st, t)
            u, v = st.velocities_numpy()
            writer.append(t, st.depth_numpy(), u, v)
            h_max = float(st.h.numpy().max())
            if verbose:
                print(f"  t={t:8.1f}s  h_max={h_max:6.3f}m  mass_rel_err={rec.rel_error:.2e}")
            if status is not None:
                status.write(
                    "running",
                    sim_time=t,
                    message=f"t={t:.0f}s  h_max={h_max:.3f}m  mass_rel_err={rec.rel_error:.1e}",
                )
            next_output += scenario.output_every

    writer.finalize(ledger.as_attrs())
    if verbose:
        print(f"done: {out_path}")
        print(f"  frames        : {len(ledger.series)}")
        print(f"  mass max rel  : {ledger.max_rel_error:.2e}  (gate {MASS_GATE:.0e})")
    if ledger.max_rel_error >= MASS_GATE:
        print(f"  WARNING: mass-balance gate exceeded ({ledger.max_rel_error:.2e})")
    return ledger


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the local-inertial solver (M2: config + loop).")
    p.add_argument("--config", default=None, help="§7.1 scenario TOML (overrides the demo flags)")
    p.add_argument("--tiles", default=None, help="M0 tiles dir (tiles.json); default from config")
    p.add_argument("--out", default="data/results/demo.zarr", help="output Zarr store")
    p.add_argument("--status", default=None, help="status.json path; default <out-dir>/status.json")
    p.add_argument("--frames-dir", default=None, help="frames/ dir; default <out-dir>/frames")
    p.add_argument("--no-frames", action="store_true", help="skip the §7.3 viewer export")
    p.add_argument("--device", default=None, help="warp device (cpu / cuda:0); auto if unset")
    p.add_argument("--end-time", type=float, default=3600.0, help="sim seconds (no --config)")
    p.add_argument("--output-every", type=float, default=300.0, help="write cadence (no --config)")
    p.add_argument("--rain-mm-hr", type=float, default=50.0, help="(no --config)")
    p.add_argument("--rain-duration", type=float, default=1800.0, help="(no --config)")
    return p.parse_args(argv)


def _resolve_scenario(args: argparse.Namespace) -> tuple[Scenario, np.ndarray]:
    """Build the run Scenario (from --config or the demo flags) and load its bed.

    ``dx``/``crs`` unset by the config inherit from the tile manifest (§7.1).
    """
    if args.config:
        scenario = load_config(args.config)
        if args.tiles:
            scenario.tiles_dir = args.tiles
    else:
        scenario = Scenario(
            tiles_dir=args.tiles or Scenario().tiles_dir,
            end_time=args.end_time,
            output_every=args.output_every,
            rain_mm_hr=args.rain_mm_hr,
            rain_duration=args.rain_duration,
        )
    bed, manifest = load_r32_bed(scenario.tiles_dir)
    if scenario.dx is None:
        scenario.dx = float(manifest["dx_m"])
    if not scenario.crs:
        scenario.crs = manifest.get("crs", "")
    return scenario, bed


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status) if args.status else out_path.parent / "status.json"
    frames_dir = Path(args.frames_dir) if args.frames_dir else out_path.parent / "frames"

    # Create the status channel FIRST and keep everything that can fail -- config
    # parsing (the §7.1 scope gate), tile loading, warp init -- inside the try, so
    # any failure is reported as state="error" instead of a silent exit that leaves
    # the viewer polling forever (§7.4). end_time is patched in once resolved.
    status = StatusWriter(status_path, end_time=1.0)
    status.write("starting", message="resolving scenario")
    try:
        device = pick_device(args.device)
        scenario, bed = _resolve_scenario(args)
        status.end_time = scenario.end_time
        print(
            f"River Basin M2 solver | device={device} | grid={bed.shape} "
            f"dx={scenario.dx:.2f}m | scenario={scenario.name}"
        )
        status.write("starting", message=f"{scenario.name}: {bed.shape} @ dx={scenario.dx:.2f}m")

        ledger = run_simulation(scenario, bed, out_path, device=device, status=status)
        if not args.no_frames:
            status.write("writing", sim_time=scenario.end_time, message="exporting viewer frames")
            manifest = export_frames(out_path, frames_dir)
            print(f"  viewer frames : {manifest}")
        status.write(
            "done",
            sim_time=scenario.end_time,
            message=f"mass_max_rel_err={ledger.max_rel_error:.2e}",
        )
    except Exception as e:  # noqa: BLE001 -- report to the viewer, then re-raise
        status.write("error", message=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
