"""Solver entry point (M1): in-code scenario -> run -> canonical Zarr results.

M1 is driven by a **minimal in-code** :class:`Scenario` rather than the §7.1 TOML
config -- the config loader and the subprocess/status protocol land in M2 when the
loop closes. The demo runs uniform rainfall over the real M0 terrain tile
(``data/tiles/demo``) with closed boundaries and writes ``results.zarr`` (§7.2)
plus a live float64/Kahan mass-balance series.

CLI::

    uv run python -m solver.run                    # demo: M0 tile + uniform rain
    uv run python -m solver.run --tiles data/tiles/demo --out data/results/demo.zarr
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

from solver.core.grid import Grid
from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State
from solver.io.zarr_writer import ZarrWriter

EPS_T = 1e-6  # time-comparison tolerance (seconds)


@dataclass
class Scenario:
    """Minimal M1 run configuration (the §7.1 TOML contract arrives in M2)."""

    name: str = "demo_basin_rain"
    dx: float = 30.0  # metres; overridden by the tile manifest when loading
    end_time: float = 3600.0  # simulated seconds
    output_every: float = 300.0
    alpha: float = 0.7  # CFL-like coefficient for the adaptive timestep
    dt_max: float = 30.0
    manning_n: float = 0.035
    rain_mm_hr: float = 50.0
    rain_duration: float = 1800.0  # rain falls for the first half-hour
    crs: str = ""
    initial_depth: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def rain_m_s(self) -> float:
        return self.rain_mm_hr / 1000.0 / 3600.0


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
) -> MassLedger:
    """Run the local-inertial solver and stream results to a Zarr store.

    Timestep is adaptive and derived from state (determinism, §8/§12) but clamped
    so a step never crosses an output time or the rainfall on/off boundary -- so
    frames land exactly on ``output_every`` and each step is either fully raining
    or fully dry (exact source accounting).
    """
    grid = Grid(ny=bed.shape[0], nx=bed.shape[1], dx=scenario.dx)
    st = State.from_bed(bed, dx=scenario.dx, depth=scenario.initial_depth, device=device)
    ledger = MassLedger.from_state(st)

    n_frames = int(round(scenario.end_time / scenario.output_every)) + 1
    attrs = {
        "scheme": "local_inertial",
        "crs": scenario.crs,
        "dx": scenario.dx,
        "units": {"depth": "m", "u": "m/s", "v": "m/s", "time": "s", "bed": "m"},
        "scenario": scenario.name,
        "rain_mm_hr": scenario.rain_mm_hr,
        "rain_duration_s": scenario.rain_duration,
        "manning_n": scenario.manning_n,
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
    # Event times a step must not cross: output cadence, rain end, end of run.
    output_times = [scenario.output_every * k for k in range(1, n_frames)]
    events = output_times + [scenario.rain_duration, scenario.end_time]

    while t < scenario.end_time - EPS_T:
        dt = compute_dt(st, alpha=scenario.alpha, dt_max=scenario.dt_max)
        dt = min(dt, _next_event_time(t, events) - t)

        raining = t < scenario.rain_duration - EPS_T
        rain = scenario.rain_m_s if raining else 0.0

        step(st, dt=dt, manning_n=scenario.manning_n, rain=rain)
        if rain > 0.0:
            ledger.add_rain_step(rain, dt, grid.n_cells)
        t += dt

        if t >= next_output - EPS_T and next_output <= scenario.end_time + EPS_T:
            rec = ledger.record(st, t)
            u, v = st.velocities_numpy()
            writer.append(t, st.depth_numpy(), u, v)
            if verbose:
                print(
                    f"  t={t:8.1f}s  h_max={float(st.h.numpy().max()):6.3f}m  "
                    f"mass_rel_err={rec.rel_error:.2e}"
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
    p = argparse.ArgumentParser(description="Run the M1 local-inertial solver (demo).")
    p.add_argument("--tiles", default="data/tiles/demo", help="M0 tiles dir (tiles.json)")
    p.add_argument("--out", default="data/results/demo.zarr", help="output Zarr store")
    p.add_argument("--device", default=None, help="warp device (cpu / cuda:0); auto if unset")
    p.add_argument("--end-time", type=float, default=3600.0, help="simulated seconds")
    p.add_argument("--output-every", type=float, default=300.0, help="write cadence (s)")
    p.add_argument("--rain-mm-hr", type=float, default=50.0)
    p.add_argument("--rain-duration", type=float, default=1800.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    device = pick_device(args.device)
    bed, manifest = load_r32_bed(args.tiles)
    scenario = Scenario(
        dx=float(manifest["dx_m"]),
        crs=manifest.get("crs", ""),
        end_time=args.end_time,
        output_every=args.output_every,
        rain_mm_hr=args.rain_mm_hr,
        rain_duration=args.rain_duration,
    )
    print(f"River Basin M1 solver | device={device} | grid={bed.shape} dx={scenario.dx:.2f}m")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run_simulation(scenario, bed, args.out, device=device)


if __name__ == "__main__":
    main()
