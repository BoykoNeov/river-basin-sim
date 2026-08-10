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
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp

from solver.core.channels import arm_channels
from solver.core.datum import resolve_datum, shift_bed, unshift_bed
from solver.core.grid import Grid
from solver.core.massbalance import MASS_GATE, SEDIMENT_GATE, MassLedger, SedimentLedger
from solver.core.schemes import get_scheme
from solver.core.sediment import MORPH_COURANT_GATE, arm_sediment
from solver.core.state import State
from solver.io.coarsen import (
    block_reduce,
    check_indices,
    coarsen_scenario,
    coarsened_shape,
    crop_report,
)
from solver.io.config import Scenario, Structure, load_config
from solver.io.fields import load_field
from solver.io.mosaic import Mosaic, assemble_mosaic
from solver.io.provenance import write_provenance
from solver.io.status import StatusWriter
from solver.io.viewer_export import export_frames
from solver.io.zarr_writer import ZarrWriter
from solver.processes.inflow import InflowInjector
from solver.processes.morphology import MorphologyProcess, bed_change_bounds
from solver.processes.reservoir import apply_barriers, build_operators
from solver.scheduler import EPS_T, MultiRateScheduler

# Scenario is defined in solver.io.config (the §7.1 contract); re-exported here so
# existing callers (`from solver.run import Scenario`) keep working.
__all__ = ["Scenario", "load_config", "run_simulation", "main"]


def load_r32_bed(tiles_dir: str | Path) -> tuple[np.ndarray, dict]:
    """Load the first tile of an M0 ``tiles.json`` manifest as a bed array.

    Returns the ``(ny, nx)`` float32 bed (metres) plus the manifest dict (for dx,
    CRS, bounds). The ``.r32`` is raw little-endian row-major float32 (HANDOFF §7).

    Kept as the single-tile shorthand (and for callers that predate M6); the run
    path assembles the whole tile set through :func:`solver.io.mosaic.assemble_mosaic`.
    """
    m = assemble_mosaic(tiles_dir, select="first")
    return m.bed, m.manifest


def field_memory_mb(shape: tuple[int, int], *, sediment: bool = False) -> float:
    """Rough device memory for one run's state fields, in MB (M6 §4).

    Counts the local-inertial working set -- ``h, z, n, eta, beta`` at cell centres
    plus ``qx, qy`` on faces, i.e. ~7 arrays of ``ny*nx`` float32 -- so a reach-scale
    domain's cost is printed *before* stepping rather than discovered as a CUDA
    out-of-memory. Optional fields (momentum, channels, loss ledger) add to this;
    the number is an order-of-magnitude guide, and says so.

    ``sediment`` adds M7's morphology working set, and the two widths are summed
    **separately** rather than folded into a "count x 4 bytes": six of those arrays
    are float32 (``d50``, ``z0``, and the four face accumulators) but ``dz_cum`` and
    ``dz_unapplied`` are float64 by necessity (M7 plan §1.1), and at 768^2 that pair
    alone is ~9 MB -- the single largest contributor, and exactly what a hard-coded
    4-byte width would hide.
    """
    ny, nx = shape
    cells = ny * nx
    f32, f64 = 7, 0
    if sediment:
        f32 += 6  # d50, z0, qs_int_x/y + their compensation terms
        f64 += 2  # dz_cum, dz_unapplied
    return (4 * f32 + 8 * f64) * cells / (1024.0 * 1024.0)


def pick_device(requested: str | None) -> str:
    """Resolve the Warp device: honour a request, else CUDA if present, else CPU."""
    wp.init()
    if requested:
        return requested
    return "cuda:0" if wp.get_cuda_devices() else "cpu"


@dataclass
class _ShiftedElevations:
    """A run's absolute elevations moved into the stepping datum (M5 §1.5)."""

    bed: np.ndarray
    z_ref: float
    stage_curves: dict[str, list[tuple[float, float]]]
    structures: list[Structure]


def shift_for_datum(scenario: Scenario, bed: np.ndarray) -> _ShiftedElevations:
    """Resolve ``[grid] datum`` and put the run into shifted coordinates (M5 §1.5).

    **Every absolute elevation the scenario carries shifts here**, in this one
    function, so the bed and the config's elevations can never disagree by
    ``z_ref``: the bed, each ``fixed_stage`` water-level curve, and each structure's
    ``crest_m`` / ``target_stage_m``. Depth, velocity and all mass accounting are
    datum-independent, and the canonical store still records the true bed
    (:func:`solver.core.datum.unshift_bed`).
    """
    z_ref = resolve_datum(scenario.datum, bed)
    curves = {
        edge: [(t, level - z_ref) for t, level in curve]
        for edge, curve in scenario.stage_curves.items()
    }
    structures = [
        replace(
            s,
            crest_m=s.crest_m - z_ref,
            target_stage_m=(None if s.target_stage_m is None else s.target_stage_m - z_ref),
        )
        for s in scenario.structures
    ]
    return _ShiftedElevations(
        bed=shift_bed(bed, z_ref), z_ref=z_ref, stage_curves=curves, structures=structures
    )


def run_simulation(
    scenario: Scenario,
    bed: np.ndarray,
    out_path: str | Path,
    *,
    device: str = "cpu",
    verbose: bool = True,
    status: StatusWriter | None = None,
    domain: Mosaic | None = None,
) -> MassLedger:
    """Run the selected scheme and stream results to a Zarr store.

    The clock is the M5 :class:`~solver.scheduler.MultiRateScheduler`: the timestep
    is adaptive and derived from state (determinism, §8/§12) but clamped so a step
    never crosses a **sync point** -- an output time, a forcing breakpoint (rainfall
    on/off, hydrograph knot) or a slow-process activation. So frames land exactly on
    ``output_every``, each step is either fully raining or fully dry (exact source
    accounting), and slow processes split on an exact simulated interval.

    If ``status`` is given, a ``running`` record is written at each output frame
    (§7.4). ``status`` is a read-only progress observer -- it never touches Δt or
    the Zarr, so determinism is unaffected.
    """
    if scenario.dx is None:
        raise ValueError("scenario.dx is unresolved; fill it from the tile manifest first")
    # Scheme dispatch (plan §1.1): the run loop stays scheme-agnostic -- it calls
    # the scheme-owned compute_dt/step pair. LI is the default coverage scheme;
    # hllc_fv is the M4 fidelity option (raises NotImplementedError until wired up).
    scheme = get_scheme(scenario.scheme)
    # Resolution choice (M6, solver.io.coarsen): fields are authored at tile
    # resolution, so they are loaded on the *source* grid and aggregated once,
    # before any water moves -- one uniform grid per run, no resolution interface.
    # coarsen == 1 is the identity on every path, so pre-M6 runs are unchanged.
    k = int(scenario.coarsen)
    src_grid = Grid(ny=bed.shape[0], nx=bed.shape[1], dx=scenario.dx)
    if k > 1:
        note = crop_report(bed.shape, k)
        if note and verbose:
            print(f"  {note}")
        bed = block_reduce(bed, k, "mean")  # volume-preserving floodplain storage
        scenario = coarsen_scenario(scenario, k)
    grid = Grid(ny=bed.shape[0], nx=bed.shape[1], dx=scenario.dx * k)
    check_indices(scenario, grid.shape)

    def _field(path, scalar, name, how="mean"):
        """Load a parameter field on the source grid, then aggregate it (M6)."""
        f = load_field(path, src_grid, scalar=scalar, name=name, nonneg=True)
        return block_reduce(f, k, how) if k > 1 else f

    manning = _field(scenario.manning_field, scenario.manning_n, "manning_n")
    # Vertical datum (M5, solver.core.datum): step in shifted coordinates so float32
    # eta = h + z keeps its precision at a high datum; the store gets the true bed
    # back. Every absolute elevation the scenario carries shifts with it, in one
    # place (shift_for_datum), so they cannot drift apart.
    elev = shift_for_datum(scenario, bed)
    z_ref = elev.z_ref
    # Structures are bed geometry (M5 plan §1.2): raise the bed to each crest before
    # stepping, so impoundment and overtopping are ordinary solver physics. The
    # modified bed is what the store records, so the viewer shows the structure.
    bed_sim = apply_barriers(elev.bed, elev.structures)
    st = State.from_bed(
        bed_sim, dx=grid.dx, depth=scenario.initial_depth, manning=manning, device=device
    )

    # --- M6 sub-grid channels ---------------------------------------------------
    # A channel narrower than a cell, as per-cell geometry: what keeps a coarse
    # reach-scale run from erasing the river (plan §1.2). Armed only when the
    # scenario carries a width; unarmed, the LI kernels are the M1 ones untouched.
    channels = None
    if scenario.has_channels:
        # Width and depth aggregate by **max**, not mean: a river passes *through*
        # a block, and averaging its width with the dry cells beside it would thin
        # away exactly what sub-grid channels exist to keep (solver.io.coarsen).
        chan_w = _field(
            scenario.channel_width_field, scenario.channel_width_m, "channel width", "max"
        )
        chan_d = _field(
            scenario.channel_depth_field, scenario.channel_depth_m, "channel depth", "max"
        )
        chan_n = None
        if scenario.channel_manning_field is not None or scenario.channel_manning is not None:
            chan_n = _field(
                scenario.channel_manning_field,
                (scenario.channel_manning or 0.0),
                "channel manning",
            )
        channels = arm_channels(st, chan_w, chan_d, chan_n)
        if verbose:
            print(f"  channels      : {channels.summary()}")

    # --- M7 morphology ----------------------------------------------------------
    # Armed after the bed is final (barriers applied, State built): `arm_sediment`
    # captures `z0` here and the bed is *rebuilt* from it at every activation, so
    # arming earlier would delete every dam at the first one.
    morphology = None
    if scenario.has_sediment:
        # d50 aggregates by mean (a grain size is a per-cell property, not a volume);
        # so does an alluvium thickness, where the mean *is* volume-preserving --
        # unlike M6's channel width, which needs max (solver.io.coarsen).
        d50 = _field(scenario.sediment_d50_field, scenario.sediment_d50_m, "d50")
        arm_sediment(st, d50, scenario.sediment_porosity)
        thickness = None
        if scenario.has_alluvium_floor:
            # Field wins, as everywhere in [parameters] -- and it must, because a
            # field-backed floor leaves the scalar at its unused 0.0 fallback, which
            # read on its own would say "bedrock everywhere" and freeze the bed with
            # no error anywhere (solver.io.config.Scenario).
            thickness = _field(
                scenario.alluvium_thickness_field,
                scenario.alluvium_thickness_m or 0.0,
                "alluvium thickness",
            )
        # A dam is engineered, not alluvial: freeze every structure cell, or the flow
        # can scour one out from under its own release rule (M7 plan §1.5). What the
        # freeze refuses is banked for the sediment ledger, never discarded.
        frozen = [cell for s in elev.structures for cell in s.cells]
        dz_lo, dz_hi = bed_change_bounds(
            grid.shape, alluvium_thickness=thickness, frozen_cells=frozen
        )
        morphology = MorphologyProcess(st, scenario.sediment_interval_s, dz_lo=dz_lo, dz_hi=dz_hi)
        if verbose:
            print(f"  morphology    : {morphology.summary()}")

    # --- M3 sources/sinks -------------------------------------------------------
    # Infiltration (constant-rate sink, mm/hr -> m/s); armed only when nonzero.
    infil = _field(scenario.infiltration_field, scenario.infiltration_mm_hr, "infiltration")
    infil_m_s = infil / 1000.0 / 3600.0
    if scenario.infiltration_field is not None or float(infil_m_s.max()) > 0.0:
        st.set_infiltration(infil_m_s)
    # Spatial rainfall field (mm/hr -> m/s); uniform rain keeps the scalar path.
    rain_is_field = scenario.rain_type == "field"
    rain_field_sum_m_s = 0.0
    if rain_is_field:
        rain_mm_hr_field = _field(scenario.rain_field, 0.0, "rainfall")
        rain_field = rain_mm_hr_field / 1000.0 / 3600.0
        st.set_rain_field(rain_field)
        rain_field_sum_m_s = float(rain_field.astype(np.float64).sum())
    # Areal sources get compensated (Kahan) accumulation -- float32 `h += rate*dt`,
    # once per cell per step over a reach-scale grid, is what puts a rain-on-grid run
    # at the mass gate for arithmetic reasons (solver.core.sources). Armed only when
    # rain actually falls, so every run without an areal source keeps the original
    # kernels and is bitwise unchanged; point sources are deliberately out of scope.
    if rain_is_field or (scenario.rain_m_s > 0.0 and scenario.rain_duration > 0.0):
        st.arm_source_compensation()
    # Inflow hydrographs (prescribed discharge point sources).
    injector = InflowInjector(scenario.inflows, grid, device) if scenario.inflows else None
    # Per-edge boundaries: open (free-outflow) and, from M5, fixed_stage water-level
    # curves (already in the stepping datum). No-op when every edge is closed.
    st.set_open_boundaries(scenario.boundaries, elev.stage_curves)
    # Reservoir release rules -- the slow processes the multi-rate scheduler exists
    # for (HANDOFF §8). Built after the State so they can arm the loss accumulator.
    reservoirs = build_operators(elev.structures, st)

    ledger = MassLedger.from_state(st)
    # The second conserved substance gets its own ledger, and only when there is a
    # bed to conserve: solid volume is closed to the domain (bedload cannot cross a
    # boundary face), so what it balances is that every metre gained somewhere came
    # from somewhere else, plus whatever the bounds refused and banked.
    sed_ledger = SedimentLedger.from_state(st) if morphology is not None else None

    n_frames = int(round(scenario.end_time / scenario.output_every)) + 1
    attrs = {
        "scheme": scenario.scheme,
        "crs": scenario.crs,
        "dx": grid.dx,
        "coarsen": k,
        "units": {"depth": "m", "u": "m/s", "v": "m/s", "time": "s", "bed": "m"},
        "scenario": scenario.name,
        "rain_type": scenario.rain_type,
        "rain_mm_hr": scenario.rain_mm_hr,
        "rain_duration_s": scenario.rain_duration,
        "manning_n": scenario.manning_n,
        "infiltration_mm_hr": scenario.infiltration_mm_hr,
        "end_time_s": scenario.end_time,
        "output_every_s": scenario.output_every,
        # Elevation offset the run stepped in (0 unless [grid] datum was set); the
        # stored bed is *un*shifted, so this is provenance, not a decoding key.
        "datum_shift_m": z_ref,
        # Static provenance (§2): source hash + resolved scenario -> reproducible.
        "provenance": write_provenance(scenario, out_path),
    }
    if channels is not None:
        attrs["channels"] = channels.as_attrs()
    if morphology is not None:
        attrs["sediment"] = {"law": scenario.sediment_law, **morphology.as_attrs()}
    if domain is not None:
        # How the domain was assembled from the tile set (M6): origin, tile count and
        # any uncovered cells, so a stored result says which patch of the world it is.
        attrs["domain"] = domain.as_attrs()
    writer = ZarrWriter(out_path, grid, n_frames, attrs, bed_change=morphology is not None)
    writer.write_bed(unshift_bed(bed_sim, z_ref))  # §7.2 stores true elevations
    if channels is not None:
        # The bed alone no longer describes what the run stepped on: store the
        # channel geometry beside it so a result is self-describing (§7.2).
        writer.write_static("channel_width", channels.w.numpy())
        writer.write_static("channel_depth", channels.d.numpy())

    def bed_change_frame() -> np.ndarray | None:
        """This frame's cumulative bed change, or ``None`` when nothing morphs.

        Deliberately **not** put through :func:`unshift_bed`: a datum shift moves the
        origin of an elevation and a *difference* of elevations has no origin to move
        (M7 plan §1.7). That is what makes `bed` + `bed_change` addable without the
        reader knowing which datum the run stepped in.
        """
        return None if morphology is None else st.sediment.bed_change_numpy()

    # Frame at t = 0 (baseline).
    u0, v0 = st.velocities_numpy()
    writer.append(0.0, st.depth_numpy(), u0, v0, bed_change_frame())

    # Forcing breakpoints a step must not cross (the scheduler adds the output
    # cadence, end_time and any slow-process activations): rain on/off and each
    # inflow-hydrograph knot, so the sampled discharge stays faithful.
    inflow_events = injector.breakpoints() if injector else []
    sched = MultiRateScheduler(
        end_time=scenario.end_time,
        output_every=scenario.output_every,
        events=[scenario.rain_duration, *inflow_events, *scenario.stage_events],
        # Order is the order they advance in on a tick both are due (M5's reservoirs
        # first): a release rule reads a stage `z + h` and should read it off the bed
        # the interval's water actually flowed over, not one morphology is about to
        # move underneath it.
        processes=[
            *(r.as_slow_process() for r in reservoirs),
            *([morphology.as_slow_process()] if morphology is not None else []),
        ],
    )

    for tick in sched.ticks(
        lambda: scheme.compute_dt(st, alpha=scenario.alpha, dt_max=scenario.dt_max)
    ):
        t, dt = tick.t0, tick.dt

        # Inject inflow hydrographs for this step (midpoint discharge -> volume).
        if injector is not None:
            ledger.add_inflow(injector.apply(st, t, dt))

        raining = t < scenario.rain_duration - EPS_T

        if rain_is_field:
            scheme.step(st, dt=dt, rain_scale=(1.0 if raining else 0.0), t=t)
            if raining:
                ledger.add_inflow(rain_field_sum_m_s * dt * grid.cell_area)
        else:
            rain = scenario.rain_m_s if raining else 0.0
            scheme.step(st, dt=dt, rain=rain, t=t)
            if rain > 0.0:
                ledger.add_rain_step(rain, dt, grid.n_cells)

        # Slow processes (operator splitting, HANDOFF §8) advance *after* the fast
        # step has landed on the sync point, by the exact elapsed simulated time.
        for proc, elapsed in tick.due:
            proc.advance(tick.t1, elapsed)

        if tick.is_output:
            rec = ledger.record(st, tick.t1)
            u, v = st.velocities_numpy()
            writer.append(tick.t1, st.depth_numpy(), u, v, bed_change_frame())
            h_max = float(st.h.numpy().max())
            line = f"  t={tick.t1:8.1f}s  h_max={h_max:6.3f}m  mass_rel_err={rec.rel_error:.2e}"
            if sed_ledger is not None:
                sed_rec = sed_ledger.record(st, tick.t1)
                # Gross displaced volume, not the net: the net is identically zero in
                # a domain closed to bedload, so it would say nothing about whether
                # any sediment moved (solver.core.massbalance.SedimentLedger).
                line += f"  bed_moved={sed_rec.gross_volume:9.1f}m3"
            if verbose:
                print(line)
            if status is not None:
                status.write(
                    "running",
                    sim_time=tick.t1,
                    message=(
                        f"t={tick.t1:.0f}s  h_max={h_max:.3f}m  mass_rel_err={rec.rel_error:.1e}"
                    ),
                )

    final_attrs = dict(ledger.as_attrs())
    if morphology is not None:
        # The bed-change history beside the mass series, for the same reason: a
        # stored run says what its slow clock did -- the activation-by-activation
        # record of what produced the `bed_change` field written at each frame.
        final_attrs["morphology"] = morphology.series
        final_attrs.update(sed_ledger.as_attrs())
    if reservoirs:
        # The release history is the evidence the slow clock actually ran; it lives
        # with the mass series so a stored run is self-describing (§7.2).
        final_attrs["reservoir_releases"] = {r.structure.name: r.series for r in reservoirs}
    writer.finalize(final_attrs)
    if verbose:
        print(f"done: {out_path}")
        print(f"  frames        : {len(ledger.series)}")
        print(f"  mass max rel  : {ledger.max_rel_error:.2e}  (gate {MASS_GATE:.0e})")
        if sed_ledger is not None:
            last = sed_ledger.series[-1]
            print(
                f"  sediment rel  : {sed_ledger.max_rel_error:.2e}  (gate {SEDIMENT_GATE:.0e}), "
                f"{last.gross_volume:.1f} m3 moved, {last.banked_volume:+.1f} m3 banked"
            )
            print(
                f"  bed courant   : {morphology.peak_courant:.2f} peak  "
                "(> 1 is a splitting artefact)"
            )
    if ledger.max_rel_error >= MASS_GATE:
        print(f"  WARNING: mass-balance gate exceeded ({ledger.max_rel_error:.2e})")
    if sed_ledger is not None and sed_ledger.max_rel_error >= SEDIMENT_GATE:
        print(f"  WARNING: sediment-balance gate exceeded ({sed_ledger.max_rel_error:.2e})")
    if morphology is not None and morphology.peak_courant >= MORPH_COURANT_GATE:
        # Warned, not raised -- the mass gates set the precedent, and a deliberately
        # coarse exploratory run is a legitimate thing to do. It is *printed
        # unconditionally* (unlike the `bed courant` line above, which is verbose-only)
        # because this is the one failure mode with no other symptom: the run finishes,
        # the mass balance is clean, and the bed is wrong (M7 build step 8).
        # The remedy deliberately does NOT read "shorten interval_s", which is what it
        # said until build step 9. Shortening it is a trade, not a fix: every activation
        # is a scheduler sync point, and clamping the fast step onto more of them
        # degrades local-inertial into a short-wavelength depth ripple (0.165 mm at a
        # 900 s cadence, 74 mm at 22.5 s -- M7 plan §4) that no gate can see, because
        # mass is conserved and only the water's *position* is wrong. Morphology then
        # rectifies that ripple into a permanent bed signature instead of averaging it
        # out. So the warning names both bounds and lets the reader pick.
        print(
            f"  WARNING: morphological Courant {morphology.peak_courant:.2f} exceeds "
            f"{MORPH_COURANT_GATE:.0f} -- the bed wave crosses more than a cell per "
            f"activation, so the bed change is a splitting artefact. [sediment] "
            f"interval_s is {scenario.sediment_interval_s:g} s; note that shortening it "
            f"trades this artefact for the sync-clamp one (M7 plan sec. 4), so check the "
            f"bed against a longer interval rather than assuming shorter is safer. "
            f"The peak is a field maximum, so a single cell at the wet/dry guard can "
            f"set it -- check where the bed change actually is before acting."
        )
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


def _resolve_scenario(args: argparse.Namespace) -> tuple[Scenario, Mosaic]:
    """Build the run Scenario (from --config or the demo flags) and its domain.

    The domain is the assembled **tile mosaic** (M6, :mod:`solver.io.mosaic`) --
    ``[grid] tiles`` selects the whole set or just tile 0, ``[grid] window`` clips a
    reach out of it. ``dx``/``crs`` unset by the config inherit from the tile
    manifest (§7.1).
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
    mosaic = assemble_mosaic(scenario.tiles_dir, select=scenario.tiles, window=scenario.window)
    if scenario.dx is None:
        scenario.dx = mosaic.dx
    if not scenario.crs:
        scenario.crs = mosaic.crs
    return scenario, mosaic


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
        scenario, mosaic = _resolve_scenario(args)
        bed = mosaic.bed
        status.end_time = scenario.end_time
        print(
            f"River Basin solver | device={device} | scenario={scenario.name} | "
            f"scheme={scenario.scheme}"
        )
        print(f"  domain        : {mosaic.summary()}")
        run_shape = coarsened_shape(bed.shape, scenario.coarsen)
        if scenario.coarsen > 1:
            print(
                f"  resolution    : coarsen={scenario.coarsen} -> {run_shape[0]}x{run_shape[1]} "
                f"cells @ dx={scenario.dx * scenario.coarsen:.2f} m"
            )
        mem = field_memory_mb(run_shape, sediment=scenario.has_sediment)
        kind = "state fields + morphology" if scenario.has_sediment else "float32 state fields"
        print(f"  field memory  : ~{mem:.1f} MB ({kind})")
        if mosaic.gap_cells:
            print(f"  WARNING: {mosaic.gap_cells} cells are not covered by any tile (filled flat)")
        status.write("starting", message=f"{scenario.name}: {mosaic.summary()}")

        ledger = run_simulation(
            scenario, bed, out_path, device=device, status=status, domain=mosaic
        )
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
