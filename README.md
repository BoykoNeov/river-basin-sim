# River Basin Simulator

A **batch GPU shallow-water river-basin simulator** for large spatial domains, with
an interactive 3D viewer for setting up scenarios and exploring results.

The workflow is **configure → run → explore**:

1. **Configure** a scenario — terrain, rainfall, parameter fields (roughness,
   infiltration), structures, boundary conditions, run settings.
2. **Run** a GPU shallow-water simulation over a large domain. The run takes
   whatever wall-time it needs (seconds to overnight) — it is *not* real-time.
3. **Explore** the stored time-series in a 3D viewer — scrub the timeline, fly the
   camera, toggle depth/velocity layers, compare scenarios.

> This is a faithful **research / education sandbox**, validated against standard
> benchmarks. It is **not** an engineering- or regulatory-certification tool, and
> makes no accuracy guarantees for that purpose.

## Architecture

Three independent components connected by **files, not code coupling** — either side
can be rewritten as long as the interchange formats hold:

- **Pipeline** (Python) — conditions raw DEMs into engine-ready terrain tiles.
- **Solver** (Python + [NVIDIA Warp](https://github.com/NVIDIA/warp), GPU) — runs the
  shallow-water simulation; emits a canonical [Zarr](https://zarr.dev) store, lean
  per-frame tiles for playback, and a `status.json` progress file.
- **Viewer** ([Godot 4](https://godotengine.org)) — sets up scenarios, launches the
  solver as a subprocess, and streams the results.

The full design — locked decisions, numerics spec, component contracts, and the
milestone build order — lives in [`HANDOFF.md`](./HANDOFF.md).

## Numerical methods

- **Local-inertial** (Bates et al. 2010) — the cheap, stable, GPU-friendly MVP scheme
  and the permanent coverage tool for lowland floodplains.
- **Well-balanced Godunov finite-volume with HLLC** (later) — correct shocks and
  transcritical flow; the validatable gold standard.
- float32 GPU fields with a **float64 / Kahan mass-balance accumulator** as the
  honest "is this still physical?" gauge. Validated against dam-break (analytical),
  lake-at-rest (well-balancedness), and the UK EA 2D benchmark suite.

## Requirements

- An **NVIDIA GPU** (developed on an RTX 5090 / Blackwell, sm_120). NVIDIA-only by
  design — no cross-vendor portability.
- A recent NVIDIA driver (Warp bundles its own CUDA runtime — no separate CUDA
  toolkit install needed).
- Python **3.13** and [`uv`](https://docs.astral.sh/uv/).
- [Godot 4.x](https://godotengine.org) for the viewer.

## Quickstart

```bash
uv sync                                   # create the venv + install deps
uv run python scripts/smoke_test.py       # verify Warp sees the GPU + Zarr round-trip
```

A green smoke test confirms the toolchain (Warp initialises, JIT-compiles a kernel
for your GPU, and round-trips data through Zarr/xarray).

## Status

**M0 — Foundation, in progress.** The toolchain is proven; the DEM pipeline and a
static Godot terrain scene are next. See [`docs/plans/`](./docs/plans/) for the
roadmap (M0–M7) and current milestone plan.

## License

[MIT](./LICENSE) — free to use, fork, and modify with attribution.
