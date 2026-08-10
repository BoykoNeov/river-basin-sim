# Roadmap (M0 – M7)

Milestone build order from HANDOFF §9. Each milestone is independently demoable;
stop at each demo and confirm before starting the next (§13). The mass-balance
diagnostic and the validation harness gate every step.

| # | Milestone | Demo / gate | Status |
|---|---|---|---|
| **M0** | **Foundation** | Pipeline + viewer + handoff proven: sample DEM conditioned + tiled, static terrain loads in Godot. *No dynamics.* | **done** |
| **M1** | **Water moves** | Local-inertial solver (Warp), uniform rainfall, closed BCs, Zarr out, live mass balance. **Validate: dam-break.** | **done** |
| **M2** | **The loop closes** | §7 contracts: config-in/results-out, subprocess + status.json, per-frame tiles; Godot timeline + depth colormap + water surface. | **done** |
| **M3** | **Real scenarios** | Scenario system + command log + spatially-varying parameter fields; inflow hydrographs + open boundaries. **Validate: channel normal depth.** | **done** |
| **M4** | **Fidelity step** | Well-balanced HLLC FV behind the same kernel interface. **Validate: lake-at-rest + UK EA 2D suite.** *Also: harden the mass-gate denominator against drain-to-empty collapse (`massbalance.py` causal peak-volume floor + a drain test) before running the suite.* | **done** |
| **M5** | **Multi-physics** | Multi-rate scheduler (single simulated clock, sync points, operator splitting), exercised by reservoir operations (`[[structures]]` + release rules). Also lands the M4 deferrals: **`fixed_stage`** BC (HLLC-only), the **datum shift**, and **EA Test 1**. Pre-M5 runs stay bitwise-identical. | **done** (confirmed 2026-08-09, after M6) |
| **M6** | **Reach** | Tiling-at-scale (the domain is the tile mosaic) + resolution choice with conservative coarsening + **sub-grid channels**, validated by a fine-vs-coarse equivalence gate. *Nested two-way multi-resolution and the 1D network stay unbuilt — §12's interface conservation is **avoided**, not solved.* | **done** |
| **M7** | **Morphology** | Sediment transport (Exner + transport capacity) on the slow clock: MPM at capacity, Exner in float64, LI-only, sub-grid channels carried. **Validate: bed-wave celerity + threshold pair + sediment mass conservation.** | **done** (signed off on GPU + CPU, 2026-08-10) |

Detailed per-milestone plans live alongside this file as `M<n>-*.md`.

**Carried debts before M7.** Two things M6 measured and deliberately did not fix.

1. ~~**Precision pass** on distributed-source accumulation.~~ **Done 2026-08-09** —
   per-cell Kahan compensation on areal sources (`solver/core/sources.py`) takes the
   failing `reach_basin` @ `coarsen = 4` case from **3.77e-6 to 1.28e-7**. Runs without
   an areal source stay bitwise unchanged; point sources are deliberately out of scope.
   **M7's sediment must go through `sources.py` rather than a bare `+=`.** See
   `precision-sources.md`.
2. ~~**Viewer terrain path** — loads tile 0 only.~~ **Done 2026-08-09** — the run's own
   bed now ships with the frames (`manifest["static"]`) and *is* the terrain, so extent,
   origin and cell size agree by construction; `--rbverify` gates that registration by
   sampling the imported surface against the exported bed.
   Remaining: the shader still lifts water as `bed + depth`, so a sub-grid channel's
   surface is drawn up to `d` high. See `viewer-terrain-mosaic.md`.

**Carried out of M7.** Two, both measured and both deliberately unshipped.

1. **The sync-point `Δt` clamp degrades local-inertial, and no gate can see it.** Not a
   morphology bug — it reproduces with sediment never armed, and it predates M5, since
   M1–M4 clamped inline. A uniform steady reach ripples 0.010 mm unclamped, 14 mm at a
   45 s cadence and **2342 mm at 11.25 s**, while mass balance reads 1e-8 throughout,
   because mass *is* conserved and only the water's position is wrong. A measured fix
   (fill each span with `ceil(span/dt)` equal steps: 14.2 → 0.009 mm) is recorded and
   unshipped, because it moves every run's `Δt` sequence and the pre-M5 bitwise-identity
   invariant with it — its own commit, like the precision pass. **This is the next
   piece of work.** See `M7-morphology.md` §4.
2. **The morphological Courant diagnostic overstates the error, and cannot be filtered
   without moving a load-bearing gate.** On the M7 demo it peaks at 46 425 (one wetting-
   front cell of 1414) and still reads 19.4 over in-range cells, while halving the
   interval changes the bed by 0.9%. Making it regime-aware would silently change what
   the Courant-3.30 fixture asserts — the only evidence that the Courant and celerity
   gates are not substitutes — so it wants its own commit and its own before/after.
