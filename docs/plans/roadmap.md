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
| **M5** | **Multi-physics** | Multi-rate scheduler (single simulated clock, sync points, operator splitting), exercised by reservoir operations (`[[structures]]` + release rules). Also lands the M4 deferrals: **`fixed_stage`** BC (HLLC-only), the **datum shift**, and **EA Test 1**. ~~Pre-M5 runs stay bitwise-identical~~ (retired 2026-08-17 — see the carried item below). | **done** (confirmed 2026-08-09, after M6) |
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

1. ~~**The sync-point `Δt` clamp degrades local-inertial, and no gate can see it.**~~
   **Done 2026-08-17** — the scheduler fills each span with `ceil(span/dt)` *equal*
   steps instead of full steps plus a remainder. The largest step-to-step change in
   `Δt` falls from **58 575 % to under 8 %**, and the interior ripple on a uniform
   steady reach from **2342 mm to 0.012 mm** at an 11.25 s cadence. Gated by
   `validation/test_clamp_ripple.py` — on depth **curvature**, because max-minus-min
   cannot tell an oscillation from a backwater slope. Retires the pre-M5
   bitwise-identity invariant on purpose (four replacement invariants in
   `solver/test_scheduler.py`); determinism is untouched. One consequence worth
   knowing: the shipped 900 s cadence was never clean either, only 20× less dirty.
   See `scheduler-equal-steps.md`.
2. **Point sources are uncompensated, and on a flood-driven scenario that is the whole
   residual.** Found by the scheduler pass (2026-08-17), which moved `reach_alluvial`'s
   mass error from 5.53e-08 to 4.79e-07 — 2.1× under the gate. A/B'd with only
   `scheduler.py` swapped, and **systematic, not a draw**: across two further Δt
   partitions the old code holds 5.5–6.9e-08 and the new one 4.8–6.2e-07. In absolute
   terms it is **−2.1 m³ in 4.45 million** (3.6e-10 m per cell), and `outflow_cum` is
   exactly 0.0 all run, so the residual is entirely float32 stored volume against the
   float64 inflow ledger. The likely seat is the four `[[inflow]]` cells, whose
   `Q·dt/A` add is float32 and deliberately uncompensated: `precision-sources.md` §2
   scoped point sources out because inflow was **~1.3 % of the residual** — but that
   was measured on a *rain-driven* scenario, and does not transfer to a flood-driven
   one where inflow is the only source. Fix is `sources.py`'s Kahan idiom at a second
   call site; it needs its own before/after across every inflow-bearing scenario, so
   it is its own commit. See `scheduler-equal-steps.md` §8.5.
3. **The morphological Courant diagnostic overstates the error, and cannot be filtered
   without moving a load-bearing gate.** On the M7 demo it peaks at 46 425 (one wetting-
   front cell of 1414) and still reads 19.4 over in-range cells, while halving the
   interval changes the bed by 0.9%. Making it regime-aware would silently change what
   the Courant-3.30 fixture asserts — the only evidence that the Courant and celerity
   gates are not substitutes — so it wants its own commit and its own before/after.
