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
   ~~Remaining: the shader still lifts water as `bed + depth`, so a sub-grid channel's
   surface is drawn up to `d` high.~~ **Done 2026-08-17** — and the fix that item
   specified was not renderable, which is the durable half. `manifest["static"]` now
   ships `channel_width`/`channel_depth`, and `viewer_export.render_eta` takes the
   **exact** curve overbank (the old lift was wrong by `h_bf` there, up to 1.39 m) and
   draws the **bank** below bank full. Evaluating the whole curve, as that item asked,
   would have put the surface *under* the rendered terrain — up to **2.46 m** under it on
   **1030 of 2232** channel cells — because the trench is sub-grid and a per-cell height
   map cannot hold it; carving one is a whole-cell groove up to **14.7×** too wide and
   costs the "terrain is the exported bed" invariant. The old render drew the river as a
   **ridge** up to **2.74 m** proud of its own valley; the residual now is one-sided,
   measured per run over every frame, and declared
   (`static.channel.in_bank_offset_m` — **3.06 m** worst on the demo, ~0.3 m near bank
   full). **368 → 374 tests green**, `--rbverify` counts the decoded channel cells
   (2232/2232), and the full loop is green at 2.30e-08. See `viewer-channel-surface.md`
   and `viewer-terrain-mosaic.md` §4.

**After the roadmap.** M0–M7 are signed off and every carried item is closed, so the
table above is finished. The first thing chosen past it is **real DEM, end to end**
(`real-dem-reach.md`, planned 2026-08-17): every reach-scale run in this repo is
synthetic, and `pipeline/channels.py` — which exists to derive channel geometry from real
flow accumulation — has never been run. The survey behind that plan found a blocker worth
knowing before anything else touches channel fields: **a D8-derived network is not
4-connected, and the solver's faces are.** 48.3 % of channel cells take a diagonal step,
which leaves 40 448 real channel cells as **19 008** rook-connected fragments (largest 37
cells) where the same mask is 61 components under 8-connectivity — a chain of pools that
would fill rather than convey, and pass the mass gate while doing it. The fix is measured
and complete (61 components, zero isolated cells, holds through coarsening), but only when
the inserted corner cell is given **the through-path's width**: on its own accumulation it
is a 3.6 % aperture, a pinhole rather than a wall. See `real-dem-reach.md` §2.

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
2. ~~**Point sources are uncompensated, and on a flood-driven scenario that is the whole
   residual.**~~ **Done 2026-08-17** — each `[[inflow]]` entry now carries its own float32
   Kahan term through `sources.kahan_add` (`solver/processes/inflow.py`), owned by the
   injector rather than arming `State.h_comp`, so neither scheme's areal-source dispatch
   moves. **The pass also corrected the finding that asked for it.** Probing the target
   cells in float64 around each launch shows the uncompensated add put **+1.215 m³** more
   into the field than the ledger banked (of 4.446 M m³ requested, 5630 steps), and
   compensated that becomes **−0.000093 m³** — a 13 000× cut in the point source's own
   error. But the run's total residual only **halves, 4.79e-07 → 2.66e-07** on CUDA
   (5.45e-07 → 2.14e-07 on CPU — both backends agree on the size), so inflow
   was not "the whole residual" as carried; the rest is the flux-divergence floor
   `precision-sources.md` §5 already named. (Resist turning that into a percentage: the
   attribution is signed *positive* and the ledger's residual row is *negative*, so they
   do not subtract — the honest pair is "13 000× on the term, roughly half on the
   total".) One lesson generalizes: the drift is
   systematic only while nothing else writes `h` (489× in pure accumulation), because
   correlated low-order bits make the same rounding decision repeatedly; once continuity
   rewrites `h` every step it decorrelates into a random walk and an A/B on a *flowing*
   fixture's mass residual reads as noise (0.8× / 2.8× / 7.0×) — so the gates are on
   accumulation, not on a stepped run's mass balance. See
   `point-source-compensation.md`.
3. ~~**The morphological Courant diagnostic overstates the error, and cannot be filtered
   without moving a load-bearing gate.**~~ **Done 2026-08-17** — and the finding as
   carried was wrong twice over, which is most of what the pass produced.
   **The stated blocker was false.** Making the diagnostic regime-aware would *not*
   have changed what the Courant-3.30 fixture asserts: that fixture runs at
   `h/d50 = 187`, its minimum over transporting cells is 130, it has zero cells below
   10, and its raw, bed-weighted and in-regime peaks are numerically **identical** on
   both arms. The real blocker was always `_sediment_bowl` at `h/d50 = 0.5`, which this
   item never named.
   **And filtering was the wrong fix anyway.** Measured across four runs: a share-based
   trigger cannot quiet the demo (0.495 gross-weighted), weighting the peak by where
   the bed moved leaves it at 39 271 unchanged (the guard cell moves bed), and —
   decisively — the *rain-sheet* configuration that transports 1.9e9 m³ in a regime
   shallower than one grain reads a share of **0.012**, an order of magnitude *below*
   the well-formed demo. **Every threshold that quiets the demo also quiets that.** So
   `MORPH_COURANT_GATE` and the trigger are **unchanged**; what shipped is the
   breakdown beside the peak (`courant_moving`, `courant_in_regime`,
   `over_courant_share`, `courant_cells`, `live_cells`, all additive in `.zattrs`) and
   a warning that reads *"39 271 peak, 19.38 over the cells the law applies to, up to
   578 of 1414 over the gate"*. Verified as a pure observer: `reach_alluvial` on CUDA
   **byte-identical** before and after in every array, with `courant` unmoved to every
   digit. **358 → 368 tests green.** The headline is **39 271**, not the 46 425 this
   item carried — the scheduler and point-source passes moved it. Carried onward, in
   writing rather than vaguely: `c_b` is a **rigid-lid, fixed-discharge upper bound**,
   the celerity fixture *enforces* the slenderness that makes it valid and a real basin
   does not, so the demo stays over the gate where its transport is real. See
   `morph-courant-diagnostic.md`.
