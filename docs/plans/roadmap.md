# Roadmap (M0 – M7)

Milestone build order from HANDOFF §9. Each milestone is independently demoable;
stop at each demo and confirm before starting the next (§13). The mass-balance
diagnostic and the validation harness gate every step.

| # | Milestone | Demo / gate | Status |
|---|---|---|---|
| **M0** | **Foundation** | Pipeline + viewer + handoff proven: sample DEM conditioned + tiled, static terrain loads in Godot. *No dynamics.* | **done** |
| **M1** | **Water moves** | Local-inertial solver (Warp), uniform rainfall, closed BCs, Zarr out, live mass balance. **Validate: dam-break.** | **done** |
| **M2** | **The loop closes** | §7 contracts: config-in/results-out, subprocess + status.json, per-frame tiles; Godot timeline + depth colormap + water surface. | **done** |
| **M3** | **Real scenarios** | Scenario system + command log + spatially-varying parameter fields; inflow hydrographs + open boundaries. **Validate: channel normal depth.** | **acceptance met; confirm before M4** |
| M4 | Fidelity step | Well-balanced HLLC FV behind the same kernel interface. **Validate: lake-at-rest + UK EA 2D suite.** | not started |
| M5 | Multi-physics | Multi-rate scheduler, exercised by reservoir operations. | not started |
| M6 | Reach | Multi-resolution / tiling-at-scale + sub-grid channels, optional 1D river network. *Highest-risk subsystem (§12).* | not started |
| M7 | Morphology | Sediment transport (Exner + transport capacity) on the slow clock. | not started |

Detailed per-milestone plans live alongside this file as `M<n>-*.md`.
