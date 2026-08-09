# M6 — Reach (tiling-at-scale, resolution choice, sub-grid channels)

**Goal:** buy **area within wall-time and memory** without losing the mass gate.
HANDOFF §2 names the scale path — *multi-resolution + sub-grid channels + 1D river
network, added later behind a stable cell interface* — and §12 flags it as the
**highest-risk subsystem**: *mass must be conserved exactly across resolution
boundaries and 1D↔2D exchange cells; build it incrementally with a dedicated
conservation test before trusting any result.*

Depends on: M5 (multi-rate scheduler, structures, `fixed_stage`, datum shift —
acceptance met, confirmed before this milestone). Gate before M7 (morphology).

---

## 0. Scope — what M6 is and is *not*

The milestone line reads *"Multi-resolution / tiling-at-scale + sub-grid channels,
optional 1D river network"*. M6 builds the three pieces that compose into a
working reach story, and is explicit about the one it does not:

**In:**

1. **Tiling-at-scale (domain side).** The run domain is the **whole tile mosaic**,
   not `tiles[0]`. `solver.io.mosaic` assembles a `tiles.json` tile set into one
   bed array with a recorded gap policy; `[grid] tiles` / `[grid] window` select
   all, one, or a sub-window of it. Parameter fields register against the mosaic.
2. **Resolution choice with conservative coarsening.** `[grid] coarsen = k` runs a
   `k`× coarser grid: **volume-preserving block-mean** bed and parameter fields,
   `dx' = k·dx`. One resolution per run — chosen up front, aggregated once, before
   any water moves.
3. **Sub-grid channels.** The capability that makes (2) honest: a channel narrower
   than a cell, carried as **per-cell geometry** (`width`, `depth`, `manning`) with
   a **storage curve** and a **two-component face conveyance** (channel +
   floodplain) in the local-inertial scheme. This is what stops a 100 m cell from
   erasing a 20 m river.
4. **Viewer stream at reach scale.** §7.3's `tile_grid` becomes real: per-frame
   export splits into tiles instead of writing one frame-sized `.raw`.

**Explicitly deferred (and *why*, so the next milestone inherits a decision, not a
gap):**

- **Nested two-way multi-resolution grids (local refinement patches).** This is the
  §12 risk in its literal form: a fine patch inside a coarse grid needs
  conservative restriction/prolongation at every interface face, plus fine-grid
  sub-cycling in time. M6 buys reach a different way — *choose* the resolution and
  recover the lost conveyance sub-grid — so **there is no resolution interface for
  mass to leak across**. That is a smaller claim than "we solved nested
  conservation", and it is the one we can validate. If a future milestone wants
  patches, the conservation test comes first (§12), and this plan is not evidence
  that it already works.
- **1D river network coupled to 2D** (`optional` even in the roadmap line). A 1D
  network adds exactly the exchange-cell conservation problem above, for capability
  that sub-grid channels largely cover at reach scale. Deferred with its gate
  named: a dedicated 1D↔2D exchange conservation test, before any result.
- **Sub-grid channels in the HLLC scheme.** Symmetric to M5's `fixed_stage`
  being HLLC-only: the sub-grid channel model is a *conveyance and storage*
  parameterization that fits LI's face-flux structure. HLLC's Riemann solver acts
  on a cell-average conservative state over a reconstructed bed; a sub-grid channel
  inside the cell has no honest expression there (a second flow path inside one
  cell average). `scheme = "hllc_fv"` + `[channels]` is a **loud config error**,
  not a silent no-op. LI is the permanent coverage scheme (§2) and reach *is*
  coverage — this is the right pairing, not a shortcut.
- **Out-of-core / streaming domains larger than GPU memory.** M6 reports the memory
  a run needs and fails honestly if it cannot fit; chunk-streamed stepping is a
  separate subsystem.
- **Temporal rainfall**, the `inflow` boundary *type* — unchanged, still gated.

---

## 1. Design decisions

### 1.1 Reach is bought by choosing resolution, not by nesting grids
Two ways to fit a big basin in memory: refine locally (nested grids, exact
interface conservation required) or coarsen globally and put the lost physics back
where it matters. M6 takes the second because it is the one whose failure mode is
*measurable* rather than structural:

- A coarser cell loses **conveyance** (the river disappears into a wide flat cell)
  and **storage** (the channel volume disappears). Sub-grid channels restore both,
  parametrically, inside the existing single-resolution kernel.
- Continuity stays a pure flux divergence over one uniform grid, so **mass
  conservation is exact to float round-off by construction** — the same property
  M1 has. No restriction/prolongation operator, no interface flux to match, no new
  way to leak.
- The honest gate is then not "is the interface conservative" (there is no
  interface) but **"does the coarse+sub-grid run reproduce the fine run"** — an
  end-to-end fidelity claim we can measure and report (§3).

### 1.2 The state variable stays cell-mean depth; the channel lives in the storage curve
The tempting move — make `h` the depth *in the channel* — breaks continuity, the
limiter, the ledger and every existing test. Instead `h` keeps its meaning
everywhere in the solver: **water volume per unit plan area**, `V = h·dx²`. The
channel enters only through the map `h → η` (water-surface elevation), which is
what the momentum kernels actually read:

```
z      floodplain bed (the DEM value, unchanged)
w, d   channel width (m) and bank-full depth below the floodplain bed (m)
z_ch = z - d                       channel bed
h_bf = w·d/dx                      cell-mean depth at bank full

h <= h_bf :  eta = z - d + h·dx/w     (all water in the channel; a narrow channel
                                       fills fast -- dx/w is the amplification)
h >  h_bf :  eta = z + (h - h_bf)     (bank full, spilling over the whole cell)
```

Continuous at `h = h_bf` (both give `η = z`), strictly monotone, exactly
invertible, and **`w = 0` collapses to `η = z + h`** — the pre-M6 relation, bit for
bit. Volume is never touched by the curve, so **the mass ledger cannot tell whether
a channel exists**; it is a diagnostic map, not an accounting one.

### 1.3 A face carries two flows, not one
The face update splits into a channel component and a floodplain component, each
its own Bates local-inertial update, summed into the flux continuity already reads
(`solver/core/grid.py`: `qx` is discharge **per unit width**, m²/s):

```
channel     (only where both cells have w > 0 -- a channel conveys only if it is
             continuous across the face; a channel that dead-ends spills, which is
             what the floodplain component then does)
  w_f  = min(w_L, w_R)                        narrowest section controls
  h_ch = min( max(eta_L,eta_R) - max(z_ch_L,z_ch_R),  bank-full cap )
  R    = w_f·h_ch / (w_f + 2·h_ch)            hydraulic radius (A/P), NOT h
  q_ch = ( q_ch - g·h_ch·dt·d(eta)/dx ) / ( 1 + g·dt·n_ch²·|q_ch| / (h_ch·R^{4/3}) )

floodplain  (exactly the M1 update, on the floodplain bed z)
  h_fp = max(eta_L,eta_R) - max(z_L,z_R)
  q_fp = ( q_fp - g·h_fp·dt·d(eta)/dx ) / ( 1 + g·dt·n²·|q_fp| / h_fp^{7/3} )

total       q = q_fp·(1 - w_f/dx) + q_ch·(w_f/dx)        [m²/s, cell-width mean]
```

- With `R → h` (a wide channel) the channel update **is** the M1 update — the
  hydraulic radius is the only place the two forms differ, and it is exactly what
  a sub-grid channel needs (a 20 m × 4 m channel has `R` ≈ 0.8·h, ~20% on
  conveyance; dropping it would be a silent bias in the direction of *more* flow).
- With `w_f = 0` the total is `q_fp·1.0 + 0` — **exactly** the M1 flux, no
  `(q·dx)/dx` round-trip.
- Both components persist as their own face arrays (`qx_ch`, `qx_fp`, …), armed
  only when channels exist. Momentum has memory; apportioning one stored total
  between two conveyances each step would fabricate that memory.

**The limiter stays the guard it is.** `compute_outflow_beta` reads `h` and the
*total* flux, so it is unchanged and still exact. The donor β is then applied to
**both components and the total** — scaling both by β scales the total by β, so
the mass-conservative donor-cell property (one face scaled once, by its donor)
survives intact.

**CFL sees the channel, not the cell mean.** `dt = α·dx/√(g·h_max)` with `h_max`
the cell-mean depth would be wrong by `dx/w`: a 0.1 m cell-mean over a 20 m channel
in a 200 m cell is a **1 m** column moving at `√(g·1)`. `compute_dt` reduces over
the true water column `η − z_ch` instead. Without channels the two are identical,
so pre-M6 timesteps are unchanged.

### 1.4 Channel geometry is data, produced offline like every other field
Two ways in, both `.r32` fields aligned to the run grid (the M3 contract):

- `[channels] width = <scalar|path>`, `depth = ...`, `manning = ...` — the config
  side, identical in shape to `[parameters]`.
- `pipeline/channels.py` — **downstream hydraulic geometry** from the M0 flow
  accumulation: `w = a·A^b`, `d = c·A^e` with `A` the upstream drainage area
  (km²), applied only where `A ≥ threshold` (elsewhere `w = 0`, i.e. no channel).
  Coefficients are config-visible and *labelled as regional-calibration inputs*,
  not universal constants — this is the honest status of hydraulic geometry.

Coarsening (§1.1) derives channel geometry **at the target resolution from the
fine-resolution accumulation** (`A_max` within the block), not by averaging a fine
width field: drainage area is the physical driver and it does not average.

### 1.5 The mosaic is assembled, and its gaps are declared
`tiles.json` tiles carry `(row, col, width, height)` in source-raster coordinates,
and an edge tile may be clipped. `assemble_mosaic` places every tile in the
bounding box of the tile set and fills any uncovered cell with a **declared fill
value** (the mosaic minimum, matching M0's per-tile nodata policy), returning the
gap count. A gap is reported, never silent: a hole in the terrain is a hole in the
hydrology. Single-tile manifests — every in-tree scenario — assemble to exactly
the array `load_r32_bed` returned before, so existing runs are unchanged.

---

## 2. Build order (each step keeps `ruff` + `pytest` green; commit each)

1. **Plan doc** (this file).
2. **Mosaic assembly** (`solver/io/mosaic.py`) + `[grid] tiles` / `[grid] window`
   + run wiring. Tests: 2×2 mosaic reassembles the source array; a run over a
   mosaic is **bitwise-identical** to the same run on the equivalent single array;
   gaps counted; window selection.
3. **Sub-grid channels** — `solver/core/channels.py` (geometry + storage curve),
   LI kernels for the two-component face update, channel-aware `compute_dt`,
   `[channels]` config + `.r32` fields, HLLC scope gate. Tests: storage-curve
   algebra; armed-but-zero-width is bitwise plain LI; **channel normal depth**
   against Manning; overbank spill conserves mass.
4. **Coarsening** (`[grid] coarsen`) — block-mean bed/fields at load time, `dx`
   scaled, channel geometry derived at target resolution. Tests: block-mean is
   volume-preserving; `coarsen = 1` is a no-op; **fine-vs-coarse equivalence** (the
   reach claim) as a validation-harness gate.
5. **Viewer frame tiling** — §7.3 `tile_grid` for real; 1×1 output stays
   byte-identical for small domains; the Godot read path handles both.
6. **Demo + docs** — a reach-scale demo scenario, `scenarios/README.md`,
   `CLAUDE.md` status, roadmap, this plan's acceptance section.

---

## 3. Validation plan (the credibility gates)

| Check | Type | Gate |
|---|---|---|
| **Mosaic reassembly** | unit | a tiled array reassembles to the source exactly; gaps counted, not hidden |
| **Mosaic run equivalence** | regression | a run on a 2×2 mosaic is **bitwise-identical** to the same run on the equivalent single bed |
| **Storage curve** | unit | continuous and monotone at bank full; `η(h)` invertible to float round-off; `w = 0` reproduces `z + h` bitwise |
| **Armed-zero-channel identity** | regression | channels armed with `w ≡ 0` reproduce plain LI bitwise (the seam cannot perturb the coverage scheme) |
| **Channel normal depth** | analytical | steady `Q` in a sub-grid channel converges to Manning normal depth (`Q = A·R^{2/3}·S^{1/2}/n`) within a stated tolerance |
| **Overbank spill** | conservation | a channel filled past bank full spills onto the floodplain; `rel_error < 1e-6` |
| **Fine-vs-coarse equivalence** | inter-resolution | the same reach at fine `dx` (channel resolved) and coarse `dx` (channel sub-grid) agree on steady discharge and normal depth within a stated tolerance — **the reach claim, measured** |
| **Coarsening is volume-preserving** | unit | block-mean bed + block-mean depth conserve total volume exactly |
| **Frame tile round-trip** | unit | exported tiles reassemble to the frame exactly; a 1×1 export is byte-identical to M2's |
| **Global mass balance** | always-on | `rel_error < 1e-6` on every new gate |

**Honesty note.** The fine-vs-coarse gate is an *equivalence* claim about our own
two runs, not a benchmark validation. Sub-grid channel models are calibrated
tools; the hydraulic-geometry coefficients in `pipeline/channels.py` are regional
inputs, and a coarse run's fidelity is only as good as the channel data fed to it.
Say this wherever the coarse run's numbers are reported.

---

## 4. Risks / watch-items

- **A sub-grid channel is a parameterization, not resolved physics.** It restores
  conveyance and storage; it does not restore planform, meander routing, or
  overbank velocity structure. State that in the docs and never let a coarse run
  carry a fidelity claim the fine run did not earn.
- **`dx/w` amplification is sharp.** A 5 m channel in a 200 m cell turns 1 mm of
  cell-mean depth into 4 cm of channel column. That is correct storage, but it
  makes the timestep channel-controlled (§1.3) and it makes a *bad* width field
  (a stray small `w`) an instant timestep collapse. The width field is validated
  on load: `0 ≤ w ≤ dx`, `d ≥ 0`, and a warning on any `w` below a floor.
- **Coarsening a bed by block mean removes barriers.** A one-cell levee crest in a
  4×4 block is averaged away — the coarse run will not hold water the fine run
  holds. `[[structures]]` are applied **after** coarsening (their crests are
  authored in metres, not cells), which covers engineered barriers; natural ridges
  are a documented limitation of coarsening, not a bug to hide.
- **Cell indices are resolution- and mosaic-dependent.** `[[inflow]] cell`,
  `[[structures]] cells`, `pool`, `outlet` are all `(row, col)`. Coarsening or a
  window changes what a row/col means. M6 resolves indices **after** the domain is
  final and reports the mapping; a scenario that indexes outside the resolved
  domain is a loud error.
- **Memory.** Fields scale as `ny·nx`; a 8192² domain is ~1 GB across the LI field
  set. The run prints the resolved domain, `dx`, cell count and estimated field
  memory before stepping, so an out-of-memory run is predictable rather than a
  CUDA abort.

---

## 5. Acceptance / demo — MET

- [x] **Mosaic domain assembly** (`solver/io/mosaic.py`): the domain is the tile
      set, placed by each tile's `(row, col)` origin, with `[grid] tiles`
      (`all`/`first`) and `[grid] window` selecting it. Only the tiles a window
      touches are read (`np.memmap`). Uncovered cells are filled flat **and
      counted**, reported at startup. Gated by: a tiled array reassembles to its
      source exactly; a run over a 2×2 mosaic is **bitwise-identical** to the same
      run over the equivalent single bed.
- [x] **Sub-grid channels** (`solver/core/channels.py` + the LI kernels): the
      storage curve `h → η`, a two-component face update (channel with `R = A/P`,
      floodplain with the M1 formula, recombined into the same total flux), a CFL
      that reduces over the water *column*, `[channels]` scalar-or-field geometry,
      validation on arming, and a loud `hllc_fv` scope gate. Armed with `w ≡ 0`
      the scheme reproduces M1 **bitwise**.
- [x] **Resolution choice** (`solver/io/coarsen.py`, `[grid] coarsen = k`): block
      **mean** for the bed and the rate fields (volume-preserving), block **max**
      for channel width/depth, cell indices mapped `i // k` and refused if they
      leave the domain, partial blocks cropped and reported. `coarsen = 1` is the
      identity on every path.
- [x] **Validated** (`validation/test_subgrid_channel.py`, CPU):
      steady flow in a 20 m channel inside 50 m cells settles at the Manning normal
      depth of **its own section** — 1.296 m modelled vs 1.298 m analytic (**−0.1%**);
      overbank flow spills onto the floodplain with the mass gate holding (3.6e-8);
      and the **reach claim** — the same 2 km reach with the channel **resolved** at
      10 m (2000 cells) and carried **sub-grid** at 100 m (20 cells) agrees to
      **0.1%** in depth (1.111 m vs 1.110 m), against analytical sections that
      themselves differ by 2.3% (the wetted-perimeter term the sub-grid model adds
      back). Storage agrees to 5.5%.
- [x] **§7.3 frame tiling**: frames larger than `tile_size` (512) split into a
      row-major tile grid whose geometry is listed once; tiles reassemble to the
      Zarr frame exactly, edge tiles clipped; an untiled export is byte-for-byte
      what M2 wrote. The Godot reader takes the results grid from the manifest
      (a mosaic/coarsened run need not match terrain tile 0) and blits tiles.
- [x] **Demo** `scenarios/reach_basin.toml` (+ `scripts/make_reach_demo.py`,
      `pipeline/channels.py`): a **6×6 tile mosaic**, 1536² @ 50 m = **76.8 km
      square**, run at `coarsen = 2` → **768² @ 100 m**, with 2232 sub-grid channel
      cells (7–45 m wide) carrying a 15 mm/hr storm plus a 60 m³/s inflow wave over
      12 h. The channel deepens from a 1.5 m column at 1 h to 3.75 m at 12 h,
      overbank cells grow 25 → 1201, 11.6 Mm³ leaves the open southern edge, and the
      frames export lands on a **2×2 tile grid**. Mass **2.79e-7**; 2.8 min on the
      CPU backend (this session has no GPU). **The sign-off re-run on the RTX 5090
      re-measured only some of these**: the resolved domain, the 2232 channel cells,
      `h_max` → channel column, the 2×2 tile grid and mass (3.08e-7). The **overbank
      25 → 1201 and 11.6 Mm³ out figures remain the CPU session's** — the solver does
      not print them and they were not instrumented at sign-off.
- [x] `ruff` + `ruff format` clean; **228 tests green** (221 without `--extra geo`).
- [x] **Signed off on GPU + Godot** (2026-08-09) — see the next section.

### Sign-off run (GPU + Godot, 2026-08-09)

The two things the authoring session could not run — the demo on real hardware and
the Godot reader — were run. Pass criteria were written down before the output was
looked at (mass ≤ 1e-6, and if not, re-run with `[channels]` stripped: an identical
residual means the documented float32 source drift, a divergent one means a channels
regression).

- **228 tests green** on the GPU box with `--extra geo` (no skips).
- **`scripts/make_reach_demo.py`**: 36 tiles, 1536² @ 50 m, 6144 channel cells,
  `check_continuity` → **connected**.
- **`scenarios/reach_basin.toml` on the RTX 5090** (Warp 1.14, CUDA 12.9): domain
  resolved 1536² from 36/36 tiles → **768² @ 100 m**, ~15.8 MB of state fields,
  **2232 channel cells**. Mass max rel **3.08e-7** vs the CPU backend's 2.79e-7 —
  the CUDA↔CPU delta is 2.9e-8, an order below the residual itself, so reduction
  order and FMA are not what sets the number. **Inside the 1e-6 gate; no re-run
  needed.** `h_max` 2.036 m at 12 h ⇒ a **3.73 m channel column** through the
  storage curve, reproducing the authoring session's 3.75 m. `status.json` reached
  `state="done"`. The overbank-cell growth and outflow volume were **not**
  re-measured — those stay the CPU session's numbers.
- **Frame tiling, on real output**: `manifest.json` reports the expected **2×2**
  grid with edge tiles clipped (512+256). All **25 frames reassemble byte-exactly**
  to the Zarr (`max|tiles − zarr| = 0.0`), including the non-square edge tiles.
- **Godot read path** (`--rbverify`, headless): 25 frames loaded, 8907 wet samples,
  against the hard case — a coarsened mosaic whose grid matches no terrain tile.
- **Godot render** (`--rbshot`): frame 24/24, `t = 43200 s`, max **2.04 m** — the
  solver's `h_max` to the pixel. The water plane draws at 76.8 km against a 28.8 km
  terrain quad, which is *positive* evidence the manifest-driven geometry is applied
  (and the reason the composite looks wrong — see the terrain limitation below).
- **Full loop** (`--rblaunch`): the viewer launched the solver as a subprocess and
  polled `starting → running → writing → done`, then auto-reloaded. Mass
  **2.12e-8** — bit-for-bit M2's recorded figure, so M6 left the pre-M6 path alone.
  That run is 1024², so it **also tiles** (2×2 of 512) and its 13 frames likewise
  reassemble exactly: the live loop did exercise the new write+read, not just M2's.
  **The Windows `os.replace`-onto-open-file race did not fire** across 4× the file
  handoffs — but that is one observation of a race, which is weak evidence, not a
  proof it is gone.

**New finding (recorded, not fixed): the frames export does not purge its output
directory, and every scenario writes to the same default one.** `reach_basin` and
`demo_basin_rain` both landed in `data/results/demo.zarr` + `frames/` — the default
output path ignores `[meta] name` — so running them back to back interleaved two
grids in one directory. After the second run, **61 of the 113 `.raw` files there were
orphans**: 13 pre-M6 untiled frames from July, and — the dangerous ones — **48 tiled
files (`f0013`–`f0024`) from the 768² reach run, sitting beside a manifest that
declares 1024²**, under an identical naming convention and distinguishable only by
byte size. Nothing reads them today, because `manifest.json` names every file the
viewer touches and the frame count shrank, so the survivors are past the end of the
list. But a reader that reconstructs filenames by index would decode a 768²-geometry
tile as 1024², and a human listing the directory has no way to tell which run a file
belongs to. The missing purge is what makes it possible; the shared default output
path is what makes it likely. Fix either end — or both — but the shared path is the
root.

- **Stop and confirm before M7.** ← M6 is signed off; M7 has not started.

### HANDOFF divergences (deliberate)

1. **"Multi-resolution" is resolution *choice*, not nested grids** (§1.1). The
   milestone line offers "multi-resolution / tiling-at-scale"; M6 builds the second
   plus a per-run resolution with conservative aggregation, and sub-grid channels to
   make coarsening survivable. §12's interface-conservation problem is **avoided,
   not solved** — there is no resolution interface in a run. Nested two-way patches
   remain unbuilt, and this milestone is not evidence they work.
2. **The 1D river network is not built** (the roadmap line already calls it
   optional). It carries the same 1D↔2D exchange-conservation risk for capability
   sub-grid channels largely cover at reach scale. Its gate is named for whoever
   builds it: a dedicated exchange-conservation test, before any result.
3. **Sub-grid channels are local-inertial-only**, the mirror of M5's `fixed_stage`
   being HLLC-only, and rejected loudly rather than ignored.

### Carried limitations (state these honestly)

- **The mass gate is now a precision envelope, not a comfortable margin.** A
  rain-on-grid run at reach scale is the first thing in this project to approach the
  1e-6 gate for *arithmetic* reasons: the same demo scenario at `coarsen = 4`
  (384² cells) **exceeds** it at 1.8e-6, and does so **identically with channels
  disabled** (1.87e-6), so it is float32 accumulation of a distributed source into a
  thin sheet — HANDOFF §12's "float32 conservation drift at scale", measured. The
  residual is generated *while it rains* (it stops growing when the storm stops) and
  is roughly fixed in absolute terms, so the relative error falls as the basin fills:
  the shipped demo sits at 2.8e-7 with a 15 mm/hr storm and would fail with a 2 mm/hr
  one. **Before treating a reach-scale gate failure as a bug, check the storm depth,
  the cell count and the step count.** A real fix (a compensated or float64 source
  accumulation) is a precision pass, and it should happen before sediment (M7) adds
  another distributed source on the same fields.
  **RESOLVED 2026-08-09** — the precision pass landed: per-cell Kahan compensation on
  areal sources (`solver/core/sources.py`). Re-measured on this hardware, the
  `coarsen = 4` case reads **3.77e-6 before / 1.28e-7 after** (this session measures the
  failing case worse than the 1.8e-6 recorded above, on the same scenario), and the
  shipped demo moves from 2.79e-7 / 3.08e-7 to **7.21e-8 CPU / 1.60e-7 CUDA**. The
  diagnosis above was correct in every particular — the drift is generated while it
  rains and is the source add, not the fluxes — which is what made the fix a targeted
  one. The residual floor is now flux/limiter round-off (~21 m³ on that run), which is
  untouched. See `precision-sources.md`.
- **A sub-grid channel is a parameterization, not resolved physics.** It restores
  conveyance and storage; it does not restore planform, meander routing or overbank
  velocity structure. And its geometry is calibration data: `pipeline/channels.py`'s
  hydraulic-geometry coefficients are regional inputs, wrong by factors outside the
  regime they were fitted for, which is why the CLI writes them beside the fields.
- **A channel must be continuous to convey.** `w_face = min(w_L, w_R)`, so a channel
  band that steps sideways faster than it is wide has a wall across it — and nothing
  in the depth field says so. `scripts/make_reach_demo.py::check_continuity` exists
  because the first demo basin did exactly that.
- **Coarsening removes natural barriers.** A one-cell ridge inside a `k × k` block is
  averaged away and the coarse run will not hold water the fine run holds.
  `[[structures]]` survive (applied after coarsening, at their authored crest); ridges
  do not.
- **The fine-vs-coarse gate is an equivalence claim about our own two runs**, not a
  benchmark validation, and the two models are not identical by construction (the
  resolved 2D channel has no side-wall drag; the sub-grid one does). Both numbers and
  both analytical sections are printed rather than hidden inside the tolerance.
- ~~**The viewer's terrain layer still loads tile 0 only.**~~ **Fixed 2026-08-09**
  (`viewer-terrain-mosaic.md`). At M6 sign-off the water path was verified
  (manifest-driven geometry, tiles blitted, frames exact) but the terrain under it was
  a single M0 tile, so the reach demo rendered 76.8 km of water over a 28.8 km quad
  from a *different* DEM — both halves correct, composite wrong. The run's own bed now
  ships with the frames and is what the viewer renders. The two claims stayed apart
  until each was proven: `--rbverify` gates the alignment one on its own.
- **Reach scale is still bounded by device memory.** The run prints its resolved
  domain and estimated field memory before stepping, but nothing streams: a domain
  larger than the GPU is a failure, not a slow run.
