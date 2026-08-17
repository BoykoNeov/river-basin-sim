# Scheduler pass — equal steps to the sync point

**Status: planned, 2026-08-17.** The first of the two items M7 measured and
deliberately did not fix (`roadmap.md` → "Carried out of M7", item 1). Not a
milestone: a scoped correctness pass on one line of arithmetic in
`solver/scheduler.py`, with a wide blast radius because that line sets the `Δt`
sequence of every run there has ever been.

Written after reproducing the defect and both candidate fixes end to end
(§2.1); every number below is measured on this machine at plan time, not carried
from the M7 sign-off.

---

## 1. The defect

`MultiRateScheduler.ticks` clamps every fast step so it cannot cross a sync point:

```python
dt = float(dt_fn())
dt = min(dt, self.next_sync(t) - t)      # scheduler.py:154
```

Sync points are the output cadence, forcing breakpoints (rain on/off, hydrograph
knots, stage-curve knots), `end_time` and every slow-process activation. The
algebra is M1–M4's inline loop, lifted verbatim into M5's scheduler, so this is not
an M5 regression and not a morphology bug — it reproduces with sediment never armed.

The clamp fills a span of length `S` with `floor(S/Δt)` full steps **plus one
remainder step**, and that remainder is whatever is left over. It is usually tiny.
Local-inertial does not tolerate an abrupt shorten-then-restore: it excites a
short-wavelength standing mode in the depth field.

**Measured** — `validation.bedwave` with the bump removed, so the reach is uniform
and steady and *every* departure is spurious. Water only, sediment never armed,
5835 s, 240 cells at 2.5 m, interior window cells [12, 228):

| sync cadence | steps | clamped | interior ripple | `h` range | mass |
|---|---|---|---|---|---|
| none | 12 879 | 1 | 0.010 mm | 1.4959..1.4959 | 7.7e-09 |
| 900 s | 12 888 | 7 | 0.165 mm | 1.4958..1.4960 | 7.8e-09 |
| 300 s | 12 911 | 20 | 9.775 mm | 1.4922..1.5019 | 1.7e-08 |
| 45 s | 13 003 | 130 | 14.248 mm | 1.4894..1.5037 | 2.5e-08 |
| 22.5 s | 13 088 | 260 | 74.424 mm | 1.4434..1.5178 | 5.0e-09 |
| 11.25 s | 14 062 | 519 | **2341.566 mm** | **0.2301..2.5716** | 9.5e-09 |

This reproduces the M7 plan §4 table to the last digit, which is the check that the
harness written for this pass measures the same thing the finding was recorded from.

**The mass gate reads 1e-8 in every row, including the one where a 1.5 m reach
swings between 0.23 m and 2.57 m.** Mass is conserved exactly. Only the water's
position is wrong, and the repository has no gate that can see that.

### What the clamp actually does to `Δt`

New measurement, and the one that names the mechanism. Over the steady tail
(`t > 2000 s`, past the fill-up):

| mode | cadence | `Δt` range | largest step-to-step jump | mean \|jump\| |
|---|---|---|---|---|
| clamp | 45 s | 0.007916 .. 0.455884 | **5 652 %** | 29.4 % |
| clamp | 11.25 s | 0.000774 .. 0.456831 | **58 575 %** | 44.1 % |

The remainder step is not "slightly shorter". At an 11.25 s cadence it is routinely
**one 585th** of the step either side of it. That is the forcing.

### Why nothing caught it

Every shipped scenario runs a 900 s output cadence — 0.165 mm here, invisible. The
validated benchmarks (dam-break, Manning normal depth to 0.59 %, the EA suite) all
bound a reach against a reference that a large ripple would break, so they are
sensitive in principle but none of them runs a frequent sync cadence. M7 is the
first milestone that *wants* one, because the morphological Courant number pushes an
erosive reach toward short activation intervals — and morphology then rectifies the
oscillation into a permanent bed signature instead of letting it average out.

## 2. The fix

Fill the span with **equal** steps instead of full steps plus a remainder: with
`S` the distance to the next sync point and `Δt_raw` the scheme's state-derived step,

```python
n  = max(1, ceil(S / Δt_raw))
dt = S / n
```

`n` is the smallest step count that keeps every step at or under the scheme's own
limit, so `dt <= Δt_raw` always: the scheduler still only ever *shortens* the step
the scheme asked for, and the sync point is still landed on exactly. The step count
per span is unchanged (`ceil(S/Δt_raw)` either way) — this redistributes the
remainder rather than adding work.

### 2.1 Which reading of "equal steps" — measured, and it matters

The M7 plan recorded the fix as one sentence, which admits two implementations. They
are not close.

**(A) Freeze `n` at the span start** — compute `n` once, take `n` steps of `S/n`
without re-querying `dt_fn` mid-span. Literally equal steps.

**(B) Recompute every step from the remaining span** — query `dt_fn`, recompute
`n = ceil(remaining/Δt_raw)`, step `remaining/n`. Self-correcting; the steps are
equal only while `Δt_raw` holds still.

Same fixture, same six cadences:

| cadence | clamp | **(B) recompute** | (A) freeze |
|---|---|---|---|
| none | 0.010 mm | 0.011 mm | **534.797 mm** |
| 900 s | 0.165 mm | 0.240 mm | **534.392 mm** |
| 300 s | 9.775 mm | 0.285 mm | **534.521 mm** |
| 45 s | 14.248 mm | **0.009 mm** | 0.009 mm |
| 22.5 s | 74.424 mm | **1.908 mm** | 1.907 mm |
| 11.25 s | 2341.566 mm | **1.908 mm** | 1.907 mm |

(A) is a defect, and the harness says why: it exceeded the state-derived step by up
to **1.59×** (`dt/Δt_raw` = 1.5906 at the 900 s cadence). A frozen `n` is a `Δt` the
scheduler *invented* and then held while the state moved underneath it — it breaks
CFL and it breaks the module's own contract that it "never invents one, it only ever
clamps the one the scheme computed". Its long-cadence rows are that violation
showing up as a wrecked reach. **Ship (B).** The M7 plan's recorded numbers
(14.248 → 0.009 at 45 s, 2341.566 → 1.908 at 11.25 s) are (B)'s, reproduced here
exactly, so the finding as recorded is sound — only its one-line description was
ambiguous.

### 2.2 What (B) leaves behind

| mode | cadence | `Δt` range | largest jump | distinct `Δt` |
|---|---|---|---|---|
| (B) | none | 0.456821 .. 0.456821 | **0.00 %** | 1 |
| (B) | 900 s | 0.456453 .. 0.456621 | 0.04 % | few |
| (B) | 45 s | 0.454545 .. 0.454545 | **0.00 %** | 1 |
| (B) | 22.5 s | 0.441176 .. 0.450000 | 1.96 % | 2 |
| (B) | 11.25 s | 0.441176 .. 0.450000 | 1.96 % | 2 |

The largest `Δt` discontinuity in the whole run goes from **58 575 % to 1.96 %**.
The residual mechanism is *re-quantization*: when `Δt_raw` drifts below the current
`S/n`, `n` ticks up by one and `dt` drops by `1/n`. That jump is bounded by `1/n`,
so it is small mid-span and largest in the last step or two of a span — worth a
sentence in the docstring, and it is why the gate is written on a short cadence
where `n` is smallest.

**This also closes one of the two numbers the M7 plan left unexplained.** 22.5 s and
11.25 s "both settle at 1.908 mm" because they quantize to the *same two step sizes*
(0.441176 and 0.450000 s) — same forcing, same response. Not a coincidence and not a
floor of the method.

### 2.3 The 1.908 mm is a boundary artefact, not the fix's floor

Advisor's call, and the experiment confirms it. Trimming cells off the ends of the
measurement window:

| window | 22.5 s | 11.25 s |
|---|---|---|
| [12, 228) — `fx.interior` | 1.908 mm | 1.908 mm |
| [14, 226) | 0.325 mm | 0.315 mm |
| [20, 220) | 0.302 mm | 0.289 mm |
| [36, 204) | 0.239 mm | 0.231 mm |

The profile says exactly where it lives. Departure from the design normal depth,
in mm:

```
cell    0      1      2      3      4      5      6      7      8      9
    -441.25   0.02  -0.30   0.50  -0.99   1.01  -0.45  -1.30   2.15   0.45
cell   10     11     12     13     14     15     16     17     18     19
      -3.71  -3.94  -1.94  -0.58  -0.13  -0.05  -0.04  -0.03  -0.04  -0.03
```

Cell 0 is the inflow cell, sitting 441 mm below normal depth — that is the M7 gotcha
about a point source being a splitting artefact with a scale, and it is outside the
window already. What the window's first two cells catch is that source's **spatial
adjustment**, a decaying alternating train that is down to 0.03 mm by cell 15 and
gone by cell 20. The toe end is a smooth monotone drawdown toward the open boundary
(−0.33 mm at cell 220 falling to −0.42 mm at cell 239), which is backwater, not
ripple.

So the honest interior floor after the fix is **~0.3 mm at every cadence** — but
see §2.4: the gate should not be written on this statistic at all.

**Still open, and cheap to close during the build:** the M7 plan also records "a
32.9 mm end-cell offset identical across all three cadences". Nothing in these runs
reproduces that figure (the end cells here read −441 mm at the head and −0.42 mm at
the toe), so it was measured on a configuration this pass has not reconstructed.
One attempt to resolve it, then strike it from §4 when rewriting that section; do
not carry it forward unexplained a second time, and do not spend build time beyond
that one attempt.

### 2.4 The gate statistic is curvature, not spread

§2.3 nearly led this plan into fitting the gate to the artefact. A max−min spread
over a window **cannot tell an oscillation from a slope**: the toe's smooth
−0.33 → −0.42 mm backwater is legitimate physics and contributes to the same
statistic as a standing wave. Choosing the window by which trim gives a small
number ([36, 204) reads 0.239 mm, [20, 220) reads 0.302 mm) is calibration, not
validation.

What actually discriminates is that the clamp's failure mode is
**short-wavelength**: it alternates cell to cell, while backwater and the inflow
train are smooth or spatially decaying. So gate the **second difference of depth
along the reach** — near zero for any smooth profile at any window, and enormous
under the clamp. Measured on the **untrimmed** `fx.interior` window, cells
[12, 228):

| cadence | spread (clamp → B) | max&#124;∂²h&#124; (clamp → B) | rms ∂²h (clamp → B) |
|---|---|---|---|
| none | 0.010 → 0.011 mm | 0.0134 → 0.0165 mm | 0.0054 → 0.0057 mm |
| 900 s | 0.165 → 0.240 mm | 0.2074 → **0.0103** mm | 0.0722 → **0.0040** mm |
| 300 s | 9.775 → 0.285 mm | 8.198 → **0.0525** mm | 3.505 → **0.0209** mm |
| 45 s | 14.248 → 0.009 mm | 13.685 → **0.0116** mm | 4.751 → **0.0049** mm |
| 22.5 s | 74.424 → 1.908 mm | 111.99 → **0.9157** mm | 71.83 → **0.0677** mm |
| 11.25 s | 2341.6 → 1.908 mm | **4369.8 → 0.9214** mm | **773.6 → 0.0680** mm |

**4700× separation on max curvature at the worst cadence, 11 000× on rms, with no
trim at all.** The residual 0.92 mm of curvature under the fix is the two inflow-
adjacent cells the profile in §2.3 shows; the rms of 0.068 mm says it is two cells
and not a reach.

Note also what the fix does at the cadences that already looked fine: the clamp's
**900 s and 45 s rows carry 0.21 and 13.7 mm of curvature** that the spread
statistic reported as 0.165 and 14.248 mm of harmless-looking wobble. Every shipped
scenario runs 900 s, so the defect was present in production runs, twenty times
smaller than at 45 s but not absent.

A sign-change *count* was also tried and is useless — it reads 0.5–0.9 of the
maximum in every row including the clean ones, because float32 noise alternates
everywhere. Curvature magnitude is the statistic; spread stays a printed
diagnostic.

## 3. Decision: ship it unconditionally, and retire the bitwise invariant

**Recommendation: no config switch. The new behaviour is the only behaviour.**

M4, M5 and M6 all carry the claim that *pre-M5 runs are bitwise-identical*, guarded
in-tree by `solver/test_scheduler.py::test_matches_the_pre_scheduler_inline_loop_exactly`,
which replays the M1–M4 clamping loop as an executable reference. This pass ends
that claim on purpose.

It is worth being precise about what is being given up, because it sounds larger
than it is. That invariant was a **refactor-safety proof** — evidence that M5's
extraction of the clock into its own module was inert — not a product guarantee that
a given scenario returns a given float forever. The repository has already broken
figure-level identity deliberately once, when the precision pass moved every
rain-bearing mass number, and it did so for the same reason: the old behaviour was a
defect, not a contract.

Against a `[run]` key: it doubles the surface every gate has to cover, and it ships a
selectable footgun whose wrongness **no gate can see** — a user who picks the old
mode gets a silently wrong bed with a clean mass balance. The precision pass set the
precedent by arming itself automatically rather than through a key.

Four invariants replace the retired one, each a test in `solver/test_scheduler.py`:

1. **`dt <= dt_raw` on every step** — the scheduler still only shortens.
2. **Every sync point is landed on exactly** (within `EPS_T`) and none is crossed.
3. **Frame count and output times are unchanged** — `n_frames`, `output_times` and
   the `is_output` tick set are pure schedule arithmetic and must not move.
4. **Run-to-run reproducibility holds** — same config, same machine, same floats.
   Determinism (HANDOFF §8/§12) is the property that actually matters and it is
   untouched; only the sequence changes, once.

`_reference_ticks` stays in the test file with a tombstone comment naming this
commit as what ended its assertion, so a future reader can see the old algebra and
why it went.

## 4. Validation — the gate that was missing

The point of this pass is not only to fix the clamp; it is to leave behind a test
that would have caught it. **Written first, and shown red against today's
scheduler.**

`validation/test_clamp_ripple.py`, driving `MultiRateScheduler.ticks` directly —
not a hand-rolled copy of the clamp, since the scheduler's algebra *is* the thing
under test. `validation/bedwave.py`'s `drive()` cannot serve: with `morphology=False`
it sets `edge = end` and never clamps at all, so it cannot reproduce the water-only
column. The new driver takes the bedwave state (bump removed) and runs it under a
real scheduler whose `output_every` is the sync cadence, with sediment never armed.

The gates, in order of how much they discriminate:

- **Depth curvature on a uniform steady reach, untrimmed interior, parametrized
  over cadence (900 s, 45 s, 11.25 s).** The primary assertion, per §2.4:
  `max|∂²h|` over cells [12, 228) must stay under **1.5 mm** at every cadence.
  Measured today: 0.21 / 13.7 / **4369.8**. Measured after: 0.010 / 0.012 / 0.921.
  That is a 4700× separation at the worst cadence with ~1.6× margin on the bound,
  and no window was chosen to get it. Tolerances get pinned from the run that lands
  the fix, not from this plan.
- **Cadence independence — the strongest gate and the user-facing property.** All
  three cadences share one window, so any artefact common to them cancels and the
  window question does not arise. Assert the curvature *ratio* between the shortest
  cadence and the no-sync-point control: `max|∂²h|(11.25 s) / max|∂²h|(none)` reads
  **326 000 today and 56 after**, so a bound of ~200 has a 3.6× margin and self-
  scales rather than hard-coding a millimetre. Ratio-style, in the idiom
  `solver/core/test_sources.py` established, so it cannot pass by measuring nothing.
- **`Δt` discontinuity bound.** The largest step-to-step relative change over the
  steady tail: **58 575 % today, 1.96 % after**, asserted under a few percent. The
  mechanism gate — it fails for the right reason if someone reintroduces a
  remainder step.
- **Interior spread, printed not asserted.** 2341.566 → 1.908 mm is the headline
  number and belongs in the output, but §2.4 is why it is not the gate.
- **Mass is asserted to stay green and explicitly documented as not the point.** It
  reads 1e-8 on both sides of the fix; a comment says so, so nobody reads a green
  mass balance as evidence this works.

Plus the four scheduler invariants of §3, unit-tested against a stub `dt_fn` with no
GPU — including the degenerate cases that are now load-bearing arithmetic:

- `S <= Δt_raw` gives exactly one step of `S` (`n = 1`);
- **`S ≈ 2·Δt_raw`**, the residual path back to the defect. With `n = 2`, a mid-span
  re-quantization to `n = 3` is a 33 % jump — the same shape as the bug, smaller.
  The `1/n` bound of §2.2 must be **asserted**, not assumed from the 1.96 % this
  fixture happened to produce;
- duplicate sync times resolve through `min` unchanged;
- `n >= 1` always, and steps-per-span never exceeds the old count.

## 5. Re-baseline surface

"Mostly re-baselining" hides a lot, so it is enumerated. Everything below runs a
simulation whose `Δt` sequence moves.

**Expected to move, and by how much:**

- The new ripple gate — that is the point.
- `validation/test_channel_flow.py` (Manning normal depth, currently within 1 % and
  0.59 %). The uniform-reach measurement is precisely the quantity the ripple
  corrupts, so this should get *better* or hold. If it degrades, stop: that inverts
  the premise.
- `validation/test_morphology_gates.py` — the threshold pair's **0-cells-moved claim
  is bit-exact** at 0.9 θ_c. A bit-exact assertion is the fragile kind under any
  trajectory change; expect to re-establish rather than re-tolerance it, and keep it
  bit-exact if it still is.
- `validation/test_bed_wave.py` — 0.993 c_b with a ±20 % gate and a 7 % spread
  across a 32× interval range. Comfortably inside its gate, but the *number* moves
  and the docstring quotes it.

**Expected to hold within their own tolerances** (analytic references, not stored
trajectories): `test_dam_break.py`, `test_ea_test1/2/3.py`, `test_subgrid_channel.py`,
`test_fixed_stage.py`, `test_hllc_*.py`, `solver/test_run.py`,
`solver/processes/test_reservoir.py`. HLLC deserves a specific look: its `Δt` is
velocity-dependent and moves faster than LI's, so re-quantization fires more often
there. Nothing predicts a problem; it is the place to check first if one appears.

**The mass gate proves nothing here.** It reads 1e-8 on both sides. Saying so in
advance is what makes it legible if something *does* move.

**Baseline, taken before any change (2026-08-17): 336 tests green, exit 0.**

**Recorded figures that go stale.** Every scenario mass number in `CLAUDE.md`, the
`M1`–`M7` plans and `precision-sources.md`, plus `--rblaunch`/`--rbverify` at
2.59e-8. Re-measure the demos on GPU and CPU (`demo_basin_rain`, `river_reach`,
`spatial_fields`, `river_reach_hllc`, `reservoir_release`, `reach_basin`,
`reach_alluvial`) and update the tables. Anything left stale must be said so
explicitly, with why.

**One item that is not a number swap.** `validation/bedwave.py` constraint (7)
currently *derives the 45 s activation interval from this artefact* — its lower
fence is "shorter intervals clamp more and ripple more". After the fix that fence
largely dissolves, so the fixture's design justification needs rewriting, not
re-tolerancing. The M7 plan §4 prose it cross-references, and the long remedy
comment in `solver/run.py` (~line 452), need the same treatment: they currently tell
the reader that a short interval is dangerous for a reason that will no longer exist.

## 6. Scope fence

**In:** `solver/scheduler.py`, the new ripple gate, the four replacement invariants,
the re-baseline, and the doc rewrites listed in §5. One commit, in the shape
`precision-sources.md` set.

**Out, explicitly, so they do not creep in:**

- The morphological Courant diagnostic overstating on a wetting front (M7 carried
  item 2 — its own commit, its own before/after).
- The viewer shader lifting a sub-grid channel by `bed + depth` instead of its
  storage curve.
- Compensating flux-divergence round-off, or point-source compensation.
- Anything about the inflow point source's 441 mm drawdown. It is a known,
  documented splitting artefact; this pass only has to keep its gate window clear
  of it.

## 7. Build order

0. **Suite baseline** — done, 336 green.
1. **Gate first, red.** `validation/test_clamp_ripple.py` + the driver. Show the
   11.25 s curvature row failing against today's scheduler (4369.8 mm against a
   1.5 mm bound) and record the failure output. Independent of anything else here.
2. **Change `ticks`.** Four lines, plus the docstring — including the honest note
   that the module now chooses `n` and therefore does more than clamp, and the
   `1/n` re-quantization bound.
3. **Replacement invariants** in `solver/test_scheduler.py`; tombstone
   `_reference_ticks`.
4. **Full suite**, triaging every failure against §5's prediction. A failure that
   was predicted is a re-baseline; one that was not is a stop-and-understand.
5. **Re-measure the demos**, GPU and CPU.
6. **Docs**: this file's results section, `roadmap.md` strike-through of carried
   item 1, `CLAUDE.md` status + the "a clamped step is not a free step" gotcha
   rewritten as fixed-with-a-gate, `bedwave.py` constraint (7), M7 plan §4,
   `run.py`'s remedy comment.
7. **Commit**, ruff clean, suite green.

## 8. Measured result

*(filled in at completion)*
