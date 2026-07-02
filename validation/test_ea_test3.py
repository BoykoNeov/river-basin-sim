"""UK EA SC080035 Test 3 -- momentum conservation over a small obstruction (M4 step 10).

The second M4 EA benchmark case (SC080035, "Benchmarking of 2D Hydraulic Modelling
Packages", Appendix A.3). Test 3 assesses a package's ability to **conserve
momentum over an obstruction**: a flood wave runs down a mild slope into a
depression whose static capacity is just filled by the inflow, and the report's
core finding is that a model carrying inertia lets *some* water pass over the
obstruction into the second depression, purely because of momentum.

**What the report actually discriminates (and what it does not).** Read literally
(report p.105), Test 3 separates *"packages without inertia terms"* (diffusive /
zero-inertia: Flood Risk Mapper, JFLOW-GPU, Direct RFSM, UIM -- which predict *no*
overtopping) from *"2D hydrodynamic packages with inertia terms"* (which overtop).
**Both of our schemes carry inertia** -- HLLC solves the full momentum equations,
and the M1 Bates ``local_inertial`` scheme retains the local acceleration
``dq/dt`` that carries flow over the crest. So both sit on the *same* (with-inertia)
side of the report's split, and -- confirmed empirically -- **both overtop**. This
is *consistent* with the report, not a discriminator between our two schemes. (An
early framing of this case as an "HLLC-vs-LI discriminator" was wrong for exactly
this reason; the LI arm is printed below as honest context, never as a gate.)

**The gate is a within-HLLC momentum-conservation test.** The clean, non-circular
way to demonstrate the report's physics with one scheme is to hold *everything*
fixed except the arrival momentum:

* the obstruction crest is fixed by an **offline, scheme-free volume anchor** --
  depression-1 capacity is integrated geometrically up to the crest, and the total
  inflow volume is set to a fixed fraction of it (0.9), so the *static* equilibrium
  sits just **below** the crest (no static spillover) in every run;
* two runs inject the **same total volume** over the same closed domain and the
  same volume-anchored crest, differing *only* in hydrograph sharpness:
  - a **gentle** pulse (tb=300 s) -- the empirical null: low arrival momentum,
    depression-1 settles below the crest, the second depression stays **dry**;
  - a **sharp** pulse (tb=80 s, identical integral) -- the signal: the flood wave
    arrives with momentum, HLLC's advective transport carries a splash over the
    crest, and the second depression **wets to a few cm** (matching the report's
    ~5-6 cm rise) -- while depression-1 *still* settles below the crest, proving the
    overtopping is dynamic, not static.

The only thing that put water over the obstruction is momentum -- the exact
capability Test 3 targets. Crest is never nudged to the outcome; only the
hydrograph peakedness (energy) differs between the two runs, at fixed volume.

**Faithful-form resolution.** The report's 300 m x 100 m domain at 5 m (~1200 nodes)
and 15 min run are already CI-tractable, so unlike Test 2 this runs at the *exact*
specified resolution and horizon on Warp's CPU backend. The DEM is an analytic
reconstruction of the described profile (1:200 slope, two depressions at x=150/250
separated by an obstruction); the exact ``Test3DEM.asc`` is in the EA dataset and
not needed. Datum is kept low (bed ~0.5-2 m) on purpose: a spurious f32 rest
velocity near the crest would be a fake momentum source that could counterfeit the
overtop, and Test 3 gates on relative behaviour so the absolute datum is free.

**Pass criteria (qualitative + mass, per the report).** Per the plan (§3 step 10:
"don't invent an nRMSE the report doesn't define"), the gate asserts the report's
*qualitative* momentum finding plus the always-on float64 mass gate.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core.grid import H_DRY
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.schemes import get_scheme
from solver.core.state import State
from solver.io.config import Inflow
from solver.processes.inflow import InflowInjector

wp.init()
DEV = "cpu"
_EDGES = ("north", "south", "east", "west")

LX, LY = 300.0, 100.0  # m (report: 300 longitudinal x 100 transverse)
DX = 5.0  # m (report resolution; 60 x 20 ~ 1200 nodes)
NX, NY = int(LX / DX), int(LY / DX)
SLOPE = 1.0 / 200.0  # report: 1:200 downslope, high at the west (inflow) end
Z_BASE = 0.5  # low datum -- protect the momentum measurement from f32 rest velocity
FILL_FRAC = 0.9  # inflow volume as a fraction of depression-1 capacity-to-crest


def build_dem(D1: float = 1.0, D2: float = 1.0, wD: float = 22.0):
    """Analytic Test 3 bed (x-only): 1:200 plane (high at west) minus two wells.

    Two Gaussian depressions at x=150 m and x=250 m; the plane between them forms
    the obstruction. Returns ``(bed[NY, NX], x_centres[NX])``.
    """
    xc = (np.arange(NX) + 0.5) * DX
    zx = Z_BASE + SLOPE * (LX - xc) - D1 * np.exp(-0.5 * ((xc - 150.0) / wD) ** 2)
    zx = zx - D2 * np.exp(-0.5 * ((xc - 250.0) / wD) ** 2)
    bed = np.repeat(zx[None, :], NY, axis=0).astype(np.float32)
    return bed, xc


def crest_and_capacity(bed: np.ndarray):
    """Scheme-free volume anchor: obstruction crest + depression-1 capacity to it.

    Crest = highest bed between the two wells. Depression-1 capacity = geometric
    volume of the basin west of the crest filled up to the crest level (cells where
    the rising plane exceeds the crest contribute nothing). Returns
    ``(crest_z, crest_col, capacity_m3, col150, col250)``.
    """
    zx = bed[0]
    c150, c250 = int(round(150.0 / DX)), int(round(250.0 / DX))
    crest_col = c150 + int(np.argmax(zx[c150:c250]))
    crest = float(zx[crest_col])
    fill = np.maximum(0.0, crest - zx[:crest_col])  # west-of-crest basin
    capacity = float(fill.sum()) * DX * LY  # m^3 (full 100 m width)
    return crest, crest_col, capacity, c150, c250


def _triangular_inflows(volume: float, tb: float) -> list[Inflow]:
    """Triangular hydrograph (integral == ``volume``, base ``tb``, peak at tb/2),
    split evenly across the whole west inflow line (col 0, every row)."""
    per = (2.0 * volume / tb) / NY
    return [
        Inflow(cell=(i, 0), hydrograph=[(0.0, 0.0), (0.5 * tb, per), (tb, 0.0), (1.0e9, 0.0)])
        for i in range(NY)
    ]


def _run(scheme: str, bed: np.ndarray, inflows: list[Inflow], alpha: float, col250: int, row: int):
    """Run one scheme to t=900 s on the closed domain; return diagnostics.

    Returns ``(h, ledger, vmax, p2_last300_max)`` where ``vmax`` is the peak wet-cell
    speed at t_end (NaN for LI, which stores staggered qx/qy not cell-centred hu/hv)
    and ``p2_last300_max`` is the largest P2 depth over the final 300 s (a settling
    check -- the null must stay dry, not merely be not-yet-arrived).
    """
    st = State.from_bed(bed, dx=DX, depth=0.0, manning=0.01, device=DEV)
    st.set_open_boundaries({e: "closed" for e in _EDGES})  # all closed (report)
    inj = InflowInjector(inflows, st.grid, DEV)
    ledger = MassLedger.from_state(st)
    sch = get_scheme(scheme)
    knots = [b for b in inj.breakpoints() if 0.0 < b < 900.0]

    t, t_end, p2_last300 = 0.0, 900.0, []
    while t < t_end - 1e-9:
        dt = sch.compute_dt(st, alpha=alpha, dt_max=5.0)
        nxt = min([b for b in knots if b > t + 1e-9], default=t_end)
        dt = min(dt, nxt - t, t_end - t)
        ledger.add_inflow(inj.apply(st, t, dt))
        sch.step(st, dt=dt)
        t += dt
        if t > t_end - 300.0:
            p2_last300.append(float(st.h.numpy()[row, col250]))

    ledger.record(st, t)
    vmax = float("nan")
    if st.hu is not None:
        h = st.h.numpy()
        wet = h > H_DRY
        if wet.any():
            spd = np.sqrt(st.hu.numpy()[wet] ** 2 + st.hv.numpy()[wet] ** 2) / h[wet]
            vmax = float(spd.max())
    return st.h.numpy(), ledger, vmax, (max(p2_last300) if p2_last300 else 0.0)


def test_ea_test3_hllc_overtops_obstruction_by_momentum():
    """EA Test 3: at fixed volume-anchored crest, a sharp inflow (momentum) makes HLLC
    overtop the obstruction while a gentle inflow of the same volume does not."""
    bed, xc = build_dem()
    crest, crest_col, capacity, c150, c250 = crest_and_capacity(bed)
    row = NY // 2  # output points at y=50
    volume = FILL_FRAC * capacity  # static equilibrium sits just below the crest

    # Two HLLC runs: identical volume/crest/domain, only the pulse sharpness differs.
    null = _triangular_inflows(volume, tb=300.0)  # gentle -> low arrival momentum
    signal = _triangular_inflows(volume, tb=80.0)  # sharp  -> high arrival momentum
    h_n, led_n, vmax_n, p2n_last = _run("hllc_fv", bed, null, 0.45, c250, row)
    h_s, led_s, vmax_s, _ = _run("hllc_fv", bed, signal, 0.45, c250, row)

    def surf(h):  # depression-1 water-surface elevation at P1
        return float(bed[0, c150] + h[row, c150])

    p2_null, p2_sig = float(h_n[row, c250]), float(h_s[row, c250])
    print(
        f"\n[EA Test 3] crest_z={crest:.3f}@x={xc[crest_col]:.0f}  cap={capacity:.0f} m3 "
        f"vol={volume:.0f} m3 (frac={FILL_FRAC})"
    )
    print(
        f"  NULL  (tb=300): P1_surf={surf(h_n):.3f} (<crest) P2={p2_null:.4f} "
        f"vmax={vmax_n:.3f} P2max_last300={p2n_last:.4f} mass={led_n.max_rel_error:.1e}"
    )
    print(
        f"  SIGNAL(tb=80 ): P1_surf={surf(h_s):.3f} (<crest) P2={p2_sig:.4f} "
        f"vmax={vmax_s:.3f} mass={led_s.max_rel_error:.1e}"
    )

    # --- Hard gate: mass + positivity (both runs). Closed domain => stored == injected.
    for led, h in ((led_n, h_n), (led_s, h_s)):
        assert np.isfinite(h).all(), "NaN/inf in depth"
        assert h.min() >= 0.0, f"depth went negative to {h.min():.3e}"
        assert led.max_rel_error < MASS_GATE, f"mass gate broke: {led.max_rel_error:.2e}"

    # --- No static spillover: depression-1 settles BELOW the crest in BOTH runs, so
    # any P2 water is dynamic (momentum), not the pool brimming over statically.
    assert surf(h_n) < crest, f"null P1 surface {surf(h_n):.3f} reached crest {crest:.3f}"
    assert surf(h_s) < crest, f"signal P1 surface {surf(h_s):.3f} reached crest {crest:.3f}"

    # --- Null is settled, not merely not-yet-arrived: P2 stays dry through the last
    # 300 s, and the field is near rest (gentle residual pool slosh only).
    assert p2_null < H_DRY, f"null overtopped: P2={p2_null:.4f}"
    assert p2n_last < H_DRY, f"null P2 not settled dry over last 300 s: {p2n_last:.4f}"
    assert vmax_n < 0.2, f"null not settled: vmax={vmax_n:.3f} m/s"

    # --- Signal: momentum carries a few cm over the obstruction (report: ~5-6 cm),
    # a clear margin above both H_DRY and the null.
    assert p2_sig > 0.02, f"sharp inflow did not overtop by momentum: P2={p2_sig:.4f}"
    assert p2_sig - p2_null > 0.03, f"momentum margin too small: {p2_sig:.4f} vs {p2_null:.4f}"


def test_ea_test3_both_inertial_schemes_overtop_context():
    """Context (not a discriminator): both LI and HLLC carry inertia, so both overtop
    the obstruction given a sharp inflow -- consistent with the report placing both on
    the with-inertia side. Printed for the record; only mass/positivity are asserted."""
    bed, _ = build_dem()
    crest, _, capacity, c150, c250 = crest_and_capacity(bed)
    row = NY // 2
    signal = _triangular_inflows(FILL_FRAC * capacity, tb=80.0)

    print("\n[EA Test 3 context] sharp inflow, both with-inertia schemes:")
    for scheme, alpha in (("hllc_fv", 0.45), ("local_inertial", 0.7)):
        h, led, vmax, _ = _run(scheme, bed, signal, alpha, c250, row)
        tag = "HLLC" if scheme == "hllc_fv" else "LI  "
        print(
            f"  {tag}: P1={h[row, c150]:.3f} P2={h[row, c250]:.4f} "
            f"col250_wet_rows={int((h[:, c250] > H_DRY).sum())}/{NY} "
            f"mass={led.max_rel_error:.1e} hmin={h.min():.1e}"
        )
        # Both overtop (P2 > 0) -- the honest result explaining why HLLC-vs-LI is not
        # a discriminator here. Gate only the always-on invariants.
        assert np.isfinite(h).all()
        assert h.min() >= 0.0
        assert led.max_rel_error < MASS_GATE, f"{scheme} mass gate broke: {led.max_rel_error:.2e}"
        assert float(h[row, c250]) > H_DRY, f"{scheme} (with inertia) failed to overtop"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
