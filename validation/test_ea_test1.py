"""UK EA SC080035 Test 1 -- flooding a disconnected water body (M5).

The third EA case in this repo (after Test 2 and Test 3 in M4) and the one that
motivated M5's two boundary features: it is driven by a **time-varying water-level
boundary** (``fixed_stage``) at a **~10 m datum**.

**What the test is for.** A low area is separated from the main water body by a
ridge. A boundary level rises above the ridge, spills over and fills it, then falls
back -- and the far depression becomes *disconnected*. The published finding is that
it **retains** its water: it can only drain back over the ridge, so it settles at
the ridge crest and stops there, while the connected part follows the boundary all
the way down. A model that treats inundation as a bathtub fill to the boundary level
gets this qualitatively wrong -- it empties the far depression too.

**Spec provenance -- read this before quoting any number here.** The SC080035
figures pinned during M4 were not reachable when this test was written, so the
geometry below is *faithful in form, reconstructed in detail*: a small domain at a
~10 m datum, two depressions separated by a ridge, and a boundary level that rises
above the ridge and falls back over a few hours. **Pinned** (confident): the test's
purpose and its qualitative result, the ~10 m datum, the small domain, the
rise-and-fall boundary. **Reconstructed** (do not cite as the report's): the exact
domain size, cell size, bed elevations, ridge height, and the stage curve's times
and levels. The gate is therefore set on the **published qualitative finding**
plus the always-on float64 mass gate -- never on invented tolerances. This is the
same policy M4 applied to Tests 2 and 3, and the numbers should be re-pinned from
the report before this is described as a quantitative reproduction.

**The datum -- and a correction.** M4's plan carried Test 1 as *needing* the datum
shift (``z' = z - z_ref``) to survive float32 at its ~10 m datum. This test runs
**both ways and gates both**, so the claim is measured rather than assumed -- and
the measurement does not support it: at this datum the two runs agree on every
reported depth to three decimals, and both sit comfortably inside the mass gate
(6.9e-7 shifted, 1.8e-7 raw -- the difference is rounding noise, not signal). So
the shift is *not* required at 10 m. It earns its keep further up:
:mod:`solver.core.test_datum` measures a lake at rest degrading ~3x at a 10 m datum
but ~1600x at 5000 m. Both parametrizations are kept here precisely so this stays
a measurement rather than a belief.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core import hllc
from solver.core.datum import shift_bed
from solver.core.grid import H_DRY
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State

wp.init()
DEV = "cpu"
_EDGES = ("north", "south", "east", "west")

# --- geometry (reconstructed; see the module docstring) ----------------------- #
LENGTH, WIDTH, DX = 700.0, 100.0, 10.0
POND1 = (0.0, 200.0)  # connected to the boundary; floor below the initial level
RIDGE = (200.0, 300.0)  # the divide that disconnects pond 2 as the level falls
POND2 = (300.0, 500.0)  # the disconnected water body -- the point of the test
Z_POND1, Z_RIDGE, Z_POND2, Z_HIGH = 9.5, 10.1, 9.8, 10.6
LEVEL_LOW, LEVEL_HIGH = 9.7, 10.35  # boundary water level, low and peak
T_PEAK, T_BACK, T_END = 3600.0, 10800.0, 18000.0  # rise 1 h, fall by 3 h, hold to 5 h
STAGE_CURVE = [(0.0, LEVEL_LOW), (T_PEAK, LEVEL_HIGH), (T_BACK, LEVEL_LOW), (T_END, LEVEL_LOW)]


def ea_test1_dem() -> np.ndarray:
    """Bed elevations (m AOD): pond 1 | ridge | pond 2 | high ground, west to east."""
    ny, nx = int(WIDTH / DX), int(LENGTH / DX)
    x = (np.arange(nx) + 0.5) * DX
    prof = np.full(nx, Z_HIGH, dtype=np.float32)
    prof[(x >= POND1[0]) & (x < POND1[1])] = Z_POND1
    prof[(x >= RIDGE[0]) & (x < RIDGE[1])] = Z_RIDGE
    prof[(x >= POND2[0]) & (x < POND2[1])] = Z_POND2
    return np.tile(prof[None, :], (ny, 1))


def _cols(span: tuple[float, float]) -> slice:
    return slice(int(span[0] / DX), int(span[1] / DX))


def _surface(h: np.ndarray, bed: np.ndarray, span: tuple[float, float]) -> float | None:
    """Mean water surface over the wet cells of a span (None if it is dry)."""
    hh, zz = h[:, _cols(span)], bed[:, _cols(span)]
    wet = hh > H_DRY
    return float((hh + zz)[wet].mean()) if wet.any() else None


@pytest.mark.parametrize("use_datum_shift", [True, False])
def test_ea_test1_disconnected_water_body_retains_its_water(use_datum_shift):
    """The far depression fills at the peak and **keeps** its water as the tide falls.

    Gated on the report's qualitative result, plus the float64 mass gate and
    non-negativity through the wetting/drying. Run at both datums (see the module
    docstring) so the M4 claim that Test 1 *needs* the shift is measured, not
    assumed.
    """
    bed_true = ea_test1_dem()
    z_ref = 9.0 if use_datum_shift else 0.0
    bed = shift_bed(bed_true, z_ref)
    curve = [(t, lvl - z_ref) for t, lvl in STAGE_CURVE]

    # Initial condition: still water at the low boundary level, so the run starts in
    # equilibrium with the boundary (pond 1 wet at 0.2 m; everything else dry).
    h0 = np.maximum(LEVEL_LOW - z_ref - bed.astype(np.float64), 0.0).astype(np.float32)
    st = State.from_bed(bed, dx=DX, depth=h0, manning=0.03, device=DEV)
    st.set_open_boundaries(
        {e: ("fixed_stage" if e == "west" else "closed") for e in _EDGES}, {"west": curve}
    )
    ledger = MassLedger.from_state(st)

    peak_pond2 = 0.0
    t, next_rec = 0.0, 900.0
    while t < T_END - 1e-9:
        dt = min(hllc.compute_dt(st, alpha=0.45, dt_max=10.0), T_END - t)
        hllc.step(st, dt=dt, t=t)
        t += dt
        if t >= next_rec - 1e-9:
            ledger.record(st, t)
            next_rec += 900.0
            peak_pond2 = max(peak_pond2, float(st.h.numpy()[:, _cols(POND2)].mean()))
    ledger.record(st, t)

    h = st.h.numpy()
    d1 = float(h[:, _cols(POND1)].mean())
    d2 = float(h[:, _cols(POND2)].mean())
    s1, s2 = _surface(h, bed, POND1), _surface(h, bed, POND2)
    high_max = float(h[:, _cols((POND2[1], LENGTH))].max())
    tag = "datum-shifted" if use_datum_shift else "raw ~10 m datum"
    print(
        f"\n[EA Test 1, {tag}] after {T_END / 3600:.0f} h:"
        f"\n  pond 1 (connected):    mean depth {d1:.3f} m  surface"
        f" {'dry' if s1 is None else f'{s1 + z_ref:.3f} m'} (boundary {LEVEL_LOW:.2f})"
        f"\n  pond 2 (disconnected): mean depth {d2:.3f} m  surface"
        f" {'dry' if s2 is None else f'{s2 + z_ref:.3f} m'} (ridge crest {Z_RIDGE:.2f})"
        f"\n  peak pond-2 depth {peak_pond2:.3f} m   high ground max {high_max:.2e} m"
        f"   mass {ledger.max_rel_error:.2e}"
    )

    # Always-on gates.
    assert np.isfinite(h).all(), "NaN/inf -- wetting/drying instability"
    assert h.min() >= 0.0, f"depth went negative to {h.min():.3e}"
    assert ledger.max_rel_error < MASS_GATE, f"mass gate broke: {ledger.max_rel_error:.2e}"

    # The mechanism: the far depression *did* flood during the peak ...
    assert peak_pond2 > 0.2, f"pond 2 never flooded (peak mean depth {peak_pond2:.3f} m)"
    # ... and -- the published finding -- it is still holding water at the end, at
    # roughly the ridge crest, because that is the only way out.
    assert d2 > 0.15, f"pond 2 drained with the boundary ({d2:.3f} m) -- bathtub behaviour"
    assert s2 is not None and abs((s2 + z_ref) - Z_RIDGE) < 0.15, (
        f"pond 2 settled at {s2 + z_ref:.3f} m, not near the ridge crest {Z_RIDGE:.2f} m"
    )
    # The connected part follows the boundary back down to the low level.
    assert s1 is not None and abs((s1 + z_ref) - LEVEL_LOW) < 0.1, (
        f"pond 1 did not return to the boundary level: {s1 + z_ref:.3f} m"
    )
    # Ground above the peak level never wets.
    assert high_max < H_DRY, f"high ground wetted to {high_max:.3e} m"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
