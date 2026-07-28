"""Sub-grid channel validation (M6 plan §3) on Warp's CPU backend.

Three gates, in order of what they prove:

1. **Normal depth** -- a steady discharge in a sub-grid channel settles at the
   Manning normal depth for its *own* section (``Q = A·R^{2/3}·S^{1/2}/n`` with
   ``R = A/P``, not the wide-channel ``R ≈ h``). This is analytic, and it is the
   check that the two-component face update is a hydraulics model rather than a
   plausible-looking interpolation.
2. **Overbank spill** -- past bank full the water leaves the channel and spreads
   over the floodplain, and the closed-domain mass balance holds through the
   transition. The storage curve is a diagnostic map, so conservation should be
   exact by construction; this is the test that says so out loud.
3. **Fine-vs-coarse equivalence** -- *the reach claim*. The same reach run twice:
   once with the channel **resolved** at 10 m cells, once with it carried
   **sub-grid** in 100 m cells (a 10x coarsening, 100x fewer cells). If the coarse
   run reproduces the fine run's depth and storage, coarsening + sub-grid channels
   buys area without losing the river.

**Honesty.** (3) is an equivalence claim about our own two runs, not a benchmark
validation, and the two models are not identical by construction: the resolved 2D
channel has no side-wall drag (LI's per-unit-width friction uses ``R ≈ h``), while
the sub-grid channel includes the wetted perimeter. That is *physics we deliberately
kept*, and for the 40 m x ~1.2 m section here it is worth ~2% in depth between the
two analytical sections. The gate is therefore set wide enough to admit that term
plus discretisation (12%), and the test **prints** both modelled depths and both
analytical sections rather than hiding the difference inside a tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from solver.core.channels import arm_channels
from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State
from solver.io.config import Inflow
from solver.processes.inflow import InflowInjector

wp.init()
DEV = "cpu"

_EAST_OPEN = {"east": "open", "west": "closed", "north": "closed", "south": "closed"}


def normal_depth(discharge: float, width: float, slope: float, manning: float) -> float:
    """Manning normal depth of a rectangular channel (bisection on ``Q(h)``).

    ``Q(h) = (1/n)·A·R^(2/3)·sqrt(S)`` with ``A = w·h`` and ``R = A/P = w·h/(w+2h)``
    -- the same section the sub-grid channel model uses, so this is a genuine
    analytical reference for it and not a restatement of the code.
    """

    def q_of(h: float) -> float:
        area = width * h
        radius = area / (width + 2.0 * h)
        return area * radius ** (2.0 / 3.0) * math.sqrt(slope) / manning

    lo, hi = 1e-6, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if q_of(mid) < discharge:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sloped_bed(ny: int, nx: int, dx: float, slope: float, base: float = 5.0) -> np.ndarray:
    """Bed falling in +x (west high, east low) -- flow runs toward the open edge."""
    xx = np.broadcast_to(np.arange(nx), (ny, nx))
    return ((nx - 1 - xx) * dx * slope + base).astype(np.float32)


def _run_steady(
    st: State,
    inflows: list[Inflow],
    *,
    steps: int,
    dt_max: float,
) -> tuple[MassLedger, float]:
    """Drive a state to steady state under constant inflow; return ledger + time."""
    inj = InflowInjector(inflows, st.grid, DEV)
    ledger = MassLedger.from_state(st)
    t = 0.0
    for _ in range(steps):
        dt = compute_dt(st, alpha=0.7, dt_max=dt_max)
        ledger.add_inflow(inj.apply(st, t, dt))
        step(st, dt=dt)
        t += dt
    ledger.record(st, t)
    return ledger, t


def test_subgrid_channel_reaches_manning_normal_depth():
    """A 20 m channel inside 50 m cells conveys its own section, not the cell's."""
    nx, dx = 100, 50.0
    slope, n_ch, width, bankfull, discharge = 0.001, 0.03, 20.0, 4.0, 30.0
    bed = _sloped_bed(1, nx, dx, slope)

    st = State.from_bed(bed, dx=dx, depth=0.0, manning=0.05, device=DEV)  # rough floodplain
    arm_channels(
        st,
        np.full((1, nx), width, np.float32),
        np.full((1, nx), bankfull, np.float32),
        np.full((1, nx), n_ch, np.float32),
    )
    st.set_open_boundaries(_EAST_OPEN)

    ledger, _ = _run_steady(
        st,
        [Inflow(cell=(0, 0), hydrograph=[(0.0, discharge), (1.0e9, discharge)])],
        steps=3000,
        dt_max=10.0,
    )

    h_n = normal_depth(discharge, width, slope, n_ch)
    # Cell-mean depth -> channel column depth (in-bank: column = h·dx/w).
    column = st.h.numpy()[0] * dx / width
    interior = column[10 : nx - 10]  # skip the injection cell and the toe drawdown
    median = float(np.median(interior))

    print(
        f"\n[subgrid] normal={h_n:.3f} m  modelled={median:.3f} m  "
        f"({100 * (median - h_n) / h_n:+.1f}%)  mass={ledger.max_rel_error:.2e}"
    )
    assert np.isfinite(column).all()
    assert ledger.max_rel_error < MASS_GATE
    assert median < bankfull, "the test is only meaningful while the flow is in-bank"
    assert median == pytest.approx(h_n, rel=0.05), (
        f"sub-grid channel depth {median:.3f} m is not the normal depth {h_n:.3f} m"
    )


def test_overbank_spill_leaves_the_channel_and_conserves_mass():
    """Past bank full the water spreads onto the floodplain; the ledger holds."""
    ny, nx, dx = 7, 30, 50.0
    bed = _sloped_bed(ny, nx, dx, 0.0005)
    st = State.from_bed(bed, dx=dx, depth=0.0, manning=0.05, device=DEV)
    row = ny // 2
    w = np.zeros((ny, nx), np.float32)
    d = np.zeros((ny, nx), np.float32)
    w[row, :] = 20.0
    d[row, :] = 1.0  # shallow banks -> easy to overtop
    arm_channels(st, w, d, np.full((ny, nx), 0.03, np.float32))

    # Closed box: the only way mass can change is the (exact) inflow accounting.
    ledger, _ = _run_steady(
        st,
        [Inflow(cell=(row, 2), hydrograph=[(0.0, 60.0), (1.0e9, 60.0)])],
        steps=1200,
        dt_max=5.0,
    )

    h = st.h.numpy()
    h_bf = w[row, 0] * d[row, 0] / dx  # cell-mean depth at bank full
    channel_row = h[row]
    floodplain = np.delete(h, row, axis=0)

    print(
        f"\n[overbank] channel mean={channel_row.mean():.3f} m (bank full {h_bf:.3f} m)  "
        f"floodplain wet cells={int((floodplain > 1e-3).sum())}  "
        f"mass={ledger.max_rel_error:.2e}"
    )
    assert np.isfinite(h).all()
    assert h.min() >= -1e-9
    assert channel_row.max() > h_bf, "the channel never reached bank full"
    assert (floodplain > 1e-3).sum() > 0, "water never left the channel"
    assert ledger.max_rel_error < MASS_GATE


def test_coarse_subgrid_channel_matches_the_resolved_channel():
    """The reach claim: 100x fewer cells, same river.

    Fine: 10 m cells, the 40 m channel resolved as four cells cut 2 m into the
    floodplain. Coarse: 100 m cells, one row, the same channel carried sub-grid.
    Same length, slope, roughness and steady discharge; compare the channel depth
    and the water stored in the reach.
    """
    length_m, slope, n_ch, width, bankfull, discharge = 2000.0, 0.0008, 0.035, 40.0, 2.0, 40.0

    # --- fine: the channel is resolved ---------------------------------------
    dxf = 10.0
    nyf, nxf = 10, int(length_m / dxf)
    bed_f = _sloped_bed(nyf, nxf, dxf, slope)
    chan_rows = slice(3, 7)  # four 10 m cells = the 40 m channel
    bed_f[chan_rows, :] -= bankfull
    fine = State.from_bed(bed_f, dx=dxf, depth=0.0, manning=n_ch, device=DEV)
    fine.set_open_boundaries(_EAST_OPEN)
    per_cell = discharge / 4.0
    fine_ledger, _ = _run_steady(
        fine,
        [Inflow(cell=(r, 0), hydrograph=[(0.0, per_cell), (1.0e9, per_cell)]) for r in range(3, 7)],
        steps=6000,
        dt_max=5.0,
    )

    # --- coarse: the channel is sub-grid --------------------------------------
    dxc = 100.0
    nxc = int(length_m / dxc)
    bed_c = _sloped_bed(1, nxc, dxc, slope)
    coarse = State.from_bed(bed_c, dx=dxc, depth=0.0, manning=n_ch, device=DEV)
    arm_channels(
        coarse,
        np.full((1, nxc), width, np.float32),
        np.full((1, nxc), bankfull, np.float32),
        np.full((1, nxc), n_ch, np.float32),
    )
    coarse.set_open_boundaries(_EAST_OPEN)
    coarse_ledger, _ = _run_steady(
        coarse,
        [Inflow(cell=(0, 0), hydrograph=[(0.0, discharge), (1.0e9, discharge)])],
        steps=2000,
        dt_max=10.0,
    )

    # --- compare --------------------------------------------------------------
    # Interior 60% of the reach, away from the head injection and the toe drawdown.
    def _interior(a: np.ndarray) -> np.ndarray:
        k = len(a)
        return a[k // 5 : -k // 5]

    fine_depth = float(np.median(_interior(fine.h.numpy()[chan_rows].mean(axis=0))))
    coarse_depth = float(np.median(_interior(coarse.h.numpy()[0] * dxc / width)))
    # Water stored in the reach (m^3) -- the storage half of the claim.
    fine_volume = float(fine.h.numpy().astype(np.float64).sum()) * dxf * dxf
    coarse_volume = float(coarse.h.numpy().astype(np.float64).sum()) * dxc * dxc

    h_wide = (discharge / width * n_ch / math.sqrt(slope)) ** 0.6  # R ~ h (the fine model)
    h_section = normal_depth(discharge, width, slope, n_ch)  # R = A/P (the sub-grid model)

    print(
        f"\n[reach] resolved={fine_depth:.3f} m ({nyf * nxf} cells)  "
        f"sub-grid={coarse_depth:.3f} m ({nxc} cells)  "
        f"diff={100 * (coarse_depth - fine_depth) / fine_depth:+.1f}%\n"
        f"        analytic: wide-channel {h_wide:.3f} m, full-section {h_section:.3f} m "
        f"({100 * (h_section - h_wide) / h_wide:+.1f}% -- the wetted-perimeter term)\n"
        f"        storage: resolved={fine_volume:,.0f} m^3  sub-grid={coarse_volume:,.0f} m^3  "
        f"mass {fine_ledger.max_rel_error:.1e} / {coarse_ledger.max_rel_error:.1e}"
    )

    assert fine_ledger.max_rel_error < MASS_GATE
    assert coarse_ledger.max_rel_error < MASS_GATE
    # Each run sits near its own analytical section -- within 10%, which is what a
    # 2 km reach with a drawdown at the open toe allows. The two sections differ by
    # the wetted-perimeter term, exactly the physics the sub-grid model adds back.
    assert fine_depth == pytest.approx(h_wide, rel=0.10)
    assert coarse_depth == pytest.approx(h_section, rel=0.10)
    # ... and therefore on each other, to within that same term plus discretisation.
    assert coarse_depth == pytest.approx(fine_depth, rel=0.12)
    assert coarse_volume == pytest.approx(fine_volume, rel=0.20)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
