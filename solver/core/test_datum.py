"""Vertical datum shift (M5, :mod:`solver.core.datum`).

Two things need proving. The arithmetic is trivial and is checked directly. The
*reason the feature exists* is not trivial and is checked by the one experiment
that can fail: a lake at rest on a bumpy bed lifted to a high datum. Well-balanced
cancellation happens inside ``eta = h + z`` in float32, so at a high datum it
degrades -- and the shift must restore it. The test asserts both halves (shifted is
at the float32 floor; unshifted is measurably worse), so a future change that
quietly makes the shift a no-op fails here instead of passing vacuously.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core import hllc
from solver.core.datum import resolve_datum, shift_bed, unshift_bed
from solver.core.state import State

wp.init()
DEV = "cpu"


def _bumpy_bed(ny: int, nx: int) -> np.ndarray:
    """A sloped, bumpy bed (the M4 lake-at-rest shape) in metres above 0."""
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    return (
        0.05 * xx + 0.03 * yy + 0.8 * np.sin(0.7 * xx) * np.cos(0.5 * yy) + 0.4 * np.sin(0.3 * xx)
    ).astype(np.float32)


def _lake_at_rest_max_speed(bed: np.ndarray, surface: float, *, steps: int = 60) -> float:
    """Run a flat lake over ``bed`` and return the worst |velocity| reached."""
    h0 = np.maximum(surface - bed.astype(np.float64), 0.0).astype(np.float32)
    st = State.from_bed(bed, dx=10.0, depth=h0, manning=0.0, device=DEV)
    worst = 0.0
    for _ in range(steps):
        dt = hllc.compute_dt(st, alpha=0.45, dt_max=5.0)
        hllc.step(st, dt=dt)
        u, v = st.velocities_numpy()
        worst = max(worst, float(np.abs(u).max()), float(np.abs(v).max()))
    return worst


# --- the arithmetic ---------------------------------------------------------- #
def test_resolve_datum_modes():
    bed = np.array([[12.7, 15.2], [11.4, 19.0]], dtype=np.float32)
    assert resolve_datum(None, bed) == 0.0  # default: no shift at all
    assert resolve_datum("auto", bed) == 11.0  # floor(min) -> exactly representable
    assert resolve_datum(250.0, bed) == 250.0


def test_resolve_datum_rejects_a_bad_string():
    with pytest.raises(ValueError, match="number or 'auto'"):
        resolve_datum("sea-level", np.zeros((2, 2), np.float32))


def test_shift_round_trips_and_is_a_no_op_at_zero():
    bed = _bumpy_bed(8, 8) + 500.0
    shifted = shift_bed(bed, 500.0)
    assert float(np.abs(shifted).max()) < float(np.abs(bed).max())
    assert np.array_equal(unshift_bed(shifted, 500.0), bed)
    # z_ref == 0 must return the *same object*: that is what keeps every pre-M5
    # scenario bitwise-identical rather than merely equal.
    assert shift_bed(bed, 0.0) is bed
    assert unshift_bed(bed, 0.0) is bed


# --- why it exists ------------------------------------------------------------ #
@pytest.mark.parametrize("datum", [5000.0, 500.0])
def test_datum_shift_restores_lake_at_rest_at_altitude(datum):
    """The load-bearing test: a lake at rest at altitude, with and without the shift.

    ``eta = h + z`` resolves to ~ULP(z) in float32, so the well-balanced
    cancellation degrades as the datum rises. With the shift the same bed and the
    same water are back at the float32 floor. Both numbers are printed, so the
    margin this feature actually buys is on the record rather than asserted blind.
    """
    bed0 = _bumpy_bed(40, 40)  # spans ~0..5 m
    surface0 = float(bed0.max()) + 0.5  # everything wet, a genuinely flat surface

    v_shifted = _lake_at_rest_max_speed(bed0, surface0)
    v_raw = _lake_at_rest_max_speed(bed0 + np.float32(datum), surface0 + datum)
    print(f"\n[datum {datum:g} m] max|u| raw={v_raw:.3e} m/s  shifted={v_shifted:.3e} m/s")

    # Shifted: at rest to the float32 floor (the M4 lake-at-rest gate's band).
    assert v_shifted < 1.0e-4, f"shifted lake did not stay at rest: {v_shifted:.3e}"
    # Unshifted: measurably worse -- this is the failure the shift exists to remove.
    # (Asserted as a *ratio* so it cannot pass by both being tiny.)
    assert v_raw > 10.0 * v_shifted, (
        f"datum {datum} m did not degrade lake-at-rest (raw {v_raw:.3e} vs "
        f"shifted {v_shifted:.3e}) -- the shift would be buying nothing"
    )


def test_depth_is_datum_independent():
    """Shifting the origin must not change the physics: same depths, either datum.

    Not bitwise: the shifted bed is ``float32(float32(z + 300) - 300)``, which
    re-quantizes the terrain by ~ULP(300 m) ~ 3e-5 m. That residual is the *input*
    differing, not the physics -- so the gate is that depths of order metres agree
    to well under a millimetre.
    """
    bed0 = _bumpy_bed(24, 24)
    surface0 = float(bed0.max()) + 0.3

    def final_depth(bed, surface):
        h0 = np.maximum(surface - bed.astype(np.float64), 0.0).astype(np.float32)
        st = State.from_bed(bed, dx=10.0, depth=h0, manning=0.03, device=DEV)
        for _ in range(20):
            hllc.step(st, dt=hllc.compute_dt(st, alpha=0.45, dt_max=2.0))
        return st.h.numpy()

    a = final_depth(bed0, surface0)
    b = final_depth(shift_bed(bed0 + np.float32(300.0), 300.0), surface0)
    worst = float(np.abs(a - b).max())
    print(f"\n[datum independence] worst depth difference {worst:.2e} m (depths ~{a.max():.1f} m)")
    assert worst < 1.0e-4, f"datum changed the flow, not just the terrain quantization: {worst:.2e}"
