"""Compensated areal-source accumulation (solver.core.sources).

The gates here are deliberately **ratios**, not thresholds. A threshold test
("drift < x") passes just as happily if the compensation array is never written --
if the Kahan terms were reassociated away by a future compiler or a fast-math
default flip, or if the arming was quietly dropped. Comparing compensated against
uncompensated under identical conditions can only pass when the compensation is
doing something, and :func:`test_compensation_term_is_actually_written` asserts the
mechanism directly rather than its effect.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core import sources
from solver.core.local_inertial import step
from solver.core.massbalance import MassLedger
from solver.core.schemes import get_scheme
from solver.core.state import State

wp.init()


def _flat_state(*, depth: float, n_cells: int = 24, dx: float = 100.0) -> State:
    """A flat closed box at rest: uniform bed, uniform depth, no flow anywhere.

    Uniform `eta` over a flat bed means every face discharge stays zero, so `h`
    changes only by the source term -- this isolates source accumulation from the
    flux divergence, which is not what this module touches.
    """
    bed = np.zeros((n_cells, n_cells), dtype=np.float32)
    return State.from_bed(bed, dx, depth=depth, manning=0.03, device="cpu")


def _accumulate(state: State, *, rain: float, dt: float, steps: int) -> float:
    """Rain on the box for `steps` steps; return the mean depth reached (m)."""
    for _ in range(steps):
        step(state, dt=dt, rain=rain, limit=False)
    return float(state.h.numpy().astype(np.float64).mean())


def test_sub_ulp_rain_is_lost_without_compensation_and_kept_with_it():
    """The sharp case: an increment below half an ulp of `h` vanishes entirely.

    At `h = 1 m` float32 has `eps = 1.19e-7`, so an increment of `4e-8` m rounds to
    nothing -- uncompensated, the depth never moves no matter how many steps run,
    and *all* of the rain is lost. This is the same mechanism that costs the reach
    demo 670 m^3, just turned up far enough to be unambiguous on a 24x24 CPU grid.
    """
    rain, dt, steps = 4.0e-8, 1.0, 2000
    expected = rain * dt * steps  # 8e-5 m

    plain = _flat_state(depth=1.0)
    got_plain = _accumulate(plain, rain=rain, dt=dt, steps=steps) - 1.0

    comp = _flat_state(depth=1.0)
    comp.arm_source_compensation()
    got_comp = _accumulate(comp, rain=rain, dt=dt, steps=steps) - 1.0

    # Uncompensated loses essentially all of it; compensated lands on the analytic
    # answer to a few ulps of the *depth*, not of the increment.
    assert got_plain == pytest.approx(0.0, abs=1e-9)
    assert got_comp == pytest.approx(expected, rel=2e-3)

    err_plain = abs(got_plain - expected)
    err_comp = abs(got_comp - expected)
    assert err_plain / max(err_comp, 1e-30) > 100.0


def test_compensation_term_is_actually_written():
    """Fast-math canary: Kahan is destroyed by `(t - h0) - y` being reassociated.

    A compiler entitled to reassociate float arithmetic folds that expression to
    zero, and every other assertion in this file would then be measuring an
    ordinary uncompensated add against itself. Assert the residue directly: after a
    rain step that demonstrably loses bits, some cell must carry a nonzero
    compensation.
    """
    st = _flat_state(depth=1.0)
    st.arm_source_compensation()
    step(st, dt=1.0, rain=4.0e-8, limit=False)
    comp = st.h_comp.numpy()
    assert np.any(comp != 0.0), "compensation term is identically zero -- Kahan was optimized away"


def test_realistic_storm_drift_is_reduced_by_orders_of_magnitude():
    """The reach-demo regime, shrunk: a thin sheet growing under a steady storm.

    15 mm/hr onto a sheet that starts at 0.3 m, at the demo's 20 s steps. The
    increment (8.3e-5 m) is a few hundred ulps of `h`, so each add loses a little
    rather than everything -- the realistic version of the case above.
    """
    rain, dt, steps = 15.0 / 1000.0 / 3600.0, 20.0, 1500
    expected = rain * dt * steps

    plain = _flat_state(depth=0.3)
    err_plain = abs((_accumulate(plain, rain=rain, dt=dt, steps=steps) - 0.3) - expected)

    comp = _flat_state(depth=0.3)
    comp.arm_source_compensation()
    err_comp = abs((_accumulate(comp, rain=rain, dt=dt, steps=steps) - 0.3) - expected)

    assert err_plain / max(err_comp, 1e-30) > 20.0


def test_mass_ledger_residual_improves_under_compensation():
    """The gate quantity itself: the ledger's relative residual, not raw depth.

    The ledger banks rain analytically in float64 (`add_rain_step`) and reads the
    stored volume back from the float32 field, so its residual *is* the
    accumulation error this module removes.
    """
    rain, dt, steps = 15.0 / 1000.0 / 3600.0, 20.0, 1500

    def run(compensate: bool) -> float:
        st = _flat_state(depth=0.3)
        if compensate:
            st.arm_source_compensation()
        ledger = MassLedger.from_state(st)
        for k in range(steps):
            step(st, dt=dt, rain=rain, limit=False)
            ledger.add_rain_step(rain, dt, st.grid.n_cells)
            if (k + 1) % 300 == 0:
                ledger.record(st, (k + 1) * dt)
        return ledger.max_rel_error

    assert run(False) / max(run(True), 1e-30) > 20.0


@pytest.mark.parametrize("scheme_name", ["local_inertial", "hllc_fv"])
def test_spatial_rain_field_is_compensated_on_both_schemes(scheme_name):
    """The field-rain branch, on both schemes -- the paths a scenario reaches.

    Nothing else in the suite exercises these. Every other test here drives the
    *uniform* source, and no shipped scenario sets ``rainfall.type = "field"``
    (``spatial_fields.toml`` has field Manning and infiltration but uniform rain),
    so without this the compensated field kernel would first execute in production.
    Both schemes are covered because each owns its own dispatch: the local-inertial
    scheme fuses uniform rain into continuity and routes the field separately, while
    HLLC keeps both as standalone kernels.

    The field carries a **uniform value** on purpose. A spatially varying rain field
    tilts `eta` on a flat bed and the water starts to flow -- and at `h = 1 m` the
    flux divergence's own float32 round-off is far larger than a sub-ulp source
    increment, so the measurement stops being about source accumulation at all
    (measured: it swamps the signal by ~9%). Keeping the box at rest isolates what
    this module owns. That the kernel really reads the field per cell is a separate,
    non-precision claim -- see :func:`test_rain_field_is_read_per_cell`.
    """
    scheme = get_scheme(scheme_name)
    dt, steps = 1.0, 1500
    # Half an ulp of h = 1 m is 5.96e-8, so this per-step increment rounds down to
    # nothing uncompensated. It has to stay under that bound: a rate *above* it
    # rounds a full ulp up every step and the cell over-accumulates instead.
    rate = 4.0e-8
    expected = rate * dt * steps

    def run(compensate: bool) -> float:
        st = _flat_state(depth=1.0)
        st.set_rain_field(np.full(st.grid.shape, rate, dtype=np.float32))
        if compensate:
            st.arm_source_compensation()
        for _ in range(steps):
            scheme.step(st, dt=dt, rain_scale=1.0)
        return float(st.h.numpy().astype(np.float64).mean()) - 1.0

    got_plain, got_comp = run(False), run(True)
    assert got_plain == pytest.approx(0.0, abs=1e-9)  # sub-ulp: all of it lost
    assert got_comp == pytest.approx(expected, rel=5e-3)

    err_plain = abs(got_plain - expected)
    err_comp = abs(got_comp - expected)
    assert err_plain / max(err_comp, 1e-30) > 100.0


@pytest.mark.parametrize("scheme_name", ["local_inertial", "hllc_fv"])
def test_rain_field_is_read_per_cell(scheme_name):
    """The compensated field kernel indexes `rain[i,j]`, not a domain-wide scalar.

    One step, rain over the left half only. At a sub-ulp rate `h` itself does not
    move, so the evidence is in the compensation term -- which is exactly where the
    debt should be: nonzero under the rain, still zero everywhere else.
    """
    scheme = get_scheme(scheme_name)
    st = _flat_state(depth=1.0)
    field = np.zeros(st.grid.shape, dtype=np.float32)
    field[:, : st.grid.shape[1] // 2] = 4.0e-8
    st.set_rain_field(field)
    st.arm_source_compensation()

    scheme.step(st, dt=1.0, rain_scale=1.0)

    comp = st.h_comp.numpy()
    half = st.grid.shape[1] // 2
    assert np.all(comp[:, :half] != 0.0), "no compensation banked under the rain"
    assert np.all(comp[:, half:] == 0.0), "compensation banked where no rain fell"


def test_unarmed_state_keeps_the_original_kernels():
    """No areal source -> no compensation array -> the pre-existing arithmetic.

    This is the property that keeps dam-break, lake-at-rest and the EA benchmarks
    bitwise unchanged: the schemes branch on `h_comp`, and nothing arms it unless
    rain actually falls.

    It is no longer the whole story for `reservoir_release`, which this docstring
    used to name. That scenario has no areal source and still takes the original
    kernels here -- but it does carry `[[inflow]]`, and point sources gained their
    own compensation on 2026-08-17 (`solver/processes/inflow.py`, a separate array
    that deliberately does *not* arm `h_comp`). So it is unchanged **on this path**
    and changed on that one; see `docs/plans/point-source-compensation.md` for its
    re-measured figures.
    """
    st = _flat_state(depth=1.0)
    assert st.h_comp is None
    step(st, dt=1.0, rain=4.0e-8, limit=False)
    assert st.h_comp is None
    with pytest.raises(ValueError, match="arm_source_compensation"):
        sources.apply_uniform_rain(st, 1.0e-6, 1.0)


def test_arming_is_idempotent_and_preserves_the_running_debt():
    """Re-arming must not wipe compensation mid-run (it is carried state)."""
    st = _flat_state(depth=1.0)
    st.arm_source_compensation()
    step(st, dt=1.0, rain=4.0e-8, limit=False)
    before = st.h_comp.numpy().copy()
    st.arm_source_compensation()
    assert np.array_equal(st.h_comp.numpy(), before)
