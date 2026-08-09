"""The morphology slow process (M7 build step 5) on Warp's CPU backend.

The gates -- sediment mass conservation, interval independence, the bed-wave
celerity, the morphological-CFL assertion -- are build step 8 and live in
``validation/``. What is checked here is the **wiring**: that the fast half reads
the right inputs at the right point in the step, that an activation moves the bed
exactly the way the kernels were designed to, that the bounds refuse rather than
clamp, and that none of it touches the water.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core.channels import arm_channels
from solver.core.local_inertial import compute_dt, step
from solver.core.sediment import SedimentError, accumulate_transport, arm_sediment
from solver.core.state import State
from solver.processes.morphology import MorphologyProcess, bed_change_bounds

wp.init()
DEV = "cpu"

NY, NX, DX = 6, 24, 10.0
SLOPE = 0.004
D50 = 0.008
POROSITY = 0.4


def _bed(ny: int = NY, nx: int = NX) -> np.ndarray:
    """A plane falling in +x, so the flow is axis-aligned and the transport is too."""
    j = np.arange(nx, dtype=np.float32)
    return np.tile(20.0 + (nx - 1 - j) * DX * SLOPE, (ny, 1)).astype(np.float32)


def _state(
    *,
    depth: float = 0.4,
    channels: bool = False,
    rain: bool = False,
    infil: bool = False,
    open_east: bool = False,
) -> State:
    """A tilted reach with any combination of the optional physics armed.

    The combination matters for the equivalence test below: rain, infiltration and
    the open-boundary sink are exactly the kernels that run *after* the transport
    hook inside a step, so they are the ones that could invalidate it.
    """
    st = State.from_bed(_bed(), dx=DX, depth=depth, manning=0.035, device=DEV)
    if channels:
        w = np.zeros((NY, NX), np.float32)
        d = np.zeros((NY, NX), np.float32)
        w[NY // 2, :] = 4.0
        d[NY // 2, :] = 1.0
        arm_channels(st, w, d)
    if infil:
        st.set_infiltration(np.full((NY, NX), 2.0e-6, np.float32))
    if open_east:
        st.set_open_boundaries(
            {"east": "open", "west": "closed", "north": "closed", "south": "closed"}
        )
    if rain:
        st.arm_source_compensation()
    return st


def _run(st: State, steps: int, *, rain: float = 0.0, hook=None) -> list[float]:
    """Step ``st`` forward, optionally calling ``hook(dt)`` after each step."""
    dts = []
    for _ in range(steps):
        dt = compute_dt(st, alpha=0.7, dt_max=2.0)
        step(st, dt=dt, rain=rain)
        if hook is not None:
            hook(dt)
        dts.append(dt)
    return dts


# --- the assertion build step 3 asked for ------------------------------------


@pytest.mark.parametrize("channels", [False, True])
def test_the_in_step_hook_is_the_same_inputs_as_accumulating_after_the_step(channels):
    """Accumulating inside ``step`` == accumulating after it returns, **bit for bit**.

    This is the one assertion build step 5 owes build step 3
    (:mod:`validation.test_bed_wave`), and it is what makes that fixture's recorded
    numbers -- 0.993 c_b by cross-correlation, and every figure derived with them --
    transfer to the real process unchanged.

    The claim is specific: everything the transport kernels read (``qx``, ``qy``,
    ``eta``, ``z``) is final once the limiter has run, and the tail of the step
    (continuity, the compensated areal sources, the infiltration sink, the
    open-boundary sink) writes only ``h``, ``h_comp`` and ``loss_cum``. So all four
    of those are armed here: they are precisely the kernels that could falsify it,
    and a future reordering that made one of them touch a face or rewrite ``eta``
    would move the gate silently rather than fail.

    The second half of the assertion is that morphology does not perturb the water:
    the depth field must come out bitwise identical to the run that never armed it.
    """
    kw = dict(channels=channels, rain=True, infil=True, open_east=True)
    steps, rain = 12, 3.0e-5

    hooked = _state(**kw)
    arm_sediment(hooked, D50, POROSITY)
    _run(hooked, steps, rain=rain)

    # The build-step-3 pattern: allocate the accumulators, then hide them from `step`
    # so nothing is launched inside it, and hand-launch after the step returns.
    after = _state(**kw)
    sed_after = arm_sediment(after, D50, POROSITY)
    after.sediment = None

    def post_step(dt: float) -> None:
        after.sediment = sed_after
        accumulate_transport(after, dt)
        after.sediment = None

    _run(after, steps, rain=rain, hook=post_step)

    plain = _state(**kw)  # never armed: the water reference
    _run(plain, steps, rain=rain)

    sed_hooked = hooked.sediment
    assert float(np.abs(sed_hooked.qs_int_x.numpy()).max()) > 0.0, "nothing was transported"
    for name in ("qs_int_x", "qs_int_y", "qs_comp_x", "qs_comp_y"):
        a = getattr(sed_hooked, name).numpy()
        b = getattr(sed_after, name).numpy()
        assert (a == b).all(), f"{name} differs between the in-step and post-step hooks"
    # Morphology reads the flow; it must not change it.
    assert (hooked.h.numpy() == plain.h.numpy()).all()
    assert (hooked.qx.numpy() == plain.qx.numpy()).all()


def test_an_unarmed_run_launches_no_transport_kernel_and_keeps_its_bed():
    """The bitwise-unchanged invariant, structurally: unarmed means nothing allocated."""
    st = _state()
    assert st.sediment is None
    z0 = st.z.numpy().copy()
    _run(st, 8)
    assert (st.z.numpy() == z0).all()
    with pytest.raises(SedimentError, match="arm_sediment"):
        accumulate_transport(st, 1.0)


# --- an activation -----------------------------------------------------------


def _armed(**kw) -> tuple[State, MorphologyProcess]:
    st = _state(**kw)
    arm_sediment(st, D50, POROSITY)
    return st, MorphologyProcess(st, interval_s=60.0)


def test_an_activation_rebuilds_the_bed_from_z0_and_clears_the_integral():
    """``z = float32(z0 + dz_cum)``, exactly -- never ``z += dz``, and the tank empties."""
    st, morph = _armed()
    sed = st.sediment
    z0 = sed.z0.numpy().copy()
    dts = _run(st, 30)

    rec = morph.advance(sum(dts), sum(dts))
    dz = sed.dz_cum.numpy()

    assert np.abs(dz).max() > 0.0, "the reach should have moved some bed"
    # Recomputed from the pristine bed, not accumulated into a float32 field.
    assert (st.z.numpy() == (z0 + dz).astype(np.float32)).all()
    # The integral is consumed by the activation it feeds; the compensation debt is
    # deliberately kept (solver.core.sediment.clear_transport_integral).
    assert (sed.qs_int_x.numpy() == 0.0).all()
    assert (sed.qs_int_y.numpy() == 0.0).all()
    # First activation: everything applied so far *is* this activation's work.
    assert rec.applied_m3 == pytest.approx(rec.cumulative_m3)
    assert rec.cumulative_m3 == pytest.approx(
        float(dz.sum()) * st.grid.cell_area * (1.0 - POROSITY)
    )
    assert rec.banked_m3 == 0.0  # nothing is bounded here, so nothing was refused
    assert rec.celerity_m_s > 0.0 and rec.courant == pytest.approx(
        rec.celerity_m_s * rec.interval_s / DX
    )
    assert morph.peak_courant == rec.courant
    assert morph.series[-1]["cumulative_m3"] == rec.cumulative_m3


def test_a_bed_update_does_not_touch_the_water_but_does_refresh_eta():
    """``h`` is volume per unit plan area: the bed rises *through* it (M7 plan §1.6).

    Water volume is conserved by construction, which is exactly why "the bed update
    quietly ate water" has to be checked rather than assumed. ``eta`` is a function
    of the bed, so it moves with it -- and it must, or a reader between two ticks
    sees a surface that disagrees with the state's own ``z``.
    """
    st, morph = _armed()
    _run(st, 30)
    h_before = st.h.numpy().copy()
    morph.advance(100.0, 100.0)

    assert (st.h.numpy() == h_before).all()
    assert (st.eta.numpy() == (h_before + st.z.numpy()).astype(np.float32)).all()


def test_still_water_moves_no_bed_at_all():
    """No motion => theta < theta_c => a **bit-exact** zero bed change.

    Cheap, and it is what proves no velocity-independent term was wired in: a lake
    at rest on a flat bed has zero shear everywhere, so every face's capacity is the
    exact zero :func:`~solver.core.sediment.mpm_capacity` returns below threshold.
    """
    flat = np.full((NY, NX), 12.0, np.float32)
    st = State.from_bed(flat, dx=DX, depth=0.75, manning=0.035, device=DEV)
    arm_sediment(st, D50, POROSITY)
    morph = MorphologyProcess(st, interval_s=60.0)
    z0 = st.z.numpy().copy()

    _run(st, 20)
    rec = morph.advance(60.0, 60.0)

    assert (st.sediment.dz_cum.numpy() == 0.0).all()
    assert (st.z.numpy() == z0).all()
    assert rec.applied_m3 == 0.0
    assert rec.celerity_m_s == 0.0


# --- the bounds --------------------------------------------------------------


def test_bed_change_bounds_are_unbounded_until_something_asks_for_a_limit():
    lo, hi = bed_change_bounds((3, 4))
    assert (lo == -np.inf).all() and (hi == np.inf).all()

    lo, hi = bed_change_bounds((3, 4), alluvium_thickness=0.25)
    assert (lo == -0.25).all() and (hi == np.inf).all()  # deposition stays free

    thick = np.full((3, 4), 0.5, np.float32)
    thick[1, 1] = 0.0
    lo, hi = bed_change_bounds((3, 4), alluvium_thickness=thick, frozen_cells=[(0, 0)])
    assert lo[1, 1] == 0.0 and hi[1, 1] == np.inf  # bedrock: erosion barred, fill free
    assert lo[0, 0] == 0.0 and hi[0, 0] == 0.0  # frozen: neither direction

    with pytest.raises(SedimentError, match="outside"):
        bed_change_bounds((3, 4), frozen_cells=[(9, 9)])
    with pytest.raises(SedimentError, match=">= 0"):
        bed_change_bounds((3, 4), alluvium_thickness=-1.0)
    with pytest.raises(SedimentError, match="shape"):
        bed_change_bounds((3, 4), alluvium_thickness=np.zeros((2, 2), np.float32))


def test_a_frozen_cell_holds_its_bed_and_banks_exactly_what_it_refused():
    """A dam is engineered, not alluvial -- and the refusal is accounted, not clamped.

    Measured against the *same* transport integral rather than asserted: one
    activation, run twice from identical states, so the unbounded run's bed change at
    the frozen cell is exactly the metres the bounded run must bank. (Over more than
    one activation the two runs diverge, because the bed each is stepping on differs.)
    """
    free_st, free = _armed()
    dts = _run(free_st, 30)
    free.advance(sum(dts), sum(dts))
    want = float(free_st.sediment.dz_cum.numpy()[NY // 2, NX // 2])
    assert want != 0.0, "pick a cell whose bed actually moves"

    st = _state()
    arm_sediment(st, D50, POROSITY)
    lo, hi = bed_change_bounds(st.grid.shape, frozen_cells=[(NY // 2, NX // 2)])
    morph = MorphologyProcess(st, interval_s=60.0, dz_lo=lo, dz_hi=hi)
    assert morph.frozen_cells == 1
    _run(st, 30)
    rec = morph.advance(sum(dts), sum(dts))

    sed = st.sediment
    assert sed.dz_cum.numpy()[NY // 2, NX // 2] == 0.0
    assert float(sed.dz_unapplied.numpy()[NY // 2, NX // 2]) == pytest.approx(want, rel=1e-9)
    assert rec.banked_m3 == pytest.approx(want * st.grid.cell_area * (1.0 - POROSITY), rel=1e-9)


def test_an_alluvium_floor_stops_the_scour_and_banks_the_rest():
    """A floor limits how far the bed may erode; deposition can lift it back off."""
    st = _state()
    arm_sediment(st, D50, POROSITY)
    lo, hi = bed_change_bounds(st.grid.shape, alluvium_thickness=1.0e-4)
    morph = MorphologyProcess(st, interval_s=60.0, dz_lo=lo, dz_hi=hi)
    assert morph.floored_cells == NY * NX

    dts = _run(st, 40)
    rec = morph.advance(sum(dts), sum(dts))
    dz = st.sediment.dz_cum.numpy()

    assert dz.min() >= -1.0e-4 - 1e-12, "the floor must hold"
    assert dz.min() == pytest.approx(-1.0e-4), "and something must actually reach it"
    assert rec.banked_m3 < 0.0, "the refused part is scour: a negative solid volume"


def test_the_process_refuses_what_it_cannot_honour():
    st = _state()
    with pytest.raises(SedimentError, match="arm_sediment"):
        MorphologyProcess(st, interval_s=60.0)
    arm_sediment(st, D50, POROSITY)
    with pytest.raises(SedimentError, match="interval_s"):
        MorphologyProcess(st, interval_s=0.0)
    lo, hi = bed_change_bounds(st.grid.shape)
    with pytest.raises(SedimentError, match="both"):
        MorphologyProcess(st, interval_s=60.0, dz_lo=lo)
    with pytest.raises(SedimentError, match="grid"):
        MorphologyProcess(st, interval_s=60.0, dz_lo=lo[:2], dz_hi=hi[:2])
    with pytest.raises(SedimentError, match="cross"):
        MorphologyProcess(st, interval_s=60.0, dz_lo=hi, dz_hi=lo)


def test_it_advances_on_the_scheduler_like_any_other_slow_process():
    st, morph = _armed()
    proc = morph.as_slow_process()
    assert proc.name == "morphology"
    assert proc.interval == 60.0
    assert proc.activations(200.0) == [60.0, 120.0, 180.0]
    _run(st, 10)
    rec = proc.advance(60.0, 60.0)
    assert rec is morph.records[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
