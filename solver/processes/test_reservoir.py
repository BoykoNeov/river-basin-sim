"""Reservoir operations on the slow clock (M5) -- the scheduler's first real load.

Four claims, each able to fail on its own:

1. **A barrier is geometry.** Raising the bed to the crest impounds water and
   overtops when the pool exceeds it -- no new momentum term, so nothing about the
   validated scheme changes.
2. **The transfer is mass-exact.** A release moves water *within* the domain, so the
   global ledger must not move: pool loss == outlet gain, to the bit, with the
   float32 rounding banked rather than lost.
3. **The rules behave.** ``fixed`` is open-loop; ``target_stage`` draws the pool
   down toward its target and then eases off.
4. **The slow clock is real.** Releases land at exact multiples of ``interval_s``
   however the fast scheme sub-cycles, and a run with a release rule is *not* the
   same as one without (a rule that silently did nothing would otherwise pass).

**On the test geometry.** The reservoir here holds metres of water, not a
centimetre-thin sheet. That is deliberate: with ``h`` ~ 1 cm on a bed at ~1 m,
``eta = h + z`` in float32 loses enough of ``h`` that a few hundred steps of
accumulated round-off pushes the closed-domain mass residual past the 1e-6 gate --
a float32 conditioning artefact (the same one :mod:`solver.core.datum` exists for),
not a defect in anything under test here. A reservoir-scaled pool keeps the gate
measuring what it is meant to measure.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
import xarray as xr

from solver.core import hllc
from solver.core.massbalance import MASS_GATE
from solver.core.state import State
from solver.io.config import Scenario, Structure
from solver.processes.reservoir import ReservoirOperator, apply_barriers, build_operators
from solver.run import run_simulation

wp.init()
DEV = "cpu"

NY, NX = 12, 8
DX = 10.0
DAM_ROW = 6
CREST = 5.0
POOL = (0, 0, 5, NX - 1)  # inclusive box behind the dam
OUTLET = (9, 4)  # downstream of the dam line


def _basin() -> np.ndarray:
    """A flat reservoir floor at 2 m (rows 0..6) with a valley falling to 0 below."""
    z = np.zeros((NY, NX), np.float32)
    z[: DAM_ROW + 1] = 2.0
    z[DAM_ROW + 1 :] = np.linspace(2.0, 0.0, NY - DAM_ROW - 1, dtype=np.float32)[:, None]
    return z


def _dam(**kw) -> Structure:
    base = dict(
        name="d",
        kind="dam",
        cells=[(DAM_ROW, j) for j in range(NX)],
        crest_m=CREST,
        release_rule="fixed",
        release_m3_s=1.0,
        pool=POOL,
        outlet=OUTLET,
        interval_s=300.0,
    )
    base.update(kw)
    return Structure(**base)


_BARRIER_ONLY = dict(release_rule="none", pool=None, outlet=None)


def _pool_depth(surface: float) -> np.ndarray:
    """Initial depth field filling the reservoir to ``surface`` (dry below the dam)."""
    bed = _basin()
    h = np.zeros((NY, NX), np.float32)
    h[:DAM_ROW] = np.maximum(surface - bed[:DAM_ROW], 0.0)
    return h


def _volume(h: np.ndarray) -> float:
    return float(np.asarray(h, dtype=np.float64).sum()) * DX * DX


# --- 1. a barrier is geometry ------------------------------------------------- #
def test_apply_barriers_raises_the_bed_and_never_lowers_it():
    bed = np.zeros((5, 5), np.float32)
    bed[2, 2] = 7.0
    out = apply_barriers(bed, [Structure(name="d", cells=[(2, 2), (2, 3)], crest_m=4.0)])
    assert out[2, 3] == 4.0  # raised to the crest
    assert out[2, 2] == 7.0  # a crest below the ground is a no-op, not a cut
    assert bed[2, 3] == 0.0  # the caller's array is untouched
    assert apply_barriers(bed, []) is bed  # no structures -> no copy, bitwise path


def _run_pool(bed: np.ndarray, h0: np.ndarray, steps: int = 300) -> np.ndarray:
    st = State.from_bed(bed, dx=DX, depth=h0, manning=0.03, device=DEV)
    for _ in range(steps):
        hllc.step(st, dt=hllc.compute_dt(st, alpha=0.45, dt_max=5.0))
    return st.h.numpy()


def test_barrier_impounds_and_the_same_water_escapes_without_it():
    """With the dam the pool is retained; take the dam away and it runs downhill.

    The paired control is the point -- "water is still upstream" alone could just
    mean nothing had time to move.
    """
    h0 = _pool_depth(4.5)  # surface 0.5 m below the crest
    kept = _run_pool(apply_barriers(_basin(), [_dam(**_BARRIER_ONLY)]), h0)
    spilt = _run_pool(_basin(), h0)

    v0 = _volume(h0)
    print(
        f"\n[barrier] upstream volume: dammed={_volume(kept[:DAM_ROW]) / v0:.1%} of V0,"
        f" undammed={_volume(spilt[:DAM_ROW]) / v0:.1%}"
    )
    assert _volume(kept[:DAM_ROW]) > 0.9 * v0, "the dam failed to impound the pool"
    assert _volume(spilt[:DAM_ROW]) < 0.5 * _volume(kept[:DAM_ROW])
    assert np.isfinite(kept).all() and kept.min() >= 0.0


def test_a_pool_above_the_crest_overtops():
    """Overtopping is ordinary solver physics, not a special case in the structure."""
    bed = apply_barriers(_basin(), [_dam(**_BARRIER_ONLY)])
    over = _run_pool(bed, _pool_depth(CREST + 0.5))  # surface above the crest
    under = _run_pool(bed, _pool_depth(CREST - 0.5))  # surface below it
    below_over = _volume(over[DAM_ROW + 1 :])
    below_under = _volume(under[DAM_ROW + 1 :])
    print(f"\n[overtop] downstream volume: over-crest={below_over:.1f} under={below_under:.1f} m3")
    assert below_over > 100.0, "a pool above the crest did not spill"
    assert below_over > 10.0 * max(below_under, 1e-9)


# --- 2. the transfer is mass-exact -------------------------------------------- #
def test_release_transfer_conserves_mass_exactly():
    """A release is internal: total volume must not move, beyond banked rounding.

    The strong form -- pool loss == released volume, and the domain total changes
    only by the float32 rounding that was *banked* -- because a leaky transfer would
    masquerade as physics.
    """
    bed = apply_barriers(_basin(), [_dam()])
    st = State.from_bed(bed, dx=DX, depth=_pool_depth(4.5), manning=0.03, device=DEV)
    op = ReservoirOperator(_dam(), st)

    r0, c0, r1, c1 = POOL
    before = st.h.numpy()
    v_before, pool_before = _volume(before), _volume(before[r0 : r1 + 1, c0 : c1 + 1])
    out_before = float(before[OUTLET]) * DX * DX

    rec = op.advance(300.0, 300.0)

    after = st.h.numpy()
    v_after, pool_after = _volume(after), _volume(after[r0 : r1 + 1, c0 : c1 + 1])
    out_after = float(after[OUTLET]) * DX * DX
    banked = st.loss_volume(st.grid.cell_area)

    print(
        f"\n[release] Q={rec.discharge_m3_s} m3/s V={rec.volume_m3:.4f} m3"
        f"  pool -{pool_before - pool_after:.6f}  outlet +{out_after - out_before:.6f}"
        f"  banked rounding {banked:.3e} m3"
    )
    assert rec.volume_m3 == pytest.approx(300.0, rel=1e-6)  # Q * dt_slow
    assert (pool_before - pool_after) == pytest.approx(rec.volume_m3, rel=1e-9)
    # The domain total changes only by the banked (i.e. accounted) rounding.
    assert (v_before - v_after) == pytest.approx(banked, abs=1e-9 * v_before)
    assert abs(v_before - v_after) < 1e-6 * v_before


def test_release_is_capped_by_what_the_pool_holds():
    """A rule that outruns its reservoir delivers less; it never invents water."""
    bed = apply_barriers(_basin(), [_dam()])
    h0 = _pool_depth(2.02)  # a shallow 2 cm film over the reservoir floor
    st = State.from_bed(bed, dx=DX, depth=h0, manning=0.03, device=DEV)
    stored = _volume(h0)

    rec = ReservoirOperator(_dam(release_m3_s=10_000.0), st).advance(300.0, 300.0)
    print(f"\n[capped] stored={stored:.3f} m3 requested=3.0e6 m3 released={rec.volume_m3:.3f} m3")
    assert rec.volume_m3 <= stored + 1e-9
    assert rec.volume_m3 == pytest.approx(stored, rel=1e-6)  # took all of it, no more
    assert float(st.h.numpy()[: DAM_ROW + 1].min()) >= 0.0


def test_release_on_a_dry_pool_is_a_no_op():
    bed = apply_barriers(_basin(), [_dam()])
    st = State.from_bed(bed, dx=DX, depth=0.0, manning=0.03, device=DEV)
    rec = ReservoirOperator(_dam(), st).advance(300.0, 300.0)
    assert rec.stage is None and rec.discharge_m3_s == 0.0 and rec.volume_m3 == 0.0


# --- 3. the rules behave ------------------------------------------------------ #
def test_target_stage_draws_the_pool_down_and_eases_off():
    """The closed-loop rule -- what makes the sync-point feedback load-bearing.

    Driven directly (no flood sub-steps in between) so the rule is isolated from
    the hydraulics; the end-to-end coupling is covered by the run tests below.
    """
    dam = _dam(release_rule="target_stage", target_stage_m=3.0, release_max_m3_s=8.0)
    bed = apply_barriers(_basin(), [dam])
    st = State.from_bed(bed, dx=DX, depth=_pool_depth(4.8), manning=0.03, device=DEV)
    op = ReservoirOperator(dam, st)

    print("\n[target_stage] stage -> Q")
    for k in range(1, 25):
        rec = op.advance(300.0 * k, 300.0)
        if k % 4 == 0:
            print(f"   {rec.stage:.4f} -> {rec.discharge_m3_s:6.3f}")
    stages = [r.stage for r in op.records]
    qs = [r.discharge_m3_s for r in op.records]

    assert stages[0] > stages[-1], "the rule never drew the pool down"
    assert qs[0] > qs[-1], "the release never eased off as the target was approached"
    assert max(qs) <= dam.release_max_m3_s + 1e-9  # the cap holds
    # Proportional control converges *toward* the target, not through it: the pool
    # settles near it and the release shuts down, rather than over-draining.
    assert stages[-1] >= dam.target_stage_m - 0.05
    assert stages[-1] < 4.8


# --- 4. the slow clock is real ------------------------------------------------ #
_RUN = dict(
    scheme="hllc_fv",
    dx=DX,
    end_time=1200.0,
    output_every=600.0,
    dt_max=5.0,
    alpha=0.45,
    rain_mm_hr=0.0,
    rain_duration=0.0,
    initial_depth=1.0,
)


def test_release_lands_on_the_slow_clock_through_a_full_run(tmp_path):
    """End to end: the scheduler fires the rule at exact multiples of interval_s,
    the store records the release series, and the run still passes the mass gate."""
    dam = _dam(interval_s=300.0, release_m3_s=2.0)
    scn = Scenario(name="reservoir_run", structures=[dam], **_RUN)
    ledger = run_simulation(scn, _basin(), tmp_path / "res.zarr", device="cpu", verbose=False)
    assert ledger.max_rel_error < MASS_GATE, f"mass gate broke: {ledger.max_rel_error:.2e}"

    ds = xr.open_zarr(tmp_path / "res.zarr", consolidated=False)
    series = ds.attrs["reservoir_releases"]["d"]
    times = [r["time"] for r in series]
    print(f"\n[slow clock] release times {times}  mass={ledger.max_rel_error:.2e}")
    assert times == pytest.approx([300.0, 600.0, 900.0, 1200.0])
    assert all(r["volume_m3"] > 0.0 for r in series), "the rule never moved any water"
    # The dam is in the bed the store records, so the viewer shows the structure.
    assert float(ds["bed"].values[DAM_ROW, 0]) == pytest.approx(CREST)


def test_a_release_rule_actually_changes_the_run(tmp_path):
    """Guard against a silently inert rule: with and without must differ in the pool."""
    off = run_simulation(
        Scenario(name="off", structures=[_dam(**_BARRIER_ONLY)], **_RUN),
        _basin(),
        tmp_path / "off.zarr",
        device="cpu",
        verbose=False,
    )
    on = run_simulation(
        Scenario(name="on", structures=[_dam(release_m3_s=4.0)], **_RUN),
        _basin(),
        tmp_path / "on.zarr",
        device="cpu",
        verbose=False,
    )
    a = xr.open_zarr(tmp_path / "off.zarr", consolidated=False)["depth"].isel(time=-1).values
    b = xr.open_zarr(tmp_path / "on.zarr", consolidated=False)["depth"].isel(time=-1).values
    drop = _volume(a[:DAM_ROW]) - _volume(b[:DAM_ROW])
    print(f"\n[rule on/off] pool volume drop with the rule on: {drop:.1f} m3")
    assert drop > 0.0, "the release rule made no difference to the pool"
    assert off.max_rel_error < MASS_GATE and on.max_rel_error < MASS_GATE


def test_the_slow_clock_is_independent_of_the_fast_timestep(tmp_path):
    """Determinism (§8/§12): halving dt_max must not move a single activation."""
    dam = _dam(interval_s=400.0, release_m3_s=2.0)
    times = []
    for tag, dt_max in (("coarse", 5.0), ("fine", 1.0)):
        scn = Scenario(name=f"clock_{tag}", structures=[dam], **{**_RUN, "dt_max": dt_max})
        run_simulation(scn, _basin(), tmp_path / f"{tag}.zarr", device="cpu", verbose=False)
        ds = xr.open_zarr(tmp_path / f"{tag}.zarr", consolidated=False)
        times.append([r["time"] for r in ds.attrs["reservoir_releases"]["d"]])
    print(f"\n[clock independence] coarse={times[0]} fine={times[1]}")
    assert times[0] == pytest.approx([400.0, 800.0, 1200.0])
    assert times[0] == pytest.approx(times[1])


def test_build_operators_skips_structures_without_a_rule():
    st = State.from_bed(_basin(), dx=DX, device=DEV)
    ops = build_operators([_dam(**_BARRIER_ONLY), _dam(name="live")], st)
    assert [o.structure.name for o in ops] == ["live"]


def test_operator_rejects_a_pool_outside_the_grid():
    st = State.from_bed(_basin(), dx=DX, device=DEV)
    with pytest.raises(ValueError, match="outside the"):
        ReservoirOperator(_dam(pool=(0, 0, 5, NX + 40)), st)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
