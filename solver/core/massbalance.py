"""Global mass-balance diagnostics (M1/M7, HANDOFF §8 -- the credibility gauges).

Two ledgers live here, one per conserved substance: :class:`MassLedger` for water
(M1) and :class:`SedimentLedger` for the solid bed (M7). They share a *shape* --
a record dataclass, a causal peak in the relative denominator, a hard gate whose
exceedance is a failing test rather than a warning -- and deliberately not their
arithmetic; see the sediment section for why.

The honest "is this still physical?" readout: each accounting point we compare
cumulative sources/sinks against the change in stored water volume::

    residual(t) = inflow_cum - outflow_cum - ( V(t) - V(0) )

For a closed domain with uniform rainfall (M1) ``outflow_cum = 0`` and
``inflow_cum`` is the accumulated rain volume. A run whose relative residual
exceeds the gate is a **bug**, not a warning (HANDOFF §8, §10).

**Why host-side float64/Kahan.** Fields are float32 on the GPU (§2), but the
accumulator that judges them must not itself leak precision. A float *sum*
reduction on the GPU is also not order-deterministic under atomics, which would
break the determinism invariant (§12). So the volume ``V(t)`` is summed on the
host in float64 (from the float32 field copied back at output cadence), and the
source accumulators use Kahan compensated summation. Computed only at output
cadence -- not every step -- so the host copy is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from solver.core.state import State

# Relative mass-balance gate (HANDOFF §8/§10): exceedance is a failing test.
MASS_GATE = 1.0e-6


@dataclass
class _Kahan:
    """Compensated (Kahan) float64 accumulator -- resists drift over many adds."""

    total: float = 0.0
    comp: float = 0.0  # running compensation for lost low-order bits

    def add(self, x: float) -> None:
        y = x - self.comp
        t = self.total + y
        self.comp = (t - self.total) - y
        self.total = t


@dataclass
class MassRecord:
    """One accounting point in the mass-balance series."""

    time: float
    volume: float
    inflow_cum: float
    outflow_cum: float
    residual: float
    rel_error: float


@dataclass
class MassLedger:
    """Float64 mass accounting for one run.

    Construct from the initial state (captures ``V(0)``), call
    :meth:`add_inflow` / :meth:`add_outflow` as sources act during stepping, and
    :meth:`record` at each output time.
    """

    cell_area: float
    v0: float
    _inflow: _Kahan = field(default_factory=_Kahan)
    _outflow: _Kahan = field(default_factory=_Kahan)
    series: list[MassRecord] = field(default_factory=list)
    # Largest stored volume seen so far in *this* run (causal -- never a future
    # value). Floors the relative-error denominator so a drain-to-empty run cannot
    # trip the gate by denominator collapse (M4 §2); see :meth:`record`.
    _peak_v: float = 0.0

    @classmethod
    def from_state(cls, state: State) -> MassLedger:
        area = state.grid.cell_area
        v0 = cls._volume(state, area)
        ledger = cls(cell_area=area, v0=v0)
        ledger.record(state, 0.0)  # t=0 baseline (residual exactly 0)
        return ledger

    @staticmethod
    def _volume(state: State, cell_area: float) -> float:
        """Stored water volume ``dx^2 * sum(h)`` in float64 from the float32 field."""
        return float(state.h.numpy().astype(np.float64).sum()) * cell_area

    def add_inflow(self, volume: float) -> None:
        """Add a positive source volume (e.g. one step of rainfall), Kahan-summed."""
        self._inflow.add(volume)

    def add_outflow(self, volume: float) -> None:
        """Add a sink volume (M1: unused; here for M3 open boundaries)."""
        self._outflow.add(volume)

    def add_rain_step(self, rain_m_s: float, dt: float, n_cells: int) -> None:
        """Uniform rain applied to every cell for ``dt`` seconds -> inflow volume."""
        self.add_inflow(rain_m_s * dt * self.cell_area * n_cells)

    def record(self, state: State, time: float) -> MassRecord:
        """Snapshot the balance at ``time`` and append it to the series.

        Local sinks (M3 infiltration + open-boundary outflow) are read as an
        **absolute** cumulative volume from ``state.loss_cum`` (float64-summed) and
        added to the incremental ``_outflow`` accumulator -- so the residual tests
        both sides of the ledger.
        """
        v = self._volume(state, self.cell_area)
        inflow = self._inflow.total
        outflow = self._outflow.total + state.loss_volume(self.cell_area)
        residual = inflow - outflow - (v - self.v0)
        # Causal peak-volume floor (M4 §2). In a drain-to-empty run with no inflow,
        # both abs(inflow) and abs(v) -> 0, so a tiny *absolute* residual (float32
        # flux-divergence roundoff) would trip the relative gate via denominator
        # collapse rather than physics -- and the EA benchmark suite drains domains
        # fully. peak_v is the largest stored volume the run has actually held so far
        # (updated *before* the denom below): it only ever *raises* denom, so
        # rel_error only ever decreases and every `< gate` test still passes; for a
        # monotonic-fill run peak_v == v at each record, so reported rel_error is
        # bitwise-identical to before (M1/M2/M3 filling numbers unchanged).
        self._peak_v = max(self._peak_v, v)
        denom = max(abs(inflow), abs(v), self._peak_v, 1e-12)
        rel = abs(residual) / denom
        rec = MassRecord(time, v, inflow, outflow, residual, rel)
        self.series.append(rec)
        return rec

    @property
    def max_rel_error(self) -> float:
        """Worst relative residual over the run so far (the gate quantity)."""
        return max((r.rel_error for r in self.series), default=0.0)

    def as_attrs(self) -> dict:
        """Serialize the series for the Zarr ``.zattrs`` (HANDOFF §7.2)."""
        return {
            "mass_gate": MASS_GATE,
            "mass_max_rel_error": self.max_rel_error,
            "mass_balance_series": [
                {
                    "time": r.time,
                    "volume": r.volume,
                    "inflow_cum": r.inflow_cum,
                    "outflow_cum": r.outflow_cum,
                    "residual": r.residual,
                    "rel_error": r.rel_error,
                }
                for r in self.series
            ],
        }


# --- Sediment (M7) -------------------------------------------------------------
# The water ledger's *idiom*, not its arithmetic. Every term below is a fresh
# float64 reduction over a float64 field at each accounting point -- not a running
# sum of many small increments -- so there is nothing for `_Kahan` to compensate and
# it is deliberately unused here. What is reused is the shape: a record dataclass, a
# causal peak in the denominator, a relative gate, `as_attrs` for the store.

# Relative sediment-balance gate. Five orders tighter than the water gate, because
# this balance is float64 arithmetic over a float64 field where the water gate has to
# absorb float32 flux divergence -- so it is set from measurement, not from MASS_GATE.
# On the celerity fixture (validation.bedwave, 103 activations) the worst relative
# residual is 5.7e-15 with the bounds firing and 2.1e-16 with them absent; the
# *absolute* residuals are 1.4e-15 and 2.5e-15 m^3, i.e. the same round-off, and the
# 27x ratio is only the pinned run's 27x smaller gross. This gate leaves ~3 orders of
# headroom over that, which is still many orders under anything a real defect does:
# a sign error, a clamp that forgets to bank, or a boundary face that starts carrying
# bedload all move the residual by an O(1) fraction of the gross, not by ulps.
SEDIMENT_GATE = 1.0e-11


@dataclass
class SedimentRecord:
    """One accounting point in the sediment series. Volumes are **solid** m^3."""

    time: float
    bed_volume: float  # net solid volume the bed has gained since t=0 (signed)
    banked_volume: float  # cumulative solid volume the bounds refused to apply
    gross_volume: float  # total displaced, sum|dz| -- the scale, see the class docstring
    residual: float
    rel_error: float


@dataclass
class SedimentLedger:
    """Float64 solid-volume accounting for one morphology run (M7 plan §3).

    **The domain is closed to bedload, and that is the whole balance.** Boundary
    faces are never written by the transport kernels (they stay zero, which *is* the
    closed BC -- :mod:`solver.core.sediment`), so ``div(qs_int)`` telescopes to
    nothing across the grid and every metre the bed gains somewhere came from
    somewhere else in it. The only other term is what the per-cell bounds refused to
    apply -- a frozen structure cell, an alluvium floor, a fixture's pinned end --
    which :func:`~solver.core.sediment.exner_update` banks in metres into
    ``dz_unapplied`` rather than discarding. A bound is therefore a *supply*: a
    frozen inlet that wanted to erode 1 m and did not has fed the domain 1 m of
    solid, and ``supplied = -banked``. So::

        residual(t) = -banked(t) - bed(t)     ==  -( bed + banked )

    and it should be float64 round-off in the Exner update, nothing more.

    **There are no inflow/outflow accumulators, on purpose.** Nothing in M7 can
    move bedload across the domain edge, so they would ship with zero callers --
    the same call step 4 made when it dropped the generalised accumulation helper.
    A future sediment supply BC adds the term here; ``test_the_boundary_faces_carry
    _no_bedload`` is the standing evidence that today it is structurally zero.

    **The causal peak is the primary scale here, not a floor.** For water,
    ``peak_v`` floors a denominator that is normally nonzero, guarding the one case
    (drain to empty) where it collapses. For sediment the net bed volume is
    *identically* zero -- that is the invariant under test -- so a peak-of-net would
    be vacuous and the denominator would end up being the residual itself. The scale
    that means anything is the **gross** volume displaced, ``A*(1-p)*sum|dz|``, and
    its causal peak is what the relative error is taken against. It is reported in
    every record for the same reason: without it a near-zero residual cannot be told
    apart from "nothing happened".

    **What this balances is ``dz_cum``, not ``z``.** The float32 bed is a *rendering*
    of the float64 cumulative change (``z = float32(z0 + dz_cum)``), and differencing
    two O(100 m) float32 elevations to recover a sub-millimetre change is exactly the
    cancellation M7 plan §1.1 built ``dz_cum`` to avoid. The ledger reads the
    authoritative array; the rendering error is bounded by ``eps(z)`` per cell by
    construction and is a separate, static fact.
    """

    cell_area: float
    solid_fraction: float  # 1 - p; dz -> solid volume is cell_area * this
    series: list[SedimentRecord] = field(default_factory=list)
    # Largest gross displaced volume seen so far in *this* run (causal -- never a
    # future value). The relative denominator; see the class docstring.
    _peak_gross: float = 0.0

    @classmethod
    def from_state(cls, state: State) -> SedimentLedger:
        """Build from an armed state and take the t=0 baseline (identically zero)."""
        sed = getattr(state, "sediment", None)
        if sed is None:
            raise ValueError(
                "SedimentLedger needs a state armed with arm_sediment(); an unarmed run "
                "has a static bed and nothing to balance"
            )
        ledger = cls(cell_area=state.grid.cell_area, solid_fraction=1.0 - sed.porosity)
        ledger.record(state, 0.0)
        return ledger

    def record(self, state: State, time: float) -> SedimentRecord:
        """Snapshot the solid-volume balance at ``time`` and append it to the series.

        Both float64 fields are copied back **once** and every term derived from that
        copy, so an accounting point costs two device reads however many numbers it
        reports. Called at output cadence, not at every activation -- the invariant
        holds continuously, since only an activation changes either array.
        """
        sed = state.sediment
        solid = self.cell_area * self.solid_fraction
        dz = sed.dz_cum.numpy()
        unapplied = sed.dz_unapplied.numpy()
        bed = float(dz.sum()) * solid
        banked = float(unapplied.sum()) * solid
        gross = float(np.abs(dz).sum()) * solid
        residual = -(bed + banked)
        self._peak_gross = max(self._peak_gross, gross)
        denom = max(abs(banked), self._peak_gross, 1e-12)
        rec = SedimentRecord(
            time=float(time),
            bed_volume=bed,
            banked_volume=banked,
            gross_volume=gross,
            residual=residual,
            rel_error=abs(residual) / denom,
        )
        self.series.append(rec)
        return rec

    @property
    def max_rel_error(self) -> float:
        """Worst relative residual over the run so far (the gate quantity)."""
        return max((r.rel_error for r in self.series), default=0.0)

    def as_attrs(self) -> dict:
        """Serialize the series for the Zarr ``.zattrs`` (HANDOFF §7.2)."""
        return {
            "sediment_gate": SEDIMENT_GATE,
            "sediment_max_rel_error": self.max_rel_error,
            "sediment_balance_series": [
                {
                    "time": r.time,
                    "bed_volume": r.bed_volume,
                    "banked_volume": r.banked_volume,
                    "gross_volume": r.gross_volume,
                    "residual": r.residual,
                    "rel_error": r.rel_error,
                }
                for r in self.series
            ],
        }
