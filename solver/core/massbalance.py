"""Global mass-balance diagnostic (M1, HANDOFF §8 -- the credibility gauge).

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
        """Snapshot the balance at ``time`` and append it to the series."""
        v = self._volume(state, self.cell_area)
        inflow = self._inflow.total
        outflow = self._outflow.total
        residual = inflow - outflow - (v - self.v0)
        denom = max(abs(inflow), abs(v), 1e-12)
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
