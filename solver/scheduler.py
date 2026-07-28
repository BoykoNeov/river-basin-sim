"""Multi-rate scheduler (M5, HANDOFF §2, §8 -- one clock, many rates).

The locked time-integration decision is *"multi-rate scheduler, single simulated
clock, deterministic adaptive CFL dt, operator splitting"*. This module owns the
clock and nothing else:

  * the **fast** process (the flood scheme) sub-cycles with its own state-derived
    ``dt`` -- the scheduler never invents one, it only ever *clamps* the one the
    scheme computed;
  * **sync points** are the simulated times a step may not cross: output cadence,
    forcing breakpoints (rain on/off, hydrograph knots, boundary-stage knots),
    ``end_time``, and every **slow-process activation**;
  * **slow processes** (M5: reservoir release rules; M7: morphology) advance at
    those activations by the *exact elapsed simulated time*, via operator
    splitting.

**It is a clock, not a driver.** :meth:`MultiRateScheduler.ticks` is a generator
yielding one :class:`Tick` per fast step; the caller still owns the state, the
stepping, the forcing, the mass ledger and the I/O (see :func:`solver.run.run_simulation`).
That keeps the hard part -- the sync-point algebra -- unit-testable with a stub
``dt_fn`` and no GPU state at all, and it keeps the scheme dispatch seam (§4 of the
M4 plan) untouched.

**Determinism (HANDOFF §8/§12).** Activation times are ``k * interval`` derived
from the *simulated* clock -- never wall-clock, never step count -- and a slow
process is handed the true elapsed simulated time since its last activation, so
the split is reproducible no matter how many fast sub-steps fell in between. The
event algebra here is exactly the ``_next_event_time`` clamping M1-M4 ran inline;
with no slow processes the event set, the arithmetic and its order are unchanged,
so pre-M5 runs remain **bitwise-identical**.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

# Time-comparison tolerance (simulated seconds). Shared with solver.run so the
# run loop and the clock agree on "lands on" for an output/sync time.
EPS_T = 1e-6


@dataclass(frozen=True)
class SlowProcess:
    """A process advanced on its own coarse clock at sync points.

    ``interval`` is the simulated-seconds cadence; activations land at
    ``interval, 2*interval, ...`` up to and including ``end_time``. The scheduler
    only *schedules* -- it never calls ``advance`` itself. The caller invokes it
    once per due tick with ``(t, elapsed)``: the simulated time of the activation
    and the simulated time that has passed since this process last ran (the
    operator-splitting interval). Keeping ``advance`` a plain callable is what lets
    the scheduler stay free of any Warp/`State` import.
    """

    name: str
    interval: float
    advance: Callable[[float, float], object]

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError(f"slow process '{self.name}': interval must be > 0")

    def activations(self, end_time: float) -> list[float]:
        """Activation times in ``(0, end_time]`` (exact multiples of ``interval``)."""
        n = int((end_time + EPS_T) // self.interval)
        return [self.interval * k for k in range(1, n + 1)]


@dataclass(frozen=True)
class Tick:
    """One fast step: advance the state from ``t0`` to ``t1 = t0 + dt``.

    ``due`` lists the slow processes to advance **after** the fast step has landed
    on ``t1``, each with the elapsed simulated time to advance it by. ``is_output``
    marks a tick that lands on the output cadence (record + write).
    """

    index: int
    t0: float
    dt: float
    t1: float
    is_output: bool
    due: tuple[tuple[SlowProcess, float], ...] = ()


@dataclass
class MultiRateScheduler:
    """The single simulated clock for one run.

    ``events`` are extra sync times the caller's forcing needs (rain on/off,
    hydrograph knots, stage-curve knots); output times and ``end_time`` are added
    here. Steps are clamped so they never cross a sync point, which is what keeps
    frames on the cadence, keeps each step wholly inside one forcing segment, and
    keeps a slow process's split interval exact.
    """

    end_time: float
    output_every: float
    events: Sequence[float] = ()
    processes: Sequence[SlowProcess] = ()
    _sync: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.end_time <= 0:
            raise ValueError(f"end_time must be > 0, got {self.end_time}")
        if self.output_every <= 0:
            raise ValueError(f"output_every must be > 0, got {self.output_every}")
        # Build the sync set in the M1-M4 order -- output times first, then the
        # caller's forcing events, then slow activations -- so a run without slow
        # processes sees the identical float list the inline loop used.
        self._sync = list(self.output_times) + list(self.events)
        for p in self.processes:
            self._sync.extend(p.activations(self.end_time))
        self._sync.append(self.end_time)

    # --- schedule queries (pure arithmetic; no state) ------------------------- #
    @property
    def n_frames(self) -> int:
        """Number of output frames including the ``t = 0`` baseline (§7.2)."""
        return int(round(self.end_time / self.output_every)) + 1

    @property
    def output_times(self) -> list[float]:
        """Output times after ``t = 0`` (``output_every * k``)."""
        return [self.output_every * k for k in range(1, self.n_frames)]

    @property
    def sync_times(self) -> list[float]:
        """All sync times (unsorted, duplicates kept -- ``min`` is what's used)."""
        return list(self._sync)

    def next_sync(self, t: float) -> float:
        """Smallest sync time strictly after ``t`` (``inf`` if none remain)."""
        future = [e for e in self._sync if e > t + EPS_T]
        return min(future) if future else float("inf")

    # --- the clock ------------------------------------------------------------ #
    def ticks(self, dt_fn: Callable[[], float]) -> Iterator[Tick]:
        """Yield one :class:`Tick` per fast step until ``end_time``.

        ``dt_fn`` is called once per step and must return the scheme's
        **state-derived** timestep (``scheme.compute_dt``); the scheduler clamps it
        to the next sync point and never otherwise modifies it. The caller advances
        the state inside the loop body, then runs ``tick.due``.
        """
        next_output = self.output_every
        next_act = {p.name: (p, list(p.activations(self.end_time))) for p in self.processes}
        last_act = dict.fromkeys(next_act, 0.0)
        t = 0.0
        index = 0
        while t < self.end_time - EPS_T:
            dt = float(dt_fn())
            dt = min(dt, self.next_sync(t) - t)
            if dt <= 0.0:
                # Cannot happen with a positive scheme dt (sync points are > t+EPS),
                # but a zero/negative dt would spin the loop forever -- fail loudly.
                raise RuntimeError(f"non-positive timestep {dt!r} at t={t}")
            t1 = t + dt

            is_output = t1 >= next_output - EPS_T and next_output <= self.end_time + EPS_T
            if is_output:
                next_output += self.output_every

            due: list[tuple[SlowProcess, float]] = []
            for name, (proc, times) in next_act.items():
                fired = False
                while times and t1 >= times[0] - EPS_T:
                    times.pop(0)
                    fired = True
                if fired:
                    due.append((proc, t1 - last_act[name]))
                    last_act[name] = t1

            yield Tick(index=index, t0=t, dt=dt, t1=t1, is_output=is_output, due=tuple(due))
            t = t1
            index += 1


def merge_events(*groups: Iterable[float]) -> list[float]:
    """Flatten forcing-breakpoint groups into one event list (order preserved)."""
    out: list[float] = []
    for g in groups:
        out.extend(float(x) for x in g)
    return out
