"""Run-status writer (M2, HANDOFF §7.4 -- the subprocess-progress channel).

The viewer launches the solver as a subprocess and cannot see into it; the only
back-channel is a small JSON file the solver rewrites as it advances::

    {"state": "running", "progress": 0.42, "sim_time": 1512.0,
     "eta_s": 8.3, "message": "t=1512s  h_max=0.61m"}

``state`` walks ``starting -> running -> writing -> done`` (or ``error``).
``progress`` is ``sim_time / end_time`` in ``[0, 1]``.

**Determinism note (§8/§12):** ``eta_s`` is derived from *wall-clock*, which is
fine here -- ``status.json`` is a read-only progress observer that never feeds the
timestep or the canonical Zarr. The simulation stepping stays wall-clock-free.

Writes are **atomic** (temp file + ``os.replace``) so a viewer polling the file
never reads a half-written document.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

VALID_STATES = {"starting", "running", "writing", "done", "error"}


class StatusWriter:
    """Atomic writer for ``status.json`` over the lifetime of one run."""

    def __init__(self, path: str | Path, end_time: float):
        self.path = Path(path)
        self.end_time = float(end_time)
        # Wall-clock start, only for the ETA estimate (never touches the sim).
        self._t0 = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _eta(self, sim_time: float) -> float | None:
        """Wall-seconds remaining, extrapolated from progress so far (or None)."""
        if sim_time <= 0.0 or self.end_time <= 0.0:
            return None
        elapsed = time.monotonic() - self._t0
        frac = min(sim_time / self.end_time, 1.0)
        if frac <= 0.0:
            return None
        return max(elapsed / frac - elapsed, 0.0)

    def write(
        self,
        state: str,
        *,
        sim_time: float = 0.0,
        message: str = "",
    ) -> None:
        """Atomically write a status record.

        ``progress`` and ``eta_s`` are derived from ``sim_time`` vs ``end_time``.
        """
        if state not in VALID_STATES:
            raise ValueError(f"invalid status state {state!r}; expected one of {VALID_STATES}")
        progress = 0.0
        if self.end_time > 0.0:
            progress = min(max(sim_time / self.end_time, 0.0), 1.0)
        if state == "done":
            progress = 1.0
        record = {
            "state": state,
            "progress": progress,
            "sim_time": float(sim_time),
            "eta_s": self._eta(sim_time),
            "message": message,
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic on POSIX and Windows
