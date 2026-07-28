"""Vertical datum shift ``z' = z - z_ref`` (M5, plan §1.5).

Fields are float32 (HANDOFF §2), and the scheme's central quantity is the water
surface ``eta = h + z``. In float32 that sum resolves to about ``ULP(z)``:

    datum ~    10 m  ->  ~1e-6 m
    datum ~   500 m  ->  ~3e-5 m
    datum ~  5000 m  ->  ~5e-4 m

The last one is a *depth-sized* error for the thin sheets these benchmarks care
about -- it shows up first as a lake-at-rest that will not quite stay flat, since
the well-balanced cancellation between the pressure flux and the bed-slope source
happens in exactly that arithmetic. The fix is not to promote fields to float64
(§2 says don't) but to move the origin: subtract a constant reference elevation
from the bed before stepping, and add it back on output.

Everything the solver reports is datum-independent (depth, velocity, volumes,
mass balance) **except** absolute elevations, so the shift must be applied to the
bed *and to every absolute elevation the scenario carries* -- boundary stage
levels and structure crests -- or they silently disagree by ``z_ref``. This module
owns the arithmetic; :func:`solver.run.shift_for_datum` owns applying it to a whole
scenario in one place, so those elevations cannot drift apart.

The canonical store still records the **true** bed (§7.2): the shift is undone on
the way out and reported as ``datum_shift_m`` in ``.zattrs``, so nothing
downstream -- analysis or viewer -- needs to know it happened.
"""

from __future__ import annotations

import numpy as np


def resolve_datum(datum: str | float | None, bed: np.ndarray) -> float:
    """Resolve a scenario's ``[grid] datum`` setting to a reference elevation (m).

    ``None`` -> 0.0 (no shift; existing scenarios are bitwise-unchanged).
    ``"auto"`` -> ``floor(min(bed))``, an integer number of metres so the shift is
    exactly representable and the same for any run of the same terrain.
    A number -> itself.
    """
    if datum is None:
        return 0.0
    if isinstance(datum, str):
        if datum != "auto":
            raise ValueError(f"[grid] datum must be a number or 'auto', got {datum!r}")
        return float(np.floor(float(np.asarray(bed, dtype=np.float64).min())))
    return float(datum)


def shift_bed(bed: np.ndarray, z_ref: float) -> np.ndarray:
    """Return the bed in shifted coordinates (``bed`` itself when ``z_ref == 0``)."""
    if z_ref == 0.0:
        return bed
    return (np.asarray(bed, dtype=np.float32) - np.float32(z_ref)).astype(np.float32)


def unshift_bed(bed: np.ndarray, z_ref: float) -> np.ndarray:
    """Return a shifted bed back in true elevations (for the canonical store)."""
    if z_ref == 0.0:
        return bed
    return (np.asarray(bed, dtype=np.float32) + np.float32(z_ref)).astype(np.float32)
