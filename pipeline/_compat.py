"""Compatibility shims for the data-prep stack.

Imported once, for its side effects, at ``pipeline`` package import time.

----------------------------------------------------------------------------
pysheds 0.5 vs NumPy >= 2.0  (`np.in1d` removal)
----------------------------------------------------------------------------
pysheds 0.5 is the latest release (its classifiers stop at Python 3.9) and is
effectively unmaintained. It still calls ``np.in1d`` in nine places in
``pysheds/sgrid.py`` -- every flow-routing op that validates the direction map,
including ``accumulation``::

    invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)

NumPy 2.0 *removed* ``np.in1d`` (deprecated in 1.25, gone in 2.0); the project
pins ``numpy >= 2.4.6``, so those ops raise ``AttributeError`` unpatched.

``np.isin`` is the exact replacement: for the old API, ``np.in1d(ar1, ar2)`` is
equivalent to ``np.isin(ar1, ar2)`` whenever ``ar1`` is already 1-D -- which it
is at all nine call sites (each passes ``fdir.ravel()``). So re-pointing the
name is correct, not merely close.

This is deliberately a narrow, reversible shim, not a fork of pysheds. If
pysheds is ever dropped for WhiteboxTools (the documented fallback,
HANDOFF/CLAUDE.md), delete this module and its import in ``pipeline/__init__``.
"""

from __future__ import annotations

import numpy as np


def _patch_numpy_in1d() -> None:
    """Restore ``np.in1d`` as an alias of ``np.isin`` on NumPy >= 2.0."""
    if not hasattr(np, "in1d"):
        # All pysheds call sites pass a 1-D array, where isin is a drop-in.
        np.in1d = np.isin  # type: ignore[attr-defined]


_patch_numpy_in1d()
