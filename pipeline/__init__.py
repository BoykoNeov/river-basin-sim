"""Offline data-prep pipeline (HANDOFF §4, §6).

Turns raw DEMs + river networks into engine-ready terrain tiles.

Planned modules [M0]:
  condition.py  sink-fill + D8 flow direction/accumulation + reproject
  tile.py       cut conditioned rasters into engine tiles

See sources.md for data sources and licensing.

NOTE: DEM conditioning uses pysheds (or WhiteboxTools), NOT richdem - richdem has
no wheels past cp37 and will not build on modern Python.

pysheds 0.5 needs a small NumPy-2.x shim (it still calls the removed np.in1d);
_compat applies it on import. See pipeline/_compat.py for the full rationale.
"""

# Side-effecting import: must run before any pysheds call. See _compat docstring.
from pipeline import _compat as _compat  # noqa: F401  (imported for side effects)
