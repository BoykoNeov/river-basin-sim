"""Offline data-prep pipeline (HANDOFF §4, §6).

Turns raw DEMs + river networks into engine-ready terrain tiles.

Planned modules [M0]:
  condition.py  sink-fill + D8 flow direction/accumulation + reproject
  tile.py       cut conditioned rasters into engine tiles

See sources.md for data sources and licensing.

NOTE: DEM conditioning uses pysheds (or WhiteboxTools), NOT richdem - richdem has
no wheels past cp37 and will not build on modern Python.
"""
