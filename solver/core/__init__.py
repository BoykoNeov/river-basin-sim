"""Solver core (HANDOFF §6, §8).

Planned modules:
  grid.py            tiled uniform raster, indexing, staggering
  state.py           h, hu, hv (+ bed z) field containers
  local_inertial.py  Bates et al. 2010 scheme                  [M1]
  hllc_fv.py         well-balanced Godunov HLLC finite volume   [M4]
  friction.py        Manning, semi-implicit
  boundaries.py      closed / open / inflow / fixed-stage
  massbalance.py     float64 / Kahan global accounting (the credibility gauge)
"""
