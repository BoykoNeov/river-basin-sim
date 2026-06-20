"""River Basin Simulator - GPU shallow-water solver (NVIDIA Warp).

Standalone and fully decoupled from the viewer (HANDOFF §4): consumes a scenario
config + terrain tiles and emits results (Zarr + per-frame tiles + status.json).

Subpackages / modules, built milestone by milestone (HANDOFF §6, §9):
  core/         grid, state, schemes, friction, boundaries, mass balance
  io/           config load, zarr writer, viewer export, status
  processes/    rainfall (M1), reservoir (M5), sediment (M7)
  scheduler.py  multi-rate clock + operator splitting           [M5]
  run.py        entry point: config -> run -> results  (`python -m solver.run`)
"""
