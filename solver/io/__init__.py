"""Solver I/O - the decoupling contract (HANDOFF §7).

The solver only ever *writes* these; the viewer only ever *reads* them.

Planned modules:
  config.py         load/validate scenario TOML              (§7.1)
  zarr_writer.py    canonical chunked result store           (§7.2)  [M2]
  viewer_export.py  per-frame raw float32 tiles + manifest   (§7.3)  [M2]
  status.py         status.json run-control progress         (§7.4)  [M2]
"""
