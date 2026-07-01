"""Parameter/rain field loading (M3, HANDOFF §7.1 -- spatially-varying inputs).

M3 lets ``manning_n``, ``infiltration`` and ``rainfall`` be a **field** aligned to
the terrain tile instead of a scalar. The contract format is raw little-endian
float32 ``.r32`` (row-major ``(y, x)``, the M0 tile convention the solver already
reads) so the solver run path stays dependency-light and the field registers with
the bed and the viewer stream cell-for-cell.

``load_field`` accepts either:
  * a **number** -> a uniform ``(ny, nx)`` field of that value, or
  * a **path** to a ``.r32`` (exact grid dims) -- or a ``.tif`` *iff* rasterio is
    importable (offline convenience; resampled/clipped to the grid).

A field whose shape does not match the grid is a hard error, never a silent
resample -- misalignment would corrupt every downstream cell (HANDOFF §7.3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from solver.core.grid import Grid


def load_r32(path: str | Path, grid: Grid) -> np.ndarray:
    """Load a raw little-endian float32 ``.r32`` as a grid-shaped field."""
    path = Path(path)
    ny, nx = grid.shape
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != ny * nx:
        raise ValueError(
            f"field {path.name} has {raw.size} values, expected {ny * nx} "
            f"for a {ny}x{nx} grid (fields must match the tile exactly)"
        )
    return np.ascontiguousarray(raw.reshape(ny, nx), dtype=np.float32)


def _load_tif(path: Path, grid: Grid) -> np.ndarray:
    """Load a GeoTIFF resampled to the grid (only if rasterio is available)."""
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError as e:  # pragma: no cover - depends on the optional geo extra
        raise ValueError(
            f"{path.name} is a .tif but rasterio is not installed; author fields as "
            "raw .r32 or `uv sync --extra geo`"
        ) from e
    ny, nx = grid.shape
    with rasterio.open(path) as src:
        data = src.read(1, out_shape=(ny, nx), resampling=Resampling.bilinear)
    return np.ascontiguousarray(data, dtype=np.float32)


def load_field(value: float | str | None, grid: Grid, *, scalar: float = 0.0) -> np.ndarray:
    """Resolve a scalar-or-path field spec into a ``(ny, nx)`` float32 array.

    ``value`` is a path (``.r32``/``.tif``) when a field is configured, else
    ``None`` -- in which case a uniform field of ``scalar`` is returned. A numeric
    ``value`` is also accepted and broadcast (uniform).
    """
    ny, nx = grid.shape
    if value is None:
        return np.full((ny, nx), float(scalar), dtype=np.float32)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return np.full((ny, nx), float(value), dtype=np.float32)
    path = Path(str(value))
    if path.suffix.lower() in (".tif", ".tiff"):
        return _load_tif(path, grid)
    return load_r32(path, grid)
