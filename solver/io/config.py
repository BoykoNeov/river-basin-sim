"""Scenario config loader (M2 + M3, HANDOFF §7.1 -- the config-in half of the loop).

The §7.1 TOML is the input side of the decoupling contract: config + parameter
fields + command log fully determine a run (§7.4). The loader parses the **full**
schema and **rejects** anything it cannot yet honour with a message naming the
field and the milestone that adds it -- that loud refusal is the scope gate: a
config never silently means less than it says.

Supported now (through M3)::

    [meta]       name, seed, scheme="local_inertial"
    [grid]       tiles_dir, dx?, crs?             (dx/crs default from the manifest)
    [run]        end_time, output_every, cfl, dt_max
    [rainfall]   type="uniform"|"field", rate_mm_hr, field?, duration_s
    [parameters] manning_n = <scalar OR field path>, infiltration = <scalar OR path>
    [[inflow]]   cell = [i, j], hydrograph = [[t, Q], ...]      (m^3/s)
    [boundaries] default="closed"|"open", north/south/east/west = "closed"|"open"

M3 adds: spatially-varying ``manning_n`` / ``infiltration`` fields, ``field``
rainfall, inflow hydrographs, and open boundaries (§9 M3). Field paths are raw
little-endian float32 ``.r32`` aligned to the terrain tile (an optional ``.tif``
is accepted when rasterio is available -- see :mod:`solver.io.fields`).

Rejected until a later milestone: ``scheme="hllc_fv"`` and ``fixed_stage``/
``inflow`` boundary *types* (M4), temporal rainfall ``timeseries``/``storm_cells``
(later), ``[[structures]]`` (M5). Field paths are resolved relative to the TOML
file's directory.
"""

from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# Rainfall types this milestone honours (spatial only; temporal rain deferred).
_RAIN_TYPES = {"uniform", "field"}
# Per-edge boundary behaviours this milestone honours.
_BC_TYPES = {"closed", "open"}
# Edge names -> which domain face they map to (see solver.core.grid docstring).
_EDGES = ("north", "south", "east", "west")


class ConfigError(ValueError):
    """A scenario config is malformed or asks for an unsupported feature."""


@dataclass
class Inflow:
    """A point-source inflow hydrograph (M3, §7.1 ``[[inflow]]``).

    ``cell`` is the ``(row, col)`` cell that receives the discharge; ``hydrograph``
    is a list of ``(time_s, discharge_m3_s)`` breakpoints, piecewise-linear and
    zero-held outside its range. Times must be non-decreasing.
    """

    cell: tuple[int, int]
    hydrograph: list[tuple[float, float]]

    @property
    def breakpoints(self) -> list[float]:
        """Hydrograph knot times (for clamping steps so Q is linear per step)."""
        return [t for t, _ in self.hydrograph]

    def discharge_at(self, t: float) -> float:
        """Piecewise-linear discharge at ``t`` (m^3/s); 0 outside the curve."""
        hg = self.hydrograph
        if not hg or t < hg[0][0] or t > hg[-1][0]:
            return 0.0
        for (t0, q0), (t1, q1) in zip(hg, hg[1:], strict=False):
            if t0 <= t <= t1:
                if t1 == t0:
                    return q1
                return q0 + (q1 - q0) * (t - t0) / (t1 - t0)
        return hg[-1][1]


@dataclass
class Scenario:
    """Solver run configuration (§7.1).

    ``dx``/``crs`` may be ``None``/"" meaning "inherit from the tile manifest";
    :func:`solver.run.main` fills them from ``tiles.json`` before stepping.

    Parameter fields (``manning_field``, ``infiltration_field``, ``rain_field``)
    are absolute paths when set (resolved relative to the source TOML); when unset
    the corresponding scalar (``manning_n``, ``infiltration_mm_hr``, ``rain_mm_hr``)
    applies uniformly.
    """

    name: str = "demo_basin_rain"
    seed: int = 0
    tiles_dir: str = "data/tiles/demo"
    dx: float | None = None  # metres; None -> take from the tile manifest
    crs: str = ""  # "" -> take from the tile manifest
    end_time: float = 3600.0  # simulated seconds
    output_every: float = 300.0
    alpha: float = 0.7  # CFL-like coefficient for the adaptive timestep (TOML: cfl)
    dt_max: float = 30.0
    # Roughness: scalar OR a field path (field wins when set).
    manning_n: float = 0.035
    manning_field: str | None = None
    # Infiltration loss (mm/hr): scalar OR a field path (0 = none).
    infiltration_mm_hr: float = 0.0
    infiltration_field: str | None = None
    # Rainfall: "uniform" (scalar rate) or "field" (rate raster).
    rain_type: str = "uniform"
    rain_mm_hr: float = 50.0
    rain_field: str | None = None
    rain_duration: float = 1800.0  # seconds rain falls for
    # Inflow hydrographs (point sources).
    inflows: list[Inflow] = field(default_factory=list)
    # Per-edge boundary behaviour: {north, south, east, west} -> "closed"|"open".
    boundaries: dict[str, str] = field(default_factory=lambda: {e: "closed" for e in _EDGES})
    initial_depth: float = 0.0
    source_path: str | None = None  # the TOML this was loaded from (provenance)
    meta: dict = field(default_factory=dict)

    @property
    def rain_m_s(self) -> float:
        return self.rain_mm_hr / 1000.0 / 3600.0

    @property
    def has_open_boundary(self) -> bool:
        return any(v == "open" for v in self.boundaries.values())

    def field_paths(self) -> dict[str, str]:
        """Referenced field files by role (for provenance hashing)."""
        return {
            role: p
            for role, p in (
                ("manning", self.manning_field),
                ("infiltration", self.infiltration_field),
                ("rain", self.rain_field),
            )
            if p
        }


# Tables/keys the loader knows about; anything else warns (typo guard).
_KNOWN_TABLES = {
    "meta",
    "grid",
    "run",
    "rainfall",
    "parameters",
    "boundaries",
    "inflow",
    "structures",
}
_KNOWN_KEYS = {
    "meta": {"name", "seed", "scheme"},
    "grid": {"tiles_dir", "dx", "crs"},
    "run": {"end_time", "output_every", "cfl", "dt_max"},
    "rainfall": {"type", "rate_mm_hr", "field", "duration_s"},
    "parameters": {"manning_n", "infiltration"},
    "boundaries": {"default", *_EDGES},
}


def _warn_unknown(table: str, data: dict) -> None:
    for key in data:
        if key not in _KNOWN_KEYS.get(table, set()):
            warnings.warn(f"[{table}] unknown key '{key}' ignored", stacklevel=3)


def _resolve_path(base_dir: Path, value: str) -> str:
    """Resolve a field path relative to the config file's directory."""
    p = Path(value)
    return str(p if p.is_absolute() else (base_dir / p))


def _parse_field_param(
    parameters: dict, key: str, base_dir: Path, *, default_scalar: float
) -> tuple[float, str | None]:
    """Parse a ``scalar OR path`` parameter -> (scalar, field_path_or_None)."""
    if key not in parameters:
        return default_scalar, None
    val = parameters[key]
    if isinstance(val, bool):  # bool is an int subclass -- reject explicitly
        raise ConfigError(f"[parameters] {key} must be a number or a field path, got {val!r}")
    if isinstance(val, (int, float)):
        return float(val), None
    if isinstance(val, str):
        return default_scalar, _resolve_path(base_dir, val)
    raise ConfigError(f"[parameters] {key} must be a number or a field path, got {val!r}")


def _parse_inflows(doc: dict, ny_nx: tuple[int, int] | None = None) -> list[Inflow]:
    """Parse the ``[[inflow]]`` array into validated :class:`Inflow` records."""
    raw = doc.get("inflow", [])
    if isinstance(raw, dict):  # a single [inflow] table rather than [[inflow]]
        raw = [raw]
    inflows: list[Inflow] = []
    for k, entry in enumerate(raw):
        cell = entry.get("cell")
        if not (isinstance(cell, list) and len(cell) == 2):
            raise ConfigError(f"[[inflow]] #{k}: 'cell' must be [row, col], got {cell!r}")
        hg = entry.get("hydrograph")
        if not (isinstance(hg, list) and hg and all(len(pt) == 2 for pt in hg)):
            raise ConfigError(
                f"[[inflow]] #{k}: 'hydrograph' must be a non-empty list of [t, Q] pairs"
            )
        pts = [(float(t), float(q)) for t, q in hg]
        times = [t for t, _ in pts]
        if any(b < a for a, b in zip(times, times[1:], strict=False)):
            raise ConfigError(f"[[inflow]] #{k}: hydrograph times must be non-decreasing")
        inflows.append(Inflow(cell=(int(cell[0]), int(cell[1])), hydrograph=pts))
    return inflows


def _parse_boundaries(boundaries: dict) -> dict[str, str]:
    """Resolve per-edge boundary behaviour, applying ``default`` to unset edges."""
    default = boundaries.get("default", "closed")
    if default not in _BC_TYPES:
        raise ConfigError(
            f"[boundaries] default='{default}' is not supported; M3 has "
            "'closed' or 'open'. 'fixed_stage'/'inflow' boundary types arrive in M4."
        )
    resolved = {}
    for edge in _EDGES:
        val = boundaries.get(edge, default)
        if val not in _BC_TYPES:
            raise ConfigError(
                f"[boundaries] {edge}='{val}' is not supported; use 'closed' or "
                "'open'. 'fixed_stage'/'inflow' boundary types arrive in M4."
            )
        resolved[edge] = val
    return resolved


def load_config(path: str | Path) -> Scenario:
    """Parse a §7.1 scenario TOML into a :class:`Scenario`, enforcing scope.

    Raises :class:`ConfigError` for malformed input or any feature deferred to a
    later milestone (the message names the field and that milestone). Field paths
    are resolved relative to ``path``'s directory.
    """
    path = Path(path)
    base_dir = path.resolve().parent
    try:
        with path.open("rb") as f:
            doc = tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {path}: {e}") from e

    for table in doc:
        if table not in _KNOWN_TABLES:
            warnings.warn(f"unknown top-level table '[{table}]' ignored", stacklevel=2)

    meta = doc.get("meta", {})
    grid = doc.get("grid", {})
    run = doc.get("run", {})
    rainfall = doc.get("rainfall", {})
    parameters = doc.get("parameters", {})
    boundaries = doc.get("boundaries", {})
    for name, table in (
        ("meta", meta),
        ("grid", grid),
        ("run", run),
        ("rainfall", rainfall),
        ("parameters", parameters),
        ("boundaries", boundaries),
    ):
        _warn_unknown(name, table)

    # --- scope gate: reject deferred features loudly ---------------------------
    scheme = meta.get("scheme", "local_inertial")
    if scheme != "local_inertial":
        raise ConfigError(
            f"[meta] scheme='{scheme}' is not supported yet; the solver runs only "
            "'local_inertial'. The HLLC FV scheme arrives in M4."
        )

    rain_type = rainfall.get("type", "uniform")
    if rain_type not in _RAIN_TYPES:
        raise ConfigError(
            f"[rainfall] type='{rain_type}' is not supported yet; M3 solves "
            "'uniform' or spatial 'field' rainfall. Temporal timeseries/storm_cells "
            "arrive later."
        )
    rain_field = None
    if rain_type == "field":
        if "field" not in rainfall:
            raise ConfigError("[rainfall] type='field' requires a 'field' path")
        rain_field = _resolve_path(base_dir, str(rainfall["field"]))

    if "structures" in doc:
        raise ConfigError(
            "[[structures]] (dams/levees) are not supported yet; structures and "
            "release rules arrive in M5."
        )

    bc = _parse_boundaries(boundaries)
    manning_n, manning_field = _parse_field_param(
        parameters, "manning_n", base_dir, default_scalar=Scenario().manning_n
    )
    infil_mm_hr, infil_field = _parse_field_param(
        parameters, "infiltration", base_dir, default_scalar=0.0
    )
    inflows = _parse_inflows(doc)

    # --- build the Scenario ----------------------------------------------------
    defaults = Scenario()
    try:
        return Scenario(
            name=str(meta.get("name", defaults.name)),
            seed=int(meta.get("seed", defaults.seed)),
            tiles_dir=str(grid.get("tiles_dir", defaults.tiles_dir)),
            dx=(float(grid["dx"]) if "dx" in grid else None),
            crs=str(grid.get("crs", "")),
            end_time=float(run.get("end_time", defaults.end_time)),
            output_every=float(run.get("output_every", defaults.output_every)),
            alpha=float(run.get("cfl", defaults.alpha)),
            dt_max=float(run.get("dt_max", defaults.dt_max)),
            manning_n=manning_n,
            manning_field=manning_field,
            infiltration_mm_hr=infil_mm_hr,
            infiltration_field=infil_field,
            rain_type=rain_type,
            rain_mm_hr=float(rainfall.get("rate_mm_hr", defaults.rain_mm_hr)),
            rain_field=rain_field,
            rain_duration=float(rainfall.get("duration_s", defaults.rain_duration)),
            inflows=inflows,
            boundaries=bc,
            source_path=str(path),
            meta={"scheme": scheme, "boundaries": bc, "rain_type": rain_type},
        )
    except (TypeError, ValueError) as e:
        raise ConfigError(f"bad value in {path}: {e}") from e
