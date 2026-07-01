"""Scenario config loader (M2, HANDOFF §7.1 -- the config-in half of the loop).

The §7.1 TOML is the input side of the decoupling contract: config + parameter
fields + command log fully determine a run (§7.4). M2 implements the *loop-closing*
subset -- uniform rainfall, closed boundaries, a scalar Manning ``n`` -- but the
loader parses the **full** schema and **rejects** anything it cannot yet honour
with a message naming the field and the milestone that adds it. That loud refusal
is the scope gate: a config never silently means less than it says.

Supported now (M2)::

    [meta]      name, seed, scheme="local_inertial"
    [grid]      tiles_dir, dx?, crs?          (dx/crs default from the tile manifest)
    [run]       end_time, output_every, cfl, dt_max
    [rainfall]  type="uniform", rate_mm_hr, duration_s
    [parameters] manning_n = <scalar>
    [boundaries] default="closed"

Rejected until a later milestone: ``scheme="hllc_fv"`` (M4), non-uniform rainfall
(M3), open/inflow/fixed-stage boundaries (M3), parameter *field* rasters and
``infiltration`` (M3), ``[[structures]]`` (M5).
"""

from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """A scenario config is malformed or asks for an unsupported feature."""


@dataclass
class Scenario:
    """Solver run configuration (§7.1).

    ``dx``/``crs`` may be ``None``/"" meaning "inherit from the tile manifest";
    :func:`solver.run.main` fills them from ``tiles.json`` before stepping.
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
    manning_n: float = 0.035
    rain_mm_hr: float = 50.0
    rain_duration: float = 1800.0  # seconds rain falls for
    initial_depth: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def rain_m_s(self) -> float:
        return self.rain_mm_hr / 1000.0 / 3600.0


# Tables/keys the M2 loader knows about; anything else warns (typo guard).
_KNOWN_TABLES = {"meta", "grid", "run", "rainfall", "parameters", "boundaries", "structures"}
_KNOWN_KEYS = {
    "meta": {"name", "seed", "scheme"},
    "grid": {"tiles_dir", "dx", "crs"},
    "run": {"end_time", "output_every", "cfl", "dt_max"},
    "rainfall": {"type", "rate_mm_hr", "duration_s"},
    "parameters": {"manning_n", "infiltration"},
    "boundaries": {"default"},
}


def _warn_unknown(table: str, data: dict) -> None:
    for key in data:
        if key not in _KNOWN_KEYS.get(table, set()):
            warnings.warn(f"[{table}] unknown key '{key}' ignored", stacklevel=3)


def load_config(path: str | Path) -> Scenario:
    """Parse a §7.1 scenario TOML into a :class:`Scenario`, enforcing M2 scope.

    Raises :class:`ConfigError` for malformed input or any feature deferred to a
    later milestone (the message names the field and that milestone).
    """
    path = Path(path)
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
            f"[meta] scheme='{scheme}' is not supported yet; M2 solves only "
            "'local_inertial'. The HLLC FV scheme arrives in M4."
        )

    rain_type = rainfall.get("type", "uniform")
    if rain_type != "uniform":
        raise ConfigError(
            f"[rainfall] type='{rain_type}' is not supported yet; M2 solves only "
            "'uniform' rainfall. Fields/timeseries/storm_cells arrive in M3."
        )

    bc = boundaries.get("default", "closed")
    if bc != "closed":
        raise ConfigError(
            f"[boundaries] default='{bc}' is not supported yet; M2 has closed "
            "(reflective) boundaries only. Open/inflow/fixed_stage arrive in M3."
        )
    # Per-edge boundary overrides (any extra key beyond 'default') are M3 too.
    extra_bc = set(boundaries) - {"default"}
    if extra_bc:
        raise ConfigError(
            f"[boundaries] per-edge overrides {sorted(extra_bc)} are not supported "
            "yet; M2 is closed boundaries only. Overrides arrive in M3."
        )

    manning = parameters.get("manning_n", 0.035)
    if not isinstance(manning, (int, float)) or isinstance(manning, bool):
        raise ConfigError(
            "[parameters] manning_n must be a scalar in M2; a parameter *field* "
            f"(got {manning!r}) arrives in M3."
        )
    if "infiltration" in parameters:
        raise ConfigError(
            "[parameters] infiltration fields are not supported yet; they arrive "
            "in M3 (M2 rainfall is net, infiltration = 0)."
        )

    if "structures" in doc:
        raise ConfigError(
            "[[structures]] (dams/levees) are not supported yet; structures and "
            "release rules arrive in M5."
        )

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
            manning_n=float(manning),
            rain_mm_hr=float(rainfall.get("rate_mm_hr", defaults.rain_mm_hr)),
            rain_duration=float(rainfall.get("duration_s", defaults.rain_duration)),
            meta={"scheme": scheme, "boundaries": bc, "rain_type": rain_type},
        )
    except (TypeError, ValueError) as e:
        raise ConfigError(f"bad value in {path}: {e}") from e
