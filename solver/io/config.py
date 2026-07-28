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
    [boundaries] default="closed"|"open"
                 north/south/east/west = "closed" | "open"
                                       | { type="fixed_stage", level=<m> }
                                       | { type="fixed_stage", stage=[[t, level], ...] }

M3 adds: spatially-varying ``manning_n`` / ``infiltration`` fields, ``field``
rainfall, inflow hydrographs, and open boundaries (§9 M3). Field paths are raw
little-endian float32 ``.r32`` aligned to the terrain tile (an optional ``.tif``
is accepted when rasterio is available -- see :mod:`solver.io.fields`).

M4 adds: ``scheme="hllc_fv"`` (the well-balanced HLLC finite-volume scheme). The
scheme name is validated against the known set here; whether a known scheme is
wired up is decided at dispatch (:mod:`solver.core.schemes`).

M5 adds: ``[grid] datum`` (vertical datum shift, :mod:`solver.core.datum`) and the
``fixed_stage`` boundary type -- a prescribed water surface, constant or
piecewise-linear in time, written as a per-edge table because it carries a level.
It is **HLLC-only** (M5 plan §1.4) and rejected with the local-inertial scheme.

Rejected until a later milestone: temporal rainfall ``timeseries``/``storm_cells``
(later) and the ``inflow`` boundary *type* (deferred indefinitely -- ``[[inflow]]``
cell sources cover prescribed discharge and their mass accounting is exact).
Field paths are resolved relative to the TOML file's directory.
"""

from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from solver.core.schemes import KNOWN_SCHEMES

# Rainfall types this milestone honours (spatial only; temporal rain deferred).
_RAIN_TYPES = {"uniform", "field"}
# Per-edge boundary behaviours writable as a bare string.
_BC_TYPES = {"closed", "open"}
# ... plus "fixed_stage" (M5), which needs a table because it carries a level.
_BC_ALL = _BC_TYPES | {"fixed_stage"}
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


_STRUCTURE_KINDS = {"dam", "levee"}
_RELEASE_RULES = {"none", "fixed", "target_stage"}
# Default slow-clock cadence for a release rule: 15 simulated minutes. Small enough
# that the feedback in `target_stage` is meaningful, coarse enough that the split is
# genuinely multi-rate against a flood step of seconds (M5 plan §4).
_DEFAULT_RELEASE_INTERVAL_S = 900.0
_STRUCTURE_KEYS = {
    "name",
    "type",
    "cell",
    "cells",
    "crest_m",
    "release_rule",
    "release_m3_s",
    "target_stage_m",
    "release_max_m3_s",
    "pool",
    "outlet",
    "interval_s",
}


@dataclass
class Structure:
    """A dam or levee with an optional release rule (M5, §7.1 ``[[structures]]``).

    A structure is **barrier geometry plus a rule** (M5 plan §1.2), not a new
    momentum term:

    * ``cells`` + ``crest_m`` raise the bed to the crest, so impoundment and
      overtopping are ordinary shallow-water physics the validated scheme already
      handles. A ``levee`` is exactly this and nothing more.
    * a ``dam`` may additionally carry a **release rule**, evaluated on the slow
      clock (``interval_s``) by :mod:`solver.processes.reservoir`, which moves water
      from the ``pool`` region to the ``outlet`` cell.

    ``pool`` is an inclusive ``(row0, col0, row1, col1)`` box. Elevations
    (``crest_m``, ``target_stage_m``) are absolute and shift with ``[grid] datum``.
    """

    name: str = "dam"
    kind: str = "dam"
    cells: list[tuple[int, int]] = field(default_factory=list)
    crest_m: float = 0.0
    release_rule: str = "none"
    release_m3_s: float = 0.0  # "fixed": the constant release
    target_stage_m: float | None = None  # "target_stage": the level to draw down to
    release_max_m3_s: float = 0.0  # "target_stage": cap on the release
    pool: tuple[int, int, int, int] | None = None
    # Inclusive (row0, col0, row1, col1) box the release is delivered into. A single
    # cell is written as [row, col] in the TOML and normalised to a 1x1 box here.
    # It is worth using a *reach* rather than one cell: operator splitting delivers a
    # whole interval's release in one instant, so a single 40 m cell can receive
    # metres of water at once -- physically absurd as a state even though it drains.
    outlet: tuple[int, int, int, int] | None = None
    interval_s: float = _DEFAULT_RELEASE_INTERVAL_S  # slow-clock cadence (sim seconds)

    def __post_init__(self) -> None:
        if self.kind not in _STRUCTURE_KINDS:
            raise ValueError(
                f"structure '{self.name}': type must be one of {sorted(_STRUCTURE_KINDS)}"
            )
        if not self.cells:
            raise ValueError(f"structure '{self.name}': needs at least one barrier cell")
        if self.release_rule not in _RELEASE_RULES:
            raise ValueError(
                f"structure '{self.name}': release_rule must be one of {sorted(_RELEASE_RULES)}"
            )
        if self.kind == "levee" and self.release_rule != "none":
            raise ValueError(
                f"structure '{self.name}': a levee is barrier geometry only; use type='dam' "
                "for a release rule"
            )
        if self.interval_s <= 0:
            raise ValueError(f"structure '{self.name}': interval_s must be > 0")
        if self.release_rule == "none":
            return
        if self.pool is None or self.outlet is None:
            raise ValueError(
                f"structure '{self.name}': release_rule='{self.release_rule}' needs both a "
                "'pool' box and an 'outlet' cell (where the released water is delivered)"
            )
        r0, c0, r1, c1 = self.pool
        if r1 < r0 or c1 < c0:
            raise ValueError(f"structure '{self.name}': pool must be [row0, col0, row1, col1]")
        o0, p0, o1, p1 = self.outlet
        if o1 < o0 or p1 < p0:
            raise ValueError(
                f"structure '{self.name}': outlet must be [row, col] or [row0, col0, row1, col1]"
            )
        if not (o1 < r0 or o0 > r1 or p1 < c0 or p0 > c1):
            raise ValueError(
                f"structure '{self.name}': the outlet {self.outlet} overlaps the pool "
                f"{self.pool}; the release would just shuffle water within the reservoir"
            )
        if self.release_rule == "fixed" and self.release_m3_s <= 0:
            raise ValueError(
                f"structure '{self.name}': release_rule='fixed' needs release_m3_s > 0"
            )
        if self.release_rule == "target_stage":
            if self.target_stage_m is None:
                raise ValueError(
                    f"structure '{self.name}': release_rule='target_stage' needs target_stage_m"
                )
            if self.release_max_m3_s <= 0:
                raise ValueError(
                    f"structure '{self.name}': release_rule='target_stage' needs "
                    "release_max_m3_s > 0 (the cap the proportional rule scales toward)"
                )
            if self.target_stage_m >= self.crest_m:
                raise ValueError(
                    f"structure '{self.name}': target_stage_m ({self.target_stage_m}) must be "
                    f"below crest_m ({self.crest_m}) -- the rule ramps between the two"
                )

    def discharge_at(self, stage: float | None) -> float:
        """Release discharge (m^3/s) the rule asks for at the given pool stage.

        ``fixed`` is open-loop (a constant). ``target_stage`` is the closed-loop
        rule -- and the one that makes the sync-point feedback path load-bearing:
        the release ramps proportionally from 0 at the target level to
        ``release_max_m3_s`` at the crest, so the pool is drawn down toward the
        target and the release shuts off once it is reached. ``stage is None``
        (a dry pool) always means no release.
        """
        if self.release_rule == "none" or stage is None:
            return 0.0
        if self.release_rule == "fixed":
            return float(self.release_m3_s)
        span = self.crest_m - float(self.target_stage_m)
        frac = (stage - float(self.target_stage_m)) / span
        return float(self.release_max_m3_s) * min(max(frac, 0.0), 1.0)


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
    scheme: str = "local_inertial"  # "local_inertial" (M1) | "hllc_fv" (M4)
    tiles_dir: str = "data/tiles/demo"
    dx: float | None = None  # metres; None -> take from the tile manifest
    crs: str = ""  # "" -> take from the tile manifest
    # Vertical datum shift (M5): None = no shift, "auto" = floor(min(bed)), or an
    # explicit reference elevation. See solver.core.datum for why it exists.
    datum: str | float | None = None
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
    # Dams / levees with optional slow-clock release rules (M5).
    structures: list[Structure] = field(default_factory=list)
    # Per-edge boundary behaviour: {north,south,east,west} -> "closed"|"open"|"fixed_stage".
    boundaries: dict[str, str] = field(default_factory=lambda: {e: "closed" for e in _EDGES})
    # Water-level curve for each "fixed_stage" edge (M5): piecewise-linear
    # [(t_s, level_m), ...], held at its end values outside the range. A constant
    # level is a one-point curve. Absolute elevations -- shifted with the datum.
    stage_curves: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    initial_depth: float = 0.0
    source_path: str | None = None  # the TOML this was loaded from (provenance)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate run parameters on *every* construction path.

        Runs for both the config loader (a raised ``ValueError`` is wrapped into a
        milestone-naming :class:`ConfigError` by :func:`load_config`) and the bare
        ``solver.run`` CLI/demo path, which builds a :class:`Scenario` directly from
        flags (e.g. ``--output-every 0``) and never touches the loader. Only timing
        params are checked; ``dx`` stays ``None`` until manifest resolution.
        """
        if self.end_time <= 0:
            raise ValueError(f"end_time must be > 0, got {self.end_time}")
        if self.output_every <= 0:
            raise ValueError(f"output_every must be > 0, got {self.output_every}")
        if self.dt_max <= 0:
            raise ValueError(f"dt_max must be > 0, got {self.dt_max}")
        if self.alpha <= 0:
            raise ValueError(f"cfl must be > 0, got {self.alpha}")
        # Frames land on output_every multiples and n_frames assumes exact division;
        # a non-divisible pair silently drops the final state (t=end_time) and leaves
        # an unfilled Zarr slot. Reject loudly (float-tolerant on the frame count).
        n = self.end_time / self.output_every
        if abs(n - round(n)) > 1e-6:
            raise ValueError(
                f"end_time ({self.end_time}) must be an exact multiple of output_every "
                f"({self.output_every}); otherwise the final frame at end_time is dropped"
            )
        # Physical scalars are non-negative (field files are checked in solver.io.fields).
        if self.manning_n < 0:
            raise ValueError(f"manning_n must be >= 0, got {self.manning_n}")
        if self.infiltration_mm_hr < 0:
            raise ValueError(f"infiltration must be >= 0 mm/hr, got {self.infiltration_mm_hr}")
        if self.rain_mm_hr < 0:
            raise ValueError(f"rainfall rate must be >= 0 mm/hr, got {self.rain_mm_hr}")
        if self.rain_duration < 0:
            raise ValueError(f"rainfall duration must be >= 0 s, got {self.rain_duration}")
        # fixed_stage is HLLC-only (M5 plan §1.4): the local-inertial scheme has no
        # boundary faces to impose a surface on -- its BCs are a zeroed edge face
        # (closed) plus a post-interior self-capping sink (open), because the M1
        # donor limiter never scales edge faces. A pressure-driven edge flux there
        # would be exactly the unprotected case that shape exists to avoid. So this
        # is a hard error on both construction paths, not a silent approximation.
        bad_bc = sorted(f"{e}={v!r}" for e, v in self.boundaries.items() if v not in _BC_ALL)
        if bad_bc:
            raise ValueError(
                f"unknown boundary type(s): {', '.join(bad_bc)}; use {sorted(_BC_ALL)}"
            )
        stage_edges = sorted(e for e, v in self.boundaries.items() if v == "fixed_stage")
        if stage_edges and self.scheme != "hllc_fv":
            raise ValueError(
                f"boundary type 'fixed_stage' ({', '.join(stage_edges)}) requires "
                f"scheme='hllc_fv'; the '{self.scheme}' scheme has no boundary faces to "
                "prescribe a water surface on (M5 plan §1.4)"
            )
        for edge in stage_edges:
            if not self.stage_curves.get(edge):
                raise ValueError(f"boundary '{edge}' is fixed_stage but carries no stage curve")
        # The local-inertial scheme is stable only to CFL ~0.7 (Bates 2010); warn
        # loudly above that band rather than fail (experimentation is allowed).
        if self.alpha > 0.9:
            warnings.warn(
                f"cfl={self.alpha} exceeds the ~0.7 local-inertial stability limit; "
                "the run may go unstable",
                stacklevel=2,
            )

    @property
    def rain_m_s(self) -> float:
        return self.rain_mm_hr / 1000.0 / 3600.0

    @property
    def has_open_boundary(self) -> bool:
        return any(v == "open" for v in self.boundaries.values())

    @property
    def stage_events(self) -> list[float]:
        """Stage-curve knot times -- sync points so a step never straddles a slope."""
        return sorted({t for curve in self.stage_curves.values() for t, _ in curve})

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
    "grid": {"tiles_dir", "dx", "crs", "datum"},
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


_STAGE_EXAMPLE = '{ type = "fixed_stage", level = 10.0 }'


def _parse_stage_curve(edge: str, table: dict) -> list[tuple[float, float]]:
    """Parse a ``fixed_stage`` edge table into a piecewise-linear stage curve.

    Either ``level = <m>`` (constant, stored as a one-point curve) or
    ``stage = [[t_s, level_m], ...]`` (piecewise-linear, held at its end values
    outside the range -- a water level does not vanish the way a hydrograph does).
    """
    has_level, has_curve = "level" in table, "stage" in table
    if has_level == has_curve:
        raise ConfigError(
            f"[boundaries] {edge}: fixed_stage needs exactly one of 'level' "
            f"(constant, e.g. {_STAGE_EXAMPLE}) or 'stage' (a [[t, level]] curve)"
        )
    if has_level:
        lvl = table["level"]
        if isinstance(lvl, bool) or not isinstance(lvl, (int, float)):
            raise ConfigError(f"[boundaries] {edge}: fixed_stage 'level' must be a number")
        return [(0.0, float(lvl))]
    raw = table["stage"]
    if not (isinstance(raw, list) and raw and all(len(pt) == 2 for pt in raw)):
        raise ConfigError(
            f"[boundaries] {edge}: fixed_stage 'stage' must be a non-empty list of "
            "[t_s, level_m] pairs"
        )
    pts = [(float(t), float(lv)) for t, lv in raw]
    times = [t for t, _ in pts]
    if any(b < a for a, b in zip(times, times[1:], strict=False)):
        raise ConfigError(f"[boundaries] {edge}: fixed_stage 'stage' times must be non-decreasing")
    return pts


def _cell_pair(value: object, what: str) -> tuple[int, int]:
    if not (isinstance(value, list) and len(value) == 2):
        raise ConfigError(f"{what} must be [row, col], got {value!r}")
    return (int(value[0]), int(value[1]))


def _parse_structures(doc: dict) -> list[Structure]:
    """Parse the ``[[structures]]`` array into validated :class:`Structure` records.

    Barrier cells may be given as a single ``cell = [i, j]`` or a ``cells`` list;
    both forms end up in ``Structure.cells``.
    """
    raw = doc.get("structures", [])
    if isinstance(raw, dict):  # a single [structures] table rather than [[structures]]
        raw = [raw]
    out: list[Structure] = []
    for k, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"[[structures]] #{k}: must be a table")
        unknown = set(entry) - _STRUCTURE_KEYS
        if unknown:
            warnings.warn(
                f"[[structures]] #{k}: unknown key(s) {sorted(unknown)} ignored", stacklevel=3
            )
        name = str(entry.get("name", f"structure_{k}"))
        cells: list[tuple[int, int]] = []
        if "cell" in entry:
            cells.append(_cell_pair(entry["cell"], f"[[structures]] #{k}: 'cell'"))
        for c in entry.get("cells", []):
            cells.append(_cell_pair(c, f"[[structures]] #{k}: each entry of 'cells'"))
        if "crest_m" not in entry:
            raise ConfigError(f"[[structures]] #{k} ('{name}'): 'crest_m' is required")
        pool = entry.get("pool")
        if pool is not None:
            if not (isinstance(pool, list) and len(pool) == 4):
                raise ConfigError(
                    f"[[structures]] #{k} ('{name}'): 'pool' must be [row0, col0, row1, col1]"
                )
            pool = tuple(int(v) for v in pool)
        outlet = entry.get("outlet")
        if outlet is not None:
            if not (isinstance(outlet, list) and len(outlet) in (2, 4)):
                raise ConfigError(
                    f"[[structures]] #{k} ('{name}'): 'outlet' must be [row, col] or "
                    "[row0, col0, row1, col1]"
                )
            vals = tuple(int(v) for v in outlet)
            outlet = (vals[0], vals[1], vals[0], vals[1]) if len(vals) == 2 else vals
        try:
            out.append(
                Structure(
                    name=name,
                    kind=str(entry.get("type", "dam")),
                    cells=cells,
                    crest_m=float(entry["crest_m"]),
                    release_rule=str(entry.get("release_rule", "none")),
                    release_m3_s=float(entry.get("release_m3_s", 0.0)),
                    target_stage_m=(
                        float(entry["target_stage_m"]) if "target_stage_m" in entry else None
                    ),
                    release_max_m3_s=float(entry.get("release_max_m3_s", 0.0)),
                    pool=pool,
                    outlet=outlet,
                    interval_s=float(entry.get("interval_s", _DEFAULT_RELEASE_INTERVAL_S)),
                )
            )
        except (TypeError, ValueError) as e:
            raise ConfigError(f"[[structures]] #{k} ('{name}'): {e}") from e
    return out


def _parse_boundaries(
    boundaries: dict,
) -> tuple[dict[str, str], dict[str, list[tuple[float, float]]]]:
    """Resolve per-edge boundary behaviour, applying ``default`` to unset edges.

    Returns ``(types, stage_curves)``: the per-edge type map and, for every
    ``fixed_stage`` edge, its water-level curve. A bare string is ``"closed"`` or
    ``"open"``; ``fixed_stage`` is a table because it carries a level (M5).
    """
    default = boundaries.get("default", "closed")
    if not isinstance(default, str) or default not in _BC_TYPES:
        raise ConfigError(
            f"[boundaries] default={default!r} is not supported; use 'closed' or 'open'. "
            f"A fixed_stage edge is per-edge and needs a level: north = {_STAGE_EXAMPLE}."
        )
    resolved: dict[str, str] = {}
    curves: dict[str, list[tuple[float, float]]] = {}
    for edge in _EDGES:
        val = boundaries.get(edge, default)
        if isinstance(val, str):
            if val not in _BC_TYPES:
                raise ConfigError(
                    f"[boundaries] {edge}='{val}' is not supported; use 'closed', 'open', or a "
                    f"fixed_stage table ({edge} = {_STAGE_EXAMPLE}). The 'inflow' boundary "
                    "*type* stays deferred -- use [[inflow]] cell sources, which are exact."
                )
            resolved[edge] = val
            continue
        if not isinstance(val, dict):
            raise ConfigError(
                f"[boundaries] {edge}={val!r} must be 'closed', 'open', or a fixed_stage "
                f"table ({edge} = {_STAGE_EXAMPLE})"
            )
        kind = val.get("type")
        if kind == "inflow":
            raise ConfigError(
                f"[boundaries] {edge}: the 'inflow' boundary *type* is deferred -- use "
                "[[inflow]] cell sources (M3), whose mass accounting is exact by construction."
            )
        if kind != "fixed_stage":
            raise ConfigError(
                f"[boundaries] {edge}: unknown boundary type {kind!r}; the table form is "
                f"only for fixed_stage ({edge} = {_STAGE_EXAMPLE})"
            )
        resolved[edge] = "fixed_stage"
        curves[edge] = _parse_stage_curve(edge, val)
    return resolved, curves


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
    # Scheme selection is validated against the known set here; whether a known
    # scheme is actually wired up is decided at dispatch (solver.core.schemes),
    # so an unknown name is a config error but a known-but-unbuilt scheme is a
    # NotImplementedError at run time.
    scheme = meta.get("scheme", "local_inertial")
    if scheme not in KNOWN_SCHEMES:
        raise ConfigError(
            f"[meta] scheme='{scheme}' is not a known scheme; choose one of "
            f"{list(KNOWN_SCHEMES)} ('local_inertial' is M1; 'hllc_fv' is the M4 "
            "well-balanced HLLC finite-volume scheme)."
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

    datum = grid.get("datum")
    if datum is not None and not isinstance(datum, (int, float, str)):
        raise ConfigError(f"[grid] datum must be a number or 'auto', got {datum!r}")
    if isinstance(datum, str) and datum != "auto":
        raise ConfigError(f"[grid] datum must be a number or 'auto', got {datum!r}")
    if isinstance(datum, bool):  # bool is an int subclass -- reject explicitly
        raise ConfigError(f"[grid] datum must be a number or 'auto', got {datum!r}")

    bc, stage_curves = _parse_boundaries(boundaries)
    manning_n, manning_field = _parse_field_param(
        parameters, "manning_n", base_dir, default_scalar=Scenario().manning_n
    )
    infil_mm_hr, infil_field = _parse_field_param(
        parameters, "infiltration", base_dir, default_scalar=0.0
    )
    inflows = _parse_inflows(doc)
    structures = _parse_structures(doc)

    # --- build the Scenario ----------------------------------------------------
    defaults = Scenario()
    try:
        return Scenario(
            name=str(meta.get("name", defaults.name)),
            seed=int(meta.get("seed", defaults.seed)),
            scheme=scheme,
            tiles_dir=str(grid.get("tiles_dir", defaults.tiles_dir)),
            dx=(float(grid["dx"]) if "dx" in grid else None),
            crs=str(grid.get("crs", "")),
            datum=(datum if isinstance(datum, str) or datum is None else float(datum)),
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
            structures=structures,
            boundaries=bc,
            stage_curves=stage_curves,
            source_path=str(path),
            meta={"scheme": scheme, "boundaries": bc, "rain_type": rain_type},
        )
    except (TypeError, ValueError) as e:
        raise ConfigError(f"bad value in {path}: {e}") from e
