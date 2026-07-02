"""Config loader + scope-gate tests (M2 + M3, HANDOFF §7.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from solver.io.config import ConfigError, Scenario, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]

_FULL = """
[meta]
name = "cfg_test"
seed = 7
scheme = "local_inertial"

[grid]
tiles_dir = "data/tiles/demo"
dx = 25.0
crs = "EPSG:32617"

[run]
end_time = 1200.0
output_every = 300.0
cfl = 0.6
dt_max = 20.0

[rainfall]
type = "uniform"
rate_mm_hr = 40.0
duration_s = 600.0

[parameters]
manning_n = 0.03

[boundaries]
default = "closed"
"""


def _write(tmp_path: Path, text: str, name: str = "s.toml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_full_config_maps_every_field(tmp_path):
    scn = load_config(_write(tmp_path, _FULL))
    assert scn.name == "cfg_test"
    assert scn.seed == 7
    assert scn.scheme == "local_inertial"
    assert scn.tiles_dir == "data/tiles/demo"
    assert scn.dx == 25.0
    assert scn.crs == "EPSG:32617"
    assert scn.end_time == 1200.0
    assert scn.output_every == 300.0
    assert scn.alpha == 0.6  # cfl -> alpha
    assert scn.dt_max == 20.0
    assert scn.rain_mm_hr == 40.0
    assert scn.rain_duration == 600.0
    assert scn.manning_n == 0.03
    # M3 defaults when omitted:
    assert scn.manning_field is None
    assert scn.infiltration_mm_hr == 0.0 and scn.infiltration_field is None
    assert scn.rain_type == "uniform" and scn.rain_field is None
    assert scn.inflows == []
    assert scn.boundaries == {e: "closed" for e in ("north", "south", "east", "west")}
    assert scn.has_open_boundary is False


def test_dx_crs_default_to_manifest_when_omitted(tmp_path):
    text = _FULL.replace("dx = 25.0\n", "").replace('crs = "EPSG:32617"\n', "")
    scn = load_config(_write(tmp_path, text))
    assert scn.dx is None  # -> resolved from tiles.json by run.main
    assert scn.crs == ""


def test_shipped_demo_scenario_loads(tmp_path):
    scn = load_config(REPO_ROOT / "scenarios" / "demo_basin_rain.toml")
    assert scn.name == "demo_basin_rain"
    assert scn.dx is None and scn.crs == ""  # inherit from the tile manifest
    # Reproduces the M1 in-code demo defaults.
    demo = Scenario()
    assert (scn.end_time, scn.output_every, scn.rain_mm_hr, scn.rain_duration) == (
        demo.end_time,
        demo.output_every,
        demo.rain_mm_hr,
        demo.rain_duration,
    )


# --- M3: features that are now SUPPORTED -----------------------------------


def test_manning_field_path_resolved_relative_to_config(tmp_path):
    text = _FULL.replace("manning_n = 0.03", 'manning_n = "fields/n.r32"')
    scn = load_config(_write(tmp_path, text))
    assert scn.manning_field == str(tmp_path / "fields" / "n.r32")
    assert scn.manning_n == Scenario().manning_n  # scalar untouched (field wins)


def test_infiltration_scalar_and_field(tmp_path):
    scn = load_config(_write(tmp_path, _FULL.replace("manning_n = 0.03", "infiltration = 5.0")))
    assert scn.infiltration_mm_hr == 5.0 and scn.infiltration_field is None
    scn2 = load_config(
        _write(tmp_path, _FULL.replace("manning_n = 0.03", 'infiltration = "f/infil.r32"'))
    )
    assert scn2.infiltration_field == str(tmp_path / "f" / "infil.r32")


def test_rain_field(tmp_path):
    text = _FULL.replace(
        'type = "uniform"\nrate_mm_hr = 40.0', 'type = "field"\nfield = "fields/rain.r32"'
    )
    scn = load_config(_write(tmp_path, text))
    assert scn.rain_type == "field"
    assert scn.rain_field == str(tmp_path / "fields" / "rain.r32")


def test_rain_field_requires_path(tmp_path):
    text = _FULL.replace('type = "uniform"\nrate_mm_hr = 40.0', 'type = "field"')
    with pytest.raises(ConfigError, match="requires a 'field' path"):
        load_config(_write(tmp_path, text))


def test_inflow_hydrograph_parsed(tmp_path):
    text = _FULL + "\n[[inflow]]\ncell = [10, 20]\nhydrograph = [[0.0, 0.0], [600.0, 5.0]]\n"
    scn = load_config(_write(tmp_path, text))
    assert len(scn.inflows) == 1
    inf = scn.inflows[0]
    assert inf.cell == (10, 20)
    assert inf.discharge_at(300.0) == pytest.approx(2.5)  # linear interp
    assert inf.discharge_at(-1.0) == 0.0 and inf.discharge_at(1e9) == 0.0
    assert inf.breakpoints == [0.0, 600.0]


def test_inflow_bad_shape_rejected(tmp_path):
    text = _FULL + "\n[[inflow]]\ncell = [10]\nhydrograph = [[0.0, 0.0]]\n"
    with pytest.raises(ConfigError, match="cell"):
        load_config(_write(tmp_path, text))


def test_open_boundaries_default_and_per_edge(tmp_path):
    scn = load_config(_write(tmp_path, _FULL.replace('default = "closed"', 'default = "open"')))
    assert all(v == "open" for v in scn.boundaries.values())
    assert scn.has_open_boundary

    text = _FULL.replace('default = "closed"', 'default = "closed"\neast = "open"')
    scn2 = load_config(_write(tmp_path, text))
    assert scn2.boundaries["east"] == "open"
    assert scn2.boundaries["west"] == "closed"
    assert scn2.has_open_boundary


# --- scope gate: features still DEFERRED ------------------------------------


@pytest.mark.parametrize(
    ("mutation", "repl", "needle"),
    [
        ('type = "uniform"', 'type = "storm_cells"', "later"),  # temporal rain
        ('default = "closed"', 'default = "fixed_stage"', "M4"),  # fixed-stage BC
    ],
)
def test_scope_gate_names_the_milestone(tmp_path, mutation, repl, needle):
    text = _FULL.replace(mutation, repl)
    with pytest.raises(ConfigError, match=needle):
        load_config(_write(tmp_path, text))


def test_hllc_scheme_is_accepted(tmp_path):
    """M4 wired up scheme='hllc_fv': config parses it (availability is decided at
    dispatch, not here) and records it on the scenario."""
    scn = load_config(
        _write(tmp_path, _FULL.replace('scheme = "local_inertial"', 'scheme = "hllc_fv"'))
    )
    assert scn.scheme == "hllc_fv"


def test_unknown_scheme_rejected(tmp_path):
    """An unknown scheme name is a config error naming the known set."""
    text = _FULL.replace('scheme = "local_inertial"', 'scheme = "quantum_flux"')
    with pytest.raises(ConfigError, match="known scheme"):
        load_config(_write(tmp_path, text))


def test_structures_rejected(tmp_path):
    text = _FULL + '\n[[structures]]\ntype = "dam"\ncell = [1, 2]\ncrest_m = 100.0\n'
    with pytest.raises(ConfigError, match="structures"):
        load_config(_write(tmp_path, text))


def test_manning_bool_rejected(tmp_path):
    text = _FULL.replace("manning_n = 0.03", "manning_n = true")
    with pytest.raises(ConfigError, match="manning_n"):
        load_config(_write(tmp_path, text))


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


# --- run-parameter validation (post_init, both construction paths) ----------


@pytest.mark.parametrize(
    ("mutation", "repl", "needle"),
    [
        ("output_every = 300.0", "output_every = 0.0", "output_every"),  # ZeroDivisionError guard
        ("end_time = 1200.0", "end_time = -5.0", "end_time"),
        ("dt_max = 20.0", "dt_max = 0.0", "dt_max"),
        ("cfl = 0.6", "cfl = 0.0", "cfl"),
        ("end_time = 1200.0", "end_time = 1000.0", "multiple of output_every"),  # non-divisible
        ("manning_n = 0.03", "manning_n = -0.01", "manning_n"),
        ("manning_n = 0.03", "infiltration = -1.0", "infiltration"),
        ("rate_mm_hr = 40.0", "rate_mm_hr = -1.0", "rainfall rate"),
    ],
)
def test_bad_run_params_rejected(tmp_path, mutation, repl, needle):
    text = _FULL.replace(mutation, repl)
    with pytest.raises(ConfigError, match=needle):
        load_config(_write(tmp_path, text))


def test_high_cfl_warns_but_loads(tmp_path):
    with pytest.warns(UserWarning, match="stability limit"):
        scn = load_config(_write(tmp_path, _FULL.replace("cfl = 0.6", "cfl = 5.0")))
    assert scn.alpha == 5.0  # still loads -- a warning, not a rejection


def test_scenario_post_init_guards_direct_construction():
    # The bare-CLI/demo path builds a Scenario directly, bypassing load_config.
    with pytest.raises(ValueError, match="output_every"):
        Scenario(output_every=0.0)
    with pytest.raises(ValueError, match="multiple of output_every"):
        Scenario(end_time=3500.0, output_every=300.0)
