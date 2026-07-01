"""Config loader + scope-gate tests (M2, HANDOFF §7.1)."""

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


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "s.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_full_config_maps_every_field(tmp_path):
    scn = load_config(_write(tmp_path, _FULL))
    assert scn.name == "cfg_test"
    assert scn.seed == 7
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


@pytest.mark.parametrize(
    ("mutation", "needle"),
    [
        ('scheme = "local_inertial"', "M4"),  # hllc_fv rejected -> names M4
        ('type = "uniform"', "M3"),  # non-uniform rainfall -> M3
        ('default = "closed"', "M3"),  # open BC -> M3
    ],
)
def test_scope_gate_names_the_milestone(tmp_path, mutation, needle):
    repl = {
        'scheme = "local_inertial"': 'scheme = "hllc_fv"',
        'type = "uniform"': 'type = "storm_cells"',
        'default = "closed"': 'default = "open"',
    }[mutation]
    text = _FULL.replace(mutation, repl)
    with pytest.raises(ConfigError, match=needle):
        load_config(_write(tmp_path, text))


def test_manning_field_rejected(tmp_path):
    text = _FULL.replace("manning_n = 0.03", 'manning_n = "data/fields/n.tif"')
    with pytest.raises(ConfigError, match="manning_n must be a scalar"):
        load_config(_write(tmp_path, text))


def test_infiltration_rejected(tmp_path):
    text = _FULL.replace("manning_n = 0.03", 'manning_n = 0.03\ninfiltration = "x.tif"')
    with pytest.raises(ConfigError, match="infiltration"):
        load_config(_write(tmp_path, text))


def test_structures_rejected(tmp_path):
    text = _FULL + '\n[[structures]]\ntype = "dam"\ncell = [1, 2]\ncrest_m = 100.0\n'
    with pytest.raises(ConfigError, match="structures"):
        load_config(_write(tmp_path, text))


def test_per_edge_boundary_override_rejected(tmp_path):
    text = _FULL + '\nnorth = "open"\n'  # extra key under [boundaries]
    with pytest.raises(ConfigError, match="per-edge"):
        load_config(_write(tmp_path, text))


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")
