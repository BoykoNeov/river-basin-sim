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
        # The `inflow` boundary *type* stays deferred: [[inflow]] cell sources cover
        # prescribed discharge and their mass accounting is exact by construction.
        ('default = "closed"', 'default = "closed"\neast = { type = "inflow" }', "deferred"),
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


# --- M5: [[structures]] ------------------------------------------------------- #
_DAM = """
[[structures]]
name = "upper_dam"
type = "dam"
cells = [[6, 4], [7, 4]]
crest_m = 145.0
release_rule = "target_stage"
target_stage_m = 143.0
release_max_m3_s = 40.0
pool = [2, 0, 5, 9]
outlet = [8, 4]
interval_s = 600.0
"""


def test_structure_parsed(tmp_path):
    scn = load_config(_write(tmp_path, _FULL + _DAM))
    (s,) = scn.structures
    assert (s.name, s.kind) == ("upper_dam", "dam")
    assert s.cells == [(6, 4), (7, 4)]
    assert (s.crest_m, s.target_stage_m, s.release_max_m3_s) == (145.0, 143.0, 40.0)
    assert s.pool == (2, 0, 5, 9) and s.outlet == (8, 4, 8, 4)  # cell -> 1x1 box
    assert s.interval_s == 600.0


def test_levee_is_barrier_only(tmp_path):
    text = _FULL + '\n[[structures]]\ntype = "levee"\ncell = [3, 3]\ncrest_m = 12.0\n'
    (s,) = load_config(_write(tmp_path, text)).structures
    assert s.kind == "levee" and s.release_rule == "none"


def test_target_stage_rule_ramps_between_target_and_crest(tmp_path):
    """The closed-loop rule: 0 at the target, capped at the crest, linear between."""
    (s,) = load_config(_write(tmp_path, _FULL + _DAM)).structures
    assert s.discharge_at(None) == 0.0  # dry pool -> no release
    assert s.discharge_at(142.0) == 0.0  # below target -> shut off
    assert s.discharge_at(144.0) == pytest.approx(20.0)  # half way -> half the cap
    assert s.discharge_at(150.0) == 40.0  # above crest -> capped, not extrapolated


def test_fixed_rule_is_open_loop(tmp_path):
    text = _FULL + (
        '\n[[structures]]\ntype = "dam"\ncell = [5, 5]\ncrest_m = 20.0\n'
        'release_rule = "fixed"\nrelease_m3_s = 7.5\npool = [0, 0, 4, 9]\noutlet = [6, 5]\n'
    )
    (s,) = load_config(_write(tmp_path, text)).structures
    assert s.discharge_at(19.0) == 7.5 and s.discharge_at(1.0) == 7.5
    assert s.discharge_at(None) == 0.0


@pytest.mark.parametrize(
    "body,needle",
    [
        ('type = "weir"\ncell = [1, 1]\ncrest_m = 5.0', "type must be one of"),
        ('type = "dam"\ncrest_m = 5.0', "at least one barrier cell"),
        ('type = "dam"\ncell = [1, 1]', "crest_m' is required"),
        ('type = "dam"\ncell = [1, 1]\ncrest_m = 5.0\nrelease_rule = "spill"', "release_rule"),
        (
            'type = "levee"\ncell = [1, 1]\ncrest_m = 5.0\nrelease_rule = "fixed"\n'
            "release_m3_s = 1.0\npool = [0, 0, 0, 1]\noutlet = [3, 3]",
            "levee is barrier geometry only",
        ),
        (
            'type = "dam"\ncell = [1, 1]\ncrest_m = 5.0\nrelease_rule = "fixed"\n'
            "release_m3_s = 1.0",
            "needs both a 'pool' box and an 'outlet'",
        ),
        (
            'type = "dam"\ncell = [1, 1]\ncrest_m = 5.0\nrelease_rule = "fixed"\n'
            "release_m3_s = 1.0\npool = [0, 0, 3, 3]\noutlet = [1, 1]",
            "overlaps the pool",
        ),
        (
            'type = "dam"\ncell = [1, 1]\ncrest_m = 5.0\nrelease_rule = "target_stage"\n'
            "target_stage_m = 6.0\nrelease_max_m3_s = 2.0\npool = [0, 0, 0, 1]\noutlet = [3, 3]",
            "must be below crest_m",
        ),
    ],
)
def test_structure_validation(tmp_path, body, needle):
    with pytest.raises(ConfigError, match=needle):
        load_config(_write(tmp_path, _FULL + "\n[[structures]]\n" + body + "\n"))


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


# --- M5: vertical datum ------------------------------------------------------ #
def test_datum_defaults_to_none(tmp_path):
    """`[grid] datum` is opt-in: absent means no shift (pre-M5 runs unchanged)."""
    assert load_config(_write(tmp_path, _FULL)).datum is None


@pytest.mark.parametrize("value,expected", [('"auto"', "auto"), ("9.7", 9.7), ("-3", -3.0)])
def test_datum_values(tmp_path, value, expected):
    cfg = _FULL.replace('crs = "EPSG:32617"', f'crs = "EPSG:32617"\ndatum = {value}')
    assert load_config(_write(tmp_path, cfg)).datum == expected


@pytest.mark.parametrize("value", ['"sea-level"', "true", "[1, 2]"])
def test_datum_rejects_nonsense(tmp_path, value):
    cfg = _FULL.replace('crs = "EPSG:32617"', f'crs = "EPSG:32617"\ndatum = {value}')
    with pytest.raises(ConfigError, match="datum"):
        load_config(_write(tmp_path, cfg))


# --- M5: fixed_stage boundaries ---------------------------------------------- #
_HLLC = _FULL.replace('scheme = "local_inertial"', 'scheme = "hllc_fv"')


def test_fixed_stage_constant_level(tmp_path):
    text = _HLLC.replace(
        'default = "closed"', 'default = "closed"\neast = { type = "fixed_stage", level = 10.35 }'
    )
    scn = load_config(_write(tmp_path, text))
    assert scn.boundaries["east"] == "fixed_stage"
    assert scn.stage_curves["east"] == [(0.0, 10.35)]  # constant -> one-point curve
    assert scn.stage_events == [0.0]


def test_fixed_stage_time_varying_curve_and_sync_events(tmp_path):
    text = _HLLC.replace(
        'default = "closed"',
        'default = "closed"\nwest = { type = "fixed_stage", '
        "stage = [[0.0, 9.7], [3600.0, 10.35], [7200.0, 9.7]] }",
    )
    scn = load_config(_write(tmp_path, text))
    assert scn.stage_curves["west"] == [(0.0, 9.7), (3600.0, 10.35), (7200.0, 9.7)]
    # Curve knots become scheduler sync points, so no step straddles a slope change.
    assert scn.stage_events == [0.0, 3600.0, 7200.0]


def test_fixed_stage_requires_the_hllc_scheme(tmp_path):
    """HLLC-only by design (M5 plan §1.4) -- a loud error, not a silent fallback."""
    text = _FULL.replace(
        'default = "closed"', 'default = "closed"\neast = { type = "fixed_stage", level = 3.0 }'
    )
    with pytest.raises(ConfigError, match="hllc_fv"):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize(
    "table,needle",
    [
        ('{ type = "fixed_stage" }', "exactly one of"),
        ('{ type = "fixed_stage", level = 1.0, stage = [[0.0, 1.0]] }', "exactly one of"),
        ('{ type = "fixed_stage", level = "high" }', "must be a number"),
        ('{ type = "fixed_stage", stage = [] }', "non-empty"),
        ('{ type = "fixed_stage", stage = [[10.0, 1.0], [0.0, 2.0]] }', "non-decreasing"),
        ('{ type = "tidal" }', "unknown boundary type"),
        ("42", "must be 'closed'"),
    ],
)
def test_fixed_stage_malformed(tmp_path, table, needle):
    text = _HLLC.replace('default = "closed"', f'default = "closed"\neast = {table}')
    with pytest.raises(ConfigError, match=needle):
        load_config(_write(tmp_path, text))


def test_fixed_stage_string_form_is_rejected_with_a_useful_message(tmp_path):
    """`east = "fixed_stage"` cannot work -- it carries no level. Say so."""
    text = _HLLC.replace('default = "closed"', 'default = "closed"\neast = "fixed_stage"')
    with pytest.raises(ConfigError, match="fixed_stage table"):
        load_config(_write(tmp_path, text))


def test_outlet_may_be_a_reach_box(tmp_path):
    """Spreading a release over a reach avoids the point-outlet delivery spike."""
    text = _FULL + _DAM.replace("outlet = [8, 4]", "outlet = [8, 2, 10, 6]")
    (s,) = load_config(_write(tmp_path, text)).structures
    assert s.outlet == (8, 2, 10, 6)


def test_outlet_overlapping_the_pool_is_rejected(tmp_path):
    text = _FULL + _DAM.replace(
        "outlet = [8, 4]", "outlet = [4, 2, 10, 6]"
    )  # rows 4.. hit the pool
    with pytest.raises(ConfigError, match="overlaps the pool"):
        load_config(_write(tmp_path, text))


def test_domain_selection_defaults_to_the_whole_mosaic(tmp_path):
    """M6: the domain is the tile set unless the scenario says otherwise."""
    scn = load_config(_write(tmp_path, _FULL))
    assert scn.tiles == "all"
    assert scn.window is None


def test_domain_selection_parses_tiles_and_window(tmp_path):
    text = _FULL.replace(
        'tiles_dir = "data/tiles/demo"',
        'tiles_dir = "data/tiles/demo"\ntiles = "first"\nwindow = [0, 0, 63, 127]',
    )
    scn = load_config(_write(tmp_path, text))
    assert scn.tiles == "first"
    assert scn.window == (0, 0, 63, 127)


def test_bad_domain_selection_is_rejected(tmp_path):
    text = _FULL.replace('tiles_dir = "data/tiles/demo"', 'tiles_dir = "d"\ntiles = "some"')
    with pytest.raises(ConfigError, match=r"\[grid\] tiles"):
        load_config(_write(tmp_path, text))

    text = _FULL.replace('tiles_dir = "data/tiles/demo"', 'tiles_dir = "d"\nwindow = [0, 0, 5]')
    with pytest.raises(ConfigError, match=r"\[grid\] window"):
        load_config(_write(tmp_path, text))

    text = _FULL.replace('tiles_dir = "data/tiles/demo"', 'tiles_dir = "d"\nwindow = [9, 0, 5, 5]')
    with pytest.raises(ConfigError, match="inclusive"):
        load_config(_write(tmp_path, text))


_CHANNELS = """
[channels]
width = 25.0
depth = 3.0
manning = 0.028
"""


def test_channels_parse_as_scalars_or_fields(tmp_path):
    scn = load_config(_write(tmp_path, _FULL + _CHANNELS))
    assert scn.has_channels
    assert scn.channel_width_m == 25.0
    assert scn.channel_depth_m == 3.0
    assert scn.channel_manning == 0.028
    assert scn.channel_width_field is None

    text = _FULL + '\n[channels]\nwidth = "w.r32"\ndepth = "d.r32"\n'
    scn = load_config(_write(tmp_path, text))
    assert scn.has_channels
    assert scn.channel_width_field.endswith("w.r32")
    assert scn.channel_manning is None  # inherits the floodplain roughness
    assert "channel_width" in scn.field_paths()


def test_no_channels_section_means_no_channels(tmp_path):
    assert not load_config(_write(tmp_path, _FULL)).has_channels


def test_channels_are_rejected_on_the_hllc_scheme(tmp_path):
    """Sub-grid channels are local-inertial-only (M6 plan §0) -- loudly."""
    text = _FULL.replace('scheme = "local_inertial"', 'scheme = "hllc_fv"') + _CHANNELS
    with pytest.raises(ConfigError, match="requires scheme='local_inertial'"):
        load_config(_write(tmp_path, text))


def test_a_channel_width_without_a_depth_is_rejected(tmp_path):
    text = _FULL + "\n[channels]\nwidth = 25.0\n"
    with pytest.raises(ConfigError, match="bank-full depth"):
        load_config(_write(tmp_path, text))


def test_coarsen_parses_and_validates(tmp_path):
    assert load_config(_write(tmp_path, _FULL)).coarsen == 1
    text = _FULL.replace('tiles_dir = "data/tiles/demo"', 'tiles_dir = "d"\ncoarsen = 4')
    assert load_config(_write(tmp_path, text)).coarsen == 4
    for bad in ("0", "2.5", "true"):
        text = _FULL.replace('tiles_dir = "data/tiles/demo"', f"tiles_dir = 'd'\ncoarsen = {bad}")
        with pytest.raises(ConfigError, match="coarsen"):
            load_config(_write(tmp_path, text))


# --- M7 morphology: the [sediment] stanza (plan build step 1) -------------------

_SEDIMENT = """
[sediment]
d50 = 0.008
porosity = 0.35
law = "mpm"
interval_s = 600.0
alluvium_thickness = 2.0
"""


def test_sediment_parses_every_field(tmp_path):
    scn = load_config(_write(tmp_path, _FULL + _SEDIMENT))
    assert scn.has_sediment
    assert scn.sediment_law == "mpm"
    assert scn.sediment_d50_m == 0.008
    assert scn.sediment_d50_field is None
    assert scn.sediment_porosity == 0.35
    assert scn.sediment_interval_s == 600.0
    assert scn.alluvium_thickness_m == 2.0
    assert scn.has_alluvium_floor


def test_no_sediment_section_means_no_morphology(tmp_path):
    scn = load_config(_write(tmp_path, _FULL))
    assert not scn.has_sediment
    assert scn.sediment_law is None
    assert not scn.has_alluvium_floor


def test_sediment_defaults_follow_the_reservoir_cadence(tmp_path):
    """An armed table needs only a grain size; the rest have defaults."""
    scn = load_config(_write(tmp_path, _FULL + "\n[sediment]\nd50 = 0.01\n"))
    assert scn.has_sediment
    assert scn.sediment_law == "mpm"
    assert scn.sediment_porosity == 0.4
    assert scn.sediment_interval_s == 900.0
    assert scn.alluvium_thickness_m is None  # unlimited alluvium, not zero thickness


def test_sediment_fields_parse_as_paths_and_are_hashed(tmp_path):
    text = _FULL + '\n[sediment]\nd50 = "grain.r32"\nalluvium_thickness = "allu.r32"\n'
    scn = load_config(_write(tmp_path, text))
    assert scn.sediment_d50_field.endswith("grain.r32")
    assert scn.alluvium_thickness_field.endswith("allu.r32")
    assert scn.has_alluvium_floor
    # Provenance hashes every referenced field file (solver/io/provenance.py).
    assert "sediment_d50" in scn.field_paths()
    assert "alluvium_thickness" in scn.field_paths()


def test_sediment_is_rejected_on_the_hllc_scheme(tmp_path):
    """Morphology is local-inertial-only (M7 plan §1.4) -- loudly, naming both."""
    text = _FULL.replace('scheme = "local_inertial"', 'scheme = "hllc_fv"') + _SEDIMENT
    with pytest.raises(ConfigError, match="requires scheme='local_inertial'"):
        load_config(_write(tmp_path, text))


def test_an_armed_sediment_table_without_a_grain_size_is_rejected(tmp_path):
    """The table arms morphology, so there is no d50 that quietly means 'off'."""
    with pytest.raises(ConfigError, match="needs a grain size"):
        load_config(_write(tmp_path, _FULL + "\n[sediment]\nporosity = 0.4\n"))
    with pytest.raises(ConfigError, match="needs a grain size"):
        load_config(_write(tmp_path, _FULL + "\n[sediment]\nd50 = 0.0\n"))


def test_a_mistyped_sediment_key_warns_and_still_fails_loudly(tmp_path):
    """The typo guard is why arming is table presence, not `d50 > 0`.

    Unknown keys only *warn*, so arming on a positive d50 would let one mistyped
    key turn a morphology run into a flood run with nothing raised anywhere.
    """
    text = _FULL + "\n[sediment]\nd_50 = 0.008\n"
    with pytest.warns(UserWarning, match="unknown key 'd_50'"):
        with pytest.raises(ConfigError, match="needs a grain size"):
            load_config(_write(tmp_path, text))


def test_sediment_scalars_are_validated(tmp_path):
    for bad, match in (
        ("porosity = 0.0", r"porosity must be in \(0, 1\)"),
        ("porosity = 1.0", r"porosity must be in \(0, 1\)"),
        ("interval_s = 0.0", "interval_s must be > 0"),
        ("alluvium_thickness = -1.0", "alluvium_thickness must be >= 0"),
        ('law = "engelund_hansen"', "not a transport law M7 ships"),
        ('porosity = "wet"', r"\[sediment\] porosity must be a number"),
    ):
        text = _FULL + f"\n[sediment]\nd50 = 0.008\n{bad}\n"
        with pytest.raises(ConfigError, match=match):
            load_config(_write(tmp_path, text))
