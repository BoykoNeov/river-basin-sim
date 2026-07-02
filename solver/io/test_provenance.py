"""Provenance / command-log tests (M3, HANDOFF §2)."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np

from solver.io.config import load_config
from solver.io.provenance import SOLVER_VERSION, build_provenance, sha256_file, write_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]

_CFG = """
[meta]
name = "prov_test"
scheme = "local_inertial"

[grid]
tiles_dir = "data/tiles/demo"
dx = 30.0

[run]
end_time = 600.0
output_every = 300.0

[rainfall]
type = "uniform"
rate_mm_hr = 20.0
duration_s = 300.0

[parameters]
manning_n = "fields/n.r32"

[boundaries]
default = "closed"
east = "open"
"""


def _setup(tmp_path):
    (tmp_path / "fields").mkdir()
    field = tmp_path / "fields" / "n.r32"
    np.full(16, 0.04, dtype="<f4").tofile(field)
    cfg = tmp_path / "s.toml"
    cfg.write_text(_CFG, encoding="utf-8")
    return cfg, field


def test_sha256_matches_hashlib(tmp_path):
    _, field = _setup(tmp_path)
    expect = hashlib.sha256(field.read_bytes()).hexdigest()
    assert sha256_file(field) == expect
    assert sha256_file(tmp_path / "nope.r32") is None


def test_provenance_captures_source_and_field_hashes(tmp_path):
    cfg, field = _setup(tmp_path)
    scn = load_config(cfg)
    rec = build_provenance(scn)

    assert rec["source_toml"] == str(cfg)
    assert rec["source_sha256"] == hashlib.sha256(cfg.read_bytes()).hexdigest()
    # The manning field was referenced -> hashed under its role.
    assert rec["field_sha256"]["manning"] == hashlib.sha256(field.read_bytes()).hexdigest()
    # The resolved scenario is complete and JSON-serializable.
    resolved = rec["resolved_scenario"]
    assert resolved["name"] == "prov_test"
    assert resolved["boundaries"]["east"] == "open"
    assert rec["solver"]["milestone"] == "M3"
    json.dumps(rec)  # must not raise


def test_solver_version_matches_pyproject():
    """SOLVER_VERSION is stamped into every run's provenance -- pin it to the single
    source of truth (pyproject [project].version) so the two can't silently drift.

    Read pyproject directly rather than importlib.metadata: the project runs via
    pythonpath, not as an installed distribution, so metadata lookup would fail.
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert SOLVER_VERSION == pyproject["project"]["version"]


def test_write_provenance_sidecar(tmp_path):
    cfg, _ = _setup(tmp_path)
    scn = load_config(cfg)
    out = tmp_path / "results.zarr"
    rec = write_provenance(scn, out)
    sidecar = tmp_path / "results.zarr.provenance.json"
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == rec
