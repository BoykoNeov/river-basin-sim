"""Run provenance / command log (M3, HANDOFF §2, §7.4 -- the reproducibility half).

HANDOFF §2: "Scenario = config + parameter fields + command log; deterministic
stepping" -> a run is fully reproducible and shareable. This module captures the
static provenance of a run: the source TOML, the fully-resolved :class:`Scenario`
(after manifest inheritance), and the **sha256 of the config and of every
referenced field file**. Together with the solver's determinism (§8/§12), that
record is enough to reproduce or diff a run byte-for-byte.

The record is written to the canonical Zarr ``.zattrs`` (so it travels with the
results) and to a sidecar ``<store>.provenance.json`` (so it is readable without
opening Zarr). Live in-run edits at scheduler sync points are an M5 concern; this
is the static half that M3 needs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from solver.io.config import Scenario

SOLVER_VERSION = "0.1.0"  # keep in step with pyproject [project].version
MILESTONE = "M3"


def sha256_file(path: str | Path) -> str | None:
    """Hex sha256 of a file's bytes, or ``None`` if it does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_provenance(scenario: Scenario) -> dict:
    """Assemble the provenance record for a resolved scenario (JSON-serializable)."""
    source = scenario.source_path
    return {
        "solver": {"version": SOLVER_VERSION, "milestone": MILESTONE, "scheme": "local_inertial"},
        "source_toml": source,
        "source_sha256": sha256_file(source) if source else None,
        "field_sha256": {role: sha256_file(path) for role, path in scenario.field_paths().items()},
        "resolved_scenario": asdict(scenario),
    }


def write_provenance(scenario: Scenario, out_path: str | Path) -> dict:
    """Write ``<out_path>.provenance.json`` next to the store; return the record."""
    record = build_provenance(scenario)
    out_path = Path(out_path)
    sidecar = out_path.with_name(out_path.name + ".provenance.json")
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record
