"""status.json writer tests (M2, HANDOFF §7.4)."""

from __future__ import annotations

import json

import pytest

from solver.io.status import StatusWriter


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_state_sequence_and_shape(tmp_path):
    p = tmp_path / "status.json"
    sw = StatusWriter(p, end_time=1000.0)

    sw.write("starting", message="go")
    rec = _read(p)
    assert set(rec) == {"state", "progress", "sim_time", "eta_s", "message"}
    assert rec["state"] == "starting" and rec["progress"] == 0.0

    sw.write("running", sim_time=500.0)
    rec = _read(p)
    assert rec["state"] == "running"
    assert rec["progress"] == pytest.approx(0.5)
    assert rec["sim_time"] == 500.0

    sw.write("done", sim_time=1000.0)
    rec = _read(p)
    assert rec["state"] == "done" and rec["progress"] == 1.0


def test_progress_clamped(tmp_path):
    p = tmp_path / "status.json"
    sw = StatusWriter(p, end_time=100.0)
    sw.write("running", sim_time=250.0)  # overshoot
    assert _read(p)["progress"] == 1.0


def test_error_state_carries_message(tmp_path):
    p = tmp_path / "status.json"
    sw = StatusWriter(p, end_time=100.0)
    sw.write("error", message="ValueError: boom")
    rec = _read(p)
    assert rec["state"] == "error" and "boom" in rec["message"]


def test_invalid_state_rejected(tmp_path):
    sw = StatusWriter(tmp_path / "status.json", end_time=100.0)
    with pytest.raises(ValueError, match="invalid status state"):
        sw.write("frobnicating")


def test_no_leftover_temp_file(tmp_path):
    p = tmp_path / "status.json"
    sw = StatusWriter(p, end_time=100.0)
    sw.write("done", sim_time=100.0)
    # Atomic replace leaves only the final file, no .tmp sidecar.
    assert p.exists()
    assert not p.with_name(p.name + ".tmp").exists()
