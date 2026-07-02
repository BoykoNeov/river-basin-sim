"""Field loader tests (M3, HANDOFF §7.1)."""

from __future__ import annotations

import numpy as np
import pytest

from solver.core.grid import Grid
from solver.io.fields import load_field, load_r32

GRID = Grid(ny=4, nx=5, dx=10.0)


def test_scalar_and_none_broadcast():
    a = load_field(None, GRID, scalar=0.035)
    assert a.shape == (4, 5) and a.dtype == np.float32
    assert np.all(a == np.float32(0.035))
    b = load_field(0.02, GRID)
    assert np.all(b == np.float32(0.02))


def test_r32_roundtrip_orientation(tmp_path):
    # A field with distinct per-cell values so any transpose/flip would show.
    src = (np.arange(20, dtype=np.float32) * 0.001).reshape(4, 5)
    p = tmp_path / "n.r32"
    src.astype("<f4").tofile(p)
    out = load_field(str(p), GRID)
    assert out.shape == (4, 5)
    assert np.array_equal(out, src)  # row-major, no transpose


def test_r32_wrong_size_is_hard_error(tmp_path):
    p = tmp_path / "bad.r32"
    np.arange(19, dtype="<f4").tofile(p)  # 19 != 20
    with pytest.raises(ValueError, match="expected 20"):
        load_r32(p, GRID)


def test_bool_not_treated_as_field():
    # bool is an int subclass -- must not be silently broadcast as 0/1.
    with pytest.raises((ValueError, OSError)):
        load_field(True, GRID)  # falls through to path handling -> fails


def test_non_finite_field_is_hard_error(tmp_path):
    src = np.full((4, 5), 0.04, dtype=np.float32)
    src[1, 2] = np.nan
    p = tmp_path / "nan.r32"
    src.astype("<f4").tofile(p)
    with pytest.raises(ValueError, match="non-finite"):
        load_field(str(p), GRID, name="manning_n")


def test_negative_field_rejected_when_nonneg(tmp_path):
    # A negative rain/infiltration rate stays ledger-consistent (the sink cancels),
    # so the mass gate can't catch it -- load_field must.
    src = np.full((4, 5), 5.0, dtype=np.float32)
    src[0, 0] = -1.0
    p = tmp_path / "neg.r32"
    src.astype("<f4").tofile(p)
    with pytest.raises(ValueError, match="negative"):
        load_field(str(p), GRID, name="rainfall", nonneg=True)
    # Without nonneg the same field loads (negatives are legitimate for e.g. bed).
    assert load_field(str(p), GRID).shape == (4, 5)
