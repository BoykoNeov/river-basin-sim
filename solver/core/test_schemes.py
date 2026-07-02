"""Scheme dispatch tests (M4, plan §1.1)."""

from __future__ import annotations

import pytest

from solver.core import local_inertial
from solver.core.schemes import KNOWN_SCHEMES, get_scheme


def test_local_inertial_dispatches_to_the_li_module():
    """get_scheme returns the LI module, exposing the compute_dt/step pair."""
    scheme = get_scheme("local_inertial")
    assert scheme is local_inertial
    assert callable(scheme.compute_dt)
    assert callable(scheme.step)


def test_hllc_is_known_but_not_yet_implemented():
    """hllc_fv is a known scheme (config accepts it) but stubbed until wired up."""
    assert "hllc_fv" in KNOWN_SCHEMES
    with pytest.raises(NotImplementedError, match="hllc_fv"):
        get_scheme("hllc_fv")


def test_unknown_scheme_is_a_value_error():
    with pytest.raises(ValueError, match="unknown scheme"):
        get_scheme("nope")
