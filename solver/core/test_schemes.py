"""Scheme dispatch tests (M4, plan §1.1)."""

from __future__ import annotations

import pytest

from solver.core import hllc, local_inertial
from solver.core.schemes import KNOWN_SCHEMES, get_scheme


def test_local_inertial_dispatches_to_the_li_module():
    """get_scheme returns the LI module, exposing the compute_dt/step pair."""
    scheme = get_scheme("local_inertial")
    assert scheme is local_inertial
    assert callable(scheme.compute_dt)
    assert callable(scheme.step)


def test_hllc_dispatches_to_the_hllc_module():
    """hllc_fv is a known scheme dispatching to the HLLC FV module (M4)."""
    assert "hllc_fv" in KNOWN_SCHEMES
    scheme = get_scheme("hllc_fv")
    assert scheme is hllc
    assert callable(scheme.compute_dt)
    assert callable(scheme.step)


def test_unknown_scheme_is_a_value_error():
    with pytest.raises(ValueError, match="unknown scheme"):
        get_scheme("nope")
