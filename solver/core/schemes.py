"""Scheme registry / dispatch (M4, plan §1.1 -- the "same kernel interface" seam).

The run loop (:mod:`solver.run`) is **scheme-agnostic**: it only ever calls a pair
of scheme-owned functions on the state,

    compute_dt(state, alpha=..., dt_max=...) -> float   # scheme-owned CFL
    step(state, dt=..., rain=..., rain_scale=...) -> None  # scheme-owned update

``get_scheme(name)`` maps a scenario's ``scheme`` string to the module (or object)
that provides that pair. The two schemes coexist by selection (HANDOFF §2):

  * ``local_inertial`` -- the M1 Bates local-inertial scheme, the **permanent
    coverage** scheme for lowland floodplains (staggered ``qx/qy`` faces).
  * ``hllc_fv`` -- the M4 well-balanced Godunov HLLC finite-volume scheme, the
    **fidelity** option for shocks / transcritical / well-balanced wet-dry
    (cell-centred conservative ``hu/hv``).

Each scheme owns its own timestep formula and its own boundary handling, so the
run loop never branches on scheme. ``alpha`` is the CFL-like coefficient the
scenario carries (TOML ``[run] cfl``): ~0.7 for LI (Bates bound), ~0.4-0.5 for
HLLC (``C * dx / (|u| + sqrt(g h))``) -- each scheme interprets it per its own
stability limit.
"""

from __future__ import annotations

from types import ModuleType

from solver.core import local_inertial

# Scheme names the config schema recognises (config.py validates against this;
# availability -- whether a known scheme is actually implemented yet -- is decided
# here in get_scheme, so an unimplemented-but-known scheme is a NotImplementedError
# at dispatch, not a config scope-gate error).
KNOWN_SCHEMES = ("local_inertial", "hllc_fv")


def get_scheme(name: str) -> ModuleType:
    """Return the module providing ``compute_dt``/``step`` for scheme ``name``.

    Raises :class:`NotImplementedError` for a *known* scheme that is not wired up
    yet (an availability stub, not a scope-gate error), and :class:`ValueError`
    for an unknown scheme name.
    """
    if name == "local_inertial":
        return local_inertial
    if name == "hllc_fv":
        raise NotImplementedError(
            "scheme='hllc_fv' (well-balanced HLLC finite volume) is not yet "
            "implemented; M4 is building it. Use 'local_inertial' for now."
        )
    raise ValueError(f"unknown scheme '{name}'; known schemes: {list(KNOWN_SCHEMES)}")
