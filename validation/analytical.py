"""Analytical dam-break references (M1 validation, HANDOFF §10).

Closed-form 1-D shallow-water solutions on a flat, frictionless bed, dam at
``x = 0``, still water either side at ``t = 0``. Used to check the local-inertial
solver's wave shape (a *loose* gate -- LI drops advective acceleration and cannot
reproduce a shock exactly; see ``docs/plans/M1-water-moves.md`` §5).

* :func:`stoker` -- **wet bed** downstream (``h_R > 0``): rarefaction fan on the
  left, a constant "star" region, and a bore (shock) propagating right. The
  enforced-gate case.
* :func:`ritter` -- **dry bed** downstream (``h_R = 0``): a single rarefaction
  running out onto the dry bed, no shock. The diagnostic case (its wetting front
  is LI's hardest regime).

No SciPy dependency: the Stoker star-depth root is found by bisection.
"""

from __future__ import annotations

import math

import numpy as np

G = 9.81


def _stoker_star_depth(h_l: float, h_r: float, g: float = G, tol: float = 1e-12) -> float:
    """Bisect for the star (middle) depth ``h_m`` in a wet-bed dam break.

    Matches the rarefaction and shock expressions for the star velocity::

        u_rare(h_m)  = 2 (sqrt(g h_l) - sqrt(g h_m))
        u_shock(h_m) = (h_m - h_r) sqrt( (g/2)(h_m + h_r)/(h_m h_r) )

    ``f = u_rare - u_shock`` is positive at ``h_r`` and negative at ``h_l``, so a
    unique root lies between.
    """

    def f(h_m: float) -> float:
        u_rare = 2.0 * (math.sqrt(g * h_l) - math.sqrt(g * h_m))
        u_shock = (h_m - h_r) * math.sqrt(0.5 * g * (h_m + h_r) / (h_m * h_r))
        return u_rare - u_shock

    lo, hi = h_r, h_l
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def stoker(x: np.ndarray, t: float, h_l: float, h_r: float, g: float = G) -> tuple:
    """Wet-bed dam-break depth and velocity at coordinates ``x`` (dam at 0), time ``t``.

    Returns ``(h, u)`` arrays. ``h_r > 0`` required.
    """
    if h_r <= 0.0:
        raise ValueError("stoker requires a wet downstream bed (h_r > 0); use ritter")
    x = np.asarray(x, dtype=np.float64)
    if t <= 0.0:
        return np.where(x < 0.0, h_l, h_r), np.zeros_like(x)

    c_l = math.sqrt(g * h_l)
    h_m = _stoker_star_depth(h_l, h_r, g)
    c_m = math.sqrt(g * h_m)
    u_m = 2.0 * (c_l - c_m)
    # Shock speed from mass conservation across the bore.
    s = h_m * u_m / (h_m - h_r)

    xi = x / t  # self-similar variable
    fan_tail = u_m - c_m  # speed of the rarefaction tail (meets the star region)

    h = np.empty_like(x)
    u = np.empty_like(x)

    left = xi <= -c_l
    fan = (xi > -c_l) & (xi < fan_tail)
    star = (xi >= fan_tail) & (xi < s)
    right = xi >= s

    h[left], u[left] = h_l, 0.0
    # Rarefaction fan: u = (2/3)(xi + c_l); c = (1/3)(2 c_l - xi); h = c^2/g.
    u_fan = (2.0 / 3.0) * (xi[fan] + c_l)
    c_fan = (1.0 / 3.0) * (2.0 * c_l - xi[fan])
    h[fan], u[fan] = c_fan * c_fan / g, u_fan
    h[star], u[star] = h_m, u_m
    h[right], u[right] = h_r, 0.0
    return h, u


def ritter(x: np.ndarray, t: float, h_l: float, g: float = G) -> tuple:
    """Dry-bed dam-break (Ritter 1892) depth and velocity at ``x`` (dam at 0), time ``t``.

    Returns ``(h, u)``. Single rarefaction onto a dry bed; front at ``xi = 2 c_l``.
    """
    x = np.asarray(x, dtype=np.float64)
    if t <= 0.0:
        return np.where(x < 0.0, h_l, 0.0), np.zeros_like(x)

    c_l = math.sqrt(g * h_l)
    xi = x / t
    h = np.empty_like(x)
    u = np.empty_like(x)

    left = xi <= -c_l
    fan = (xi > -c_l) & (xi < 2.0 * c_l)
    dry = xi >= 2.0 * c_l

    h[left], u[left] = h_l, 0.0
    h[fan] = (1.0 / (9.0 * g)) * (2.0 * c_l - xi[fan]) ** 2
    u[fan] = (2.0 / 3.0) * (c_l + xi[fan])
    h[dry], u[dry] = 0.0, 0.0
    return h, u


def stoker_front_position(t: float, h_l: float, h_r: float, g: float = G) -> float:
    """x-position of the wet-bed shock front at time ``t`` (dam at 0)."""
    h_m = _stoker_star_depth(h_l, h_r, g)
    c_l = math.sqrt(g * h_l)
    c_m = math.sqrt(g * h_m)
    u_m = 2.0 * (c_l - c_m)
    s = h_m * u_m / (h_m - h_r)
    return s * t
