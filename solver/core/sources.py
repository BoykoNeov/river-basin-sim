"""Compensated (Kahan) accumulation of **areal** sources (HANDOFF §8, §12).

Rain is a distributed source: every wet cell in the domain receives ``rate*dt``
metres, every step, for as long as the storm lasts. That addition is where a
rain-on-grid run at reach scale loses its mass balance -- not in the physics, and
not in the flux divergence, but in float32 arithmetic:

    h[i,j] = h[i,j] + rate*dt        # float32, once per cell per step

At M6's reach demo the increment is ``rate*dt ~ 8e-5`` m onto a depth already at
``0.1..1.4`` m. ``eps(1.0)`` in float32 is ``1.2e-7``, so each add discards a few
tens of nanometres, the discarded part does **not** average to zero over the
population, and a couple of hundred thousand cells x a few hundred storm steps
turns that into a measurable volume. Measured on ``scenarios/reach_basin.toml`` at
``coarsen = 4``: the residual climbs ~220 m^3 per 1800 s **while it rains**, reaches
670 m^3, then goes essentially flat over the following 10 h with the storm off even
though water is still moving and draining. Fluxes are not the leak; the source add
is. (The same run with sub-grid channels disabled drifts identically, which is what
told M6 this was arithmetic rather than new code.)

**The fix is the ledger's own idiom, moved onto the grid.**
:class:`solver.core.massbalance._Kahan` already protects the host-side float64
accumulator with compensated summation. Here each cell carries its own float32
compensation term ``comp``, holding the low-order bits the last source add threw
away, which the next add pays back::

    y    = rate*dt - comp[i,j]
    t    = h[i,j] + y
    comp = (t - h[i,j]) - y          # what did not fit
    h    = t

``h`` stays float32 (§2 is untouched -- no field is promoted), and the state grows
by exactly one float32 array, armed only when an areal source exists.

**Arming, and what stays bitwise.** ``State.h_comp`` is armed by
:meth:`solver.core.state.State.arm_source_compensation`, which the run loop calls
only for a scenario carrying uniform rain or a rain field. With it unarmed the
schemes launch their original kernels, so every run without an areal source --
dam-break, lake-at-rest, the EA benchmarks, M5's ``reservoir_release`` -- is
bitwise unchanged. Rain-bearing runs *do* change, by design: uncompensated
accumulation was a defect, not a contract (the same call M4 made when the
conservative donor-beta limiter replaced the non-conservative ``max(h, 0)`` clamp).

**Scope: areal sources.** This module owns the *distributed* case -- rain here,
M7's sediment transport integral through :func:`kahan_add`. Inflow hydrographs were
scoped out when this landed, on evidence that they were ~1.3% of the reach-demo
residual; that measurement was taken on a **rain-driven** run and did not transfer
to a flood-driven one, so :mod:`solver.processes.inflow` now carries its own
per-entry compensation through :func:`kahan_add` as well (2026-08-17). It owns that
array rather than arming ``State.h_comp``, because the schemes dispatch their
*areal* source kernels on ``h_comp`` and a point source has no business moving a
rain-free run onto a different code path.

**Sub-ULP caveat.** The compensation term is exact under Fast2Sum's condition
``|h| >= |y|``. A cell drier than the increment itself (``h < ~1e-4`` m during the
first steps of a storm on a dry basin) violates that and its ``comp`` is then only
approximate -- but at that depth the discarded quantity is below ``1e-11`` m, which
is ten thousand times smaller than the drift this module exists to remove.
"""

from __future__ import annotations

import warp as wp


@wp.func
def kahan_add(total: wp.float32, comp: wp.float32, inc: wp.float32) -> wp.vec2f:
    """One compensated add: returns ``(new_total, new_comp)``.

    The four lines every accumulator in this module runs, behind one name, so a
    distributed quantity added into a float32 field has exactly one definition of
    "how" -- M7's sediment transport integral accumulates through here too
    (:mod:`solver.core.sediment`), which is the carried requirement M6 left: *any*
    new distributed source goes through this module rather than a bare ``+=``.

    ``comp`` is the running debt of low-order bits and is **subtracted** from the
    next increment, so a caller must carry it across steps for the compensation to
    mean anything.
    """
    y = inc - comp
    t = total + y
    return wp.vec2f(t, (t - total) - y)


@wp.kernel
def add_uniform_rain_c(
    h: wp.array2d(dtype=wp.float32),
    comp: wp.array2d(dtype=wp.float32),
    rate: wp.float32,
    dt: wp.float32,
):
    """Kahan-add a uniform ``rate*dt`` to every cell, carrying ``comp`` forward.

    ``comp`` persists across steps: it is the running debt of low-order bits, so
    once the storm stops (``rate == 0``) the kernel keeps being launched with a
    zero increment and simply pays the outstanding remainder back into ``h``.
    """
    i, j = wp.tid()
    out = kahan_add(h[i, j], comp[i, j], rate * dt)
    h[i, j] = out[0]
    comp[i, j] = out[1]


@wp.kernel
def add_rain_field_c(
    h: wp.array2d(dtype=wp.float32),
    comp: wp.array2d(dtype=wp.float32),
    rain: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    scale: wp.float32,
):
    """Kahan-add a spatial rainfall field ``rain[i,j]*scale*dt`` (M3 field rain).

    ``scale`` is the temporal on/off multiplier, exactly as in the uncompensated
    :func:`solver.core.local_inertial.apply_rain_field`; the spatial pattern is
    static, so the ledger's inflow stays analytic.
    """
    i, j = wp.tid()
    out = kahan_add(h[i, j], comp[i, j], rain[i, j] * scale * dt)
    h[i, j] = out[0]
    comp[i, j] = out[1]


def apply_uniform_rain(state, rain: float, dt: float) -> None:
    """Add uniform rain to ``state.h``, compensated when ``h_comp`` is armed.

    Callers that fuse the uniform source into their continuity kernel (the
    local-inertial scheme) must pass ``rain = 0`` to that kernel when compensation
    is armed and call this instead, so the source add is the one this module owns.
    """
    if state.h_comp is None:
        raise ValueError("apply_uniform_rain requires arm_source_compensation()")
    wp.launch(
        add_uniform_rain_c,
        dim=state.grid.shape,
        inputs=[state.h, state.h_comp, wp.float32(rain), wp.float32(dt)],
        device=state.device,
    )


def apply_rain_field(state, dt: float, scale: float) -> None:
    """Add the spatial rain field to ``state.h``, compensated (``h_comp`` armed)."""
    if state.h_comp is None:
        raise ValueError("apply_rain_field requires arm_source_compensation()")
    wp.launch(
        add_rain_field_c,
        dim=state.grid.shape,
        inputs=[state.h, state.h_comp, state.rain, wp.float32(dt), wp.float32(scale)],
        device=state.device,
    )
