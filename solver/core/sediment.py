"""Bedload transport at capacity + the Exner bed update (M7, HANDOFF §9).

The bed stops being static: sediment is entrained where the flow can carry it,
deposited where it cannot, and the difference moves ``z``. This module is the
arithmetic -- the transport law, the face flux, the divergence and the bed rebuild.
It owns no schedule and no state; :mod:`solver.processes.morphology` (M7 build
step 5) drives it on the slow clock, the way :mod:`solver.processes.reservoir`
drives its release rule.

**Transport is at capacity, instantaneously, everywhere** (Meyer-Peter-Mueller).
There is no suspended load, no washload and no non-equilibrium adaptation length,
so a bed feature sharpens faster and stays crisper than a real one, and fines
cannot travel through a reach. That is what "Exner + transport capacity" means and
it is a stated simplification, not an oversight (M7 plan §0).

Law (per unit width, magnitude)::

    theta = tau / (rho * s' * g * d50)          Shields number
    q_s   = 8 * (theta - theta_c)^1.5 * sqrt(s' * g * d50^3)     theta > theta_c
          = 0                                                    otherwise

with ``s' = 1.65`` (submerged specific gravity of quartz) and ``theta_c = 0.047``.
The direction is the flow's, so on a face ``q_s`` carries ``sign(q)``.

**Shear is the Manning shear the scheme already computes**, and ``rho`` cancels out
of ``theta`` entirely -- so this module needs no water density::

    tau/rho = g * n^2 * q^2 / (h^2 * R^(1/3))

which at ``R = h`` (wide floodplain) is the familiar ``g n^2 q^2 / h^(7/3)``, the
same term :func:`solver.core.friction.manning_denominator` carries, and at
``R = A/P`` is the channel form of
:func:`solver.core.friction.manning_denominator_radius`. It is written out here
rather than recovered as ``(D - 1)/dt`` from those functions, because that
difference cancels catastrophically in float32 for small ``dt`` (``D`` is ``1 +
epsilon``); ``test_sediment.test_shear_is_the_manning_denominators_own_friction_term``
ties the two together instead, so there is still exactly one Manning form and it is
checked rather than assumed.

**A channel face's transport uses the channel's own section.** ``h_ch`` and
``R = A/P`` come from :mod:`solver.core.channels`' face-geometry functions -- the
*same* ones the momentum update calls -- never the cell-mean ``h`` and never
:func:`~solver.core.channels.column_depth`. Those differ by up to ``dx/w`` (~15x on
the reach demo) and ``q_s`` goes as roughly ``theta^1.5``, so getting it wrong in
either direction is the single most likely physics bug in this milestone (M7 plan
§4). The two components recombine into a per-cell-width flux exactly as the
discharges do, ``q_s = q_s_fp*(1-frac) + q_s_ch*frac`` with ``frac = w_face/dx``.

**Face ``d50`` is the mean of its two cells** -- following the ``n`` idiom, so a
uniform grain size stays bit-exact. Like ``coarsen.py``'s block mean it is an
*engineering choice*, not a conservation law: transport goes as ``d50^1.5``, so the
mean of two grain sizes is not the grain size of their mixture.

**Where the accumulation hook goes, and it is not a free choice.** ``q_s`` must be
evaluated from the **post-limiter** face discharge, with ``eta`` still current and
``h`` not yet advanced -- i.e. after ``limit_qx``/``limit_qy`` and before
``update_h`` in :func:`solver.core.local_inertial.step`. The pre-limiter discharge
is not the water that moved (in steep cells the donor limiter rescales it hard),
and after continuity ``eta`` no longer matches the fluxes that produced it. The
accumulation kernels are launched over *interior* faces exactly as the momentum
kernels are, so the caller owes them the same ``nx > 1`` / ``ny > 1`` guards.

**Two accumulators, two different arithmetic decisions** (M7 plan §1.1, §1.3):

* ``qs_int`` (faces, **float32 + Kahan** via :func:`solver.core.sources.kahan_add`)
  gains ``q_s*dt`` every fast step. Its increments are the same order as the sum
  over one interval, which is exactly the regime compensation was built for.
* ``dz_cum`` (cells, **float64**) holds the cumulative bed change, and ``z`` is
  *recomputed* as ``float32(z0 + dz_cum)`` -- never incremented in place. A
  morphological increment is sub-millimetre while ``z`` is O(100 m), so
  ``eps(z) ~ 1.5e-5 m`` swallows it: float32 ``z += dz`` is not drift, it is a term
  silently deleted from the physics. The precedent is ``loss_cum``, not ``h_comp``:
  a grid-sized float64 *ledger* sits outside HANDOFF §2's float32 stepping fields.

**``z0`` is the bed after barriers, not the DEM.** ``z`` is rebuilt from it every
activation, so a ``z0`` captured before
:func:`solver.processes.reservoir.apply_barriers` would delete every dam at the
first rebuild. Capture it from ``state.z`` once the state is fully built.

**Limits are banked, never clamped away.** Structure cells (a dam is engineered,
not alluvial) and an alluvium floor ("do not erode below bedrock") both arrive as
one mechanism: per-cell bounds ``dz_lo``/``dz_hi`` on the *cumulative* change, with
the unapplied part accumulated in metres into ``dz_unapplied`` for the sediment
ledger to convert at ``A*(1-p)``. Silently clamping would invent or destroy solid
mass exactly the way a bare ``max(h, 0)`` invents water (M4).

**Carried limitations**, each real and each declared:

* *The channel section is frozen.* Exner moves ``z``; the invert ``z - d``
  translates with it and ``(w, d)`` do not change. A channel that aggrades rises
  bodily rather than filling in. Evolving ``d`` would change the storage curve's
  shape mid-run and re-open M6's CFL derivation.
* *Water is untouched by a bed update.* ``h`` is volume per unit plan area and is
  not read here, so ``eta = z + h`` simply rises with the bed and water volume is
  conserved by construction. There is deliberately **no** special rule for a cell
  that deposits its way dry -- the existing ``H_DRY`` guard already decides what
  dry means everywhere else (M7 plan §1.6).
* *Shear on a face uses that face's own normal discharge.* ``theta`` goes as
  ``q^2``, so flow at 45 degrees reads roughly half the Shields number in each
  component and can fall below threshold when the true flow is transporting. This
  is the staggered scheme's own approximation -- the friction denominator makes it
  too -- but it means a threshold or celerity fixture must be **axis-aligned** to
  measure the law rather than the projection.
* *Sediment cannot leave through an open boundary.* Boundary faces are never
  updated by the momentum kernels (they stay zero, which is the closed BC) and the
  local-inertial open boundary is a post-interior sink on the edge *cell*, not a
  face. So water leaves and its load does not, and an outlet cell aggrades. Keep a
  measurement clear of the outlet, or read the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from solver.core.channels import (
    channel_flow_depth,
    channel_radius,
    face_channel_bed,
    face_channel_width,
    floodplain_flow_depth,
)
from solver.core.grid import GRAVITY, H_DRY
from solver.core.sources import kahan_add

# Submerged specific gravity (rho_s - rho_w)/rho_w for quartz sand/gravel.
SUBMERGED_SG = 1.65
# Critical Shields number for incipient motion (Meyer-Peter-Mueller's value).
SHIELDS_CRITICAL = 0.047
# MPM's coefficient and exponent on the excess Shields stress.
MPM_COEFFICIENT = 8.0
MPM_EXPONENT = 1.5


# --- The law -----------------------------------------------------------------
# `mpm_capacity` is the only place the transport law appears, so a second law is an
# addition rather than a rewrite. M7 ships one, and the bed-wave celerity gate is
# derived for *that* one (M7 plan §1.2): gating against an external flume dataset
# that cannot be verified from here would repeat M5's EA Test 1 cost.


@wp.func
def kinematic_shear(q: wp.float32, h: wp.float32, radius: wp.float32, n: wp.float32) -> wp.float32:
    """Bed shear per unit fluid density, ``tau/rho`` (m^2/s^2), from Manning.

    ``q`` is discharge per unit width (m^2/s) through the face, ``h`` the flow depth
    that carries it and ``radius`` its hydraulic radius (``= h`` in the wide case).
    """
    return GRAVITY * n * n * q * q / (h * h * wp.pow(radius, 1.0 / 3.0))


@wp.func
def shields_number(
    q: wp.float32, h: wp.float32, radius: wp.float32, n: wp.float32, d50: wp.float32
) -> wp.float32:
    """Shields number ``theta = tau / (rho s' g d50)`` -- density-free by cancellation."""
    return kinematic_shear(q, h, radius, n) / (SUBMERGED_SG * GRAVITY * d50)


@wp.func
def mpm_capacity(theta: wp.float32, d50: wp.float32) -> wp.float32:
    """Meyer-Peter-Mueller bedload capacity per unit width (m^2/s), magnitude only.

    Below the critical Shields number this returns a **bit-exact zero**, which is
    what makes "no motion => no bed change" a cheap, sharp assertion: it catches a
    sign error, a units error, or a velocity-independent term wired in by accident.
    """
    if theta <= SHIELDS_CRITICAL:
        return 0.0
    excess = theta - SHIELDS_CRITICAL
    return (
        MPM_COEFFICIENT
        * wp.pow(excess, MPM_EXPONENT)
        * wp.sqrt(SUBMERGED_SG * GRAVITY * d50 * d50 * d50)
    )


@wp.func
def face_capacity(
    q: wp.float32, h: wp.float32, radius: wp.float32, n: wp.float32, d50: wp.float32
) -> wp.float32:
    """Signed bedload flux (m^2/s) on one face: capacity carrying ``sign(q)``."""
    if h < H_DRY or d50 <= 0.0 or q == 0.0:
        return 0.0
    qs = mpm_capacity(shields_number(q, h, radius, n, d50), d50)
    if q < 0.0:
        return -qs
    return qs


# --- Accumulating the transport integral -------------------------------------
# The fast loop accumulates and the slow clock applies (M7 plan §1.3). Sampling
# `q_s` at the activation instant and scaling by the interval would misrepresent a
# passing flood wave badly in both directions, since `q_s` goes as roughly
# `h^1.5 S_f^1.5`. What the bed moves by is a proper time integral; *when* it moves
# is still the slow clock, so the operator splitting is intact.


@wp.kernel
def accumulate_qs_x(
    qx: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    n: wp.array2d(dtype=wp.float32),
    d50: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    qs_int: wp.array2d(dtype=wp.float32),
    qs_comp: wp.array2d(dtype=wp.float32),
):
    """Integrate ``q_s*dt`` on interior x-faces. Launched over ``(ny, nx-1)``."""
    i, jj = wp.tid()
    j = jj + 1
    eta_max = wp.max(eta[i, j - 1], eta[i, j])
    z_bank = wp.max(z[i, j - 1], z[i, j])
    h_face = floodplain_flow_depth(eta_max, z_bank)  # == face_h_flow, w = 0 case
    qs = face_capacity(
        qx[i, j],
        h_face,
        h_face,
        0.5 * (n[i, j - 1] + n[i, j]),  # uniform n -> bit-exact n (the M1 idiom)
        0.5 * (d50[i, j - 1] + d50[i, j]),  # an engineering choice, not a law: see below
    )
    out = kahan_add(qs_int[i, j], qs_comp[i, j], qs * dt)
    qs_int[i, j] = out[0]
    qs_comp[i, j] = out[1]


@wp.kernel
def accumulate_qs_y(
    qy: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    n: wp.array2d(dtype=wp.float32),
    d50: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    qs_int: wp.array2d(dtype=wp.float32),
    qs_comp: wp.array2d(dtype=wp.float32),
):
    """Integrate ``q_s*dt`` on interior y-faces. Launched over ``(ny-1, nx)``."""
    ii, j = wp.tid()
    i = ii + 1
    eta_max = wp.max(eta[i - 1, j], eta[i, j])
    z_bank = wp.max(z[i - 1, j], z[i, j])
    h_face = floodplain_flow_depth(eta_max, z_bank)
    qs = face_capacity(
        qy[i, j],
        h_face,
        h_face,
        0.5 * (n[i - 1, j] + n[i, j]),
        0.5 * (d50[i - 1, j] + d50[i, j]),
    )
    out = kahan_add(qs_int[i, j], qs_comp[i, j], qs * dt)
    qs_int[i, j] = out[0]
    qs_comp[i, j] = out[1]


@wp.func
def _combined_capacity(
    q_ch: wp.float32,
    q_fp: wp.float32,
    eta_max: wp.float32,
    z_bank: wp.float32,
    z_ch: wp.float32,
    w_face: wp.float32,
    n_fp: wp.float32,
    n_ch: wp.float32,
    d50: wp.float32,
    dx: wp.float32,
) -> wp.float32:
    """Two-component bedload flux per unit **cell** width, mirroring the momentum mix.

    The channel component is per unit *channel* width and sees the channel's own
    flow depth and hydraulic radius; the floodplain component is the M1 face. They
    recombine by area fraction, so a cell whose flow is entirely in the channel
    carries ``q_s*dx == q_s_ch*w_face`` -- the ``dx/w`` bookkeeping that §4 names as
    the milestone's likeliest error, in one line.
    """
    h_fp = floodplain_flow_depth(eta_max, z_bank)
    qs_fp = face_capacity(q_fp, h_fp, h_fp, n_fp, d50)
    qs_ch = wp.float32(0.0)
    if w_face > 0.0:
        h_ch = channel_flow_depth(eta_max, z_bank, z_ch)
        if h_ch >= H_DRY:
            qs_ch = face_capacity(q_ch, h_ch, channel_radius(w_face, h_ch), n_ch, d50)
    frac = w_face / dx
    return qs_fp * (1.0 - frac) + qs_ch * frac


@wp.kernel
def accumulate_qs_x_channels(
    qx_ch: wp.array2d(dtype=wp.float32),
    qx_fp: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    n: wp.array2d(dtype=wp.float32),
    d50: wp.array2d(dtype=wp.float32),
    chan_w: wp.array2d(dtype=wp.float32),
    chan_d: wp.array2d(dtype=wp.float32),
    chan_n: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    dt: wp.float32,
    qs_int: wp.array2d(dtype=wp.float32),
    qs_comp: wp.array2d(dtype=wp.float32),
):
    """Two-component x-face transport integral. Launched over ``(ny, nx-1)``."""
    i, jj = wp.tid()
    j = jj + 1
    qs = _combined_capacity(
        qx_ch[i, j],
        qx_fp[i, j],
        wp.max(eta[i, j - 1], eta[i, j]),
        wp.max(z[i, j - 1], z[i, j]),
        face_channel_bed(z[i, j - 1], chan_d[i, j - 1], z[i, j], chan_d[i, j]),
        face_channel_width(chan_w[i, j - 1], chan_w[i, j]),
        0.5 * (n[i, j - 1] + n[i, j]),
        0.5 * (chan_n[i, j - 1] + chan_n[i, j]),
        0.5 * (d50[i, j - 1] + d50[i, j]),
        dx,
    )
    out = kahan_add(qs_int[i, j], qs_comp[i, j], qs * dt)
    qs_int[i, j] = out[0]
    qs_comp[i, j] = out[1]


@wp.kernel
def accumulate_qs_y_channels(
    qy_ch: wp.array2d(dtype=wp.float32),
    qy_fp: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    n: wp.array2d(dtype=wp.float32),
    d50: wp.array2d(dtype=wp.float32),
    chan_w: wp.array2d(dtype=wp.float32),
    chan_d: wp.array2d(dtype=wp.float32),
    chan_n: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    dt: wp.float32,
    qs_int: wp.array2d(dtype=wp.float32),
    qs_comp: wp.array2d(dtype=wp.float32),
):
    """Two-component y-face transport integral. Launched over ``(ny-1, nx)``."""
    ii, j = wp.tid()
    i = ii + 1
    qs = _combined_capacity(
        qy_ch[i, j],
        qy_fp[i, j],
        wp.max(eta[i - 1, j], eta[i, j]),
        wp.max(z[i - 1, j], z[i, j]),
        face_channel_bed(z[i - 1, j], chan_d[i - 1, j], z[i, j], chan_d[i, j]),
        face_channel_width(chan_w[i - 1, j], chan_w[i, j]),
        0.5 * (n[i - 1, j] + n[i, j]),
        0.5 * (chan_n[i - 1, j] + chan_n[i, j]),
        0.5 * (d50[i - 1, j] + d50[i, j]),
        dx,
    )
    out = kahan_add(qs_int[i, j], qs_comp[i, j], qs * dt)
    qs_int[i, j] = out[0]
    qs_comp[i, j] = out[1]


# --- Exner ---------------------------------------------------------------------


@wp.kernel
def exner_update(
    qs_int_x: wp.array2d(dtype=wp.float32),
    qs_int_y: wp.array2d(dtype=wp.float32),
    dz_lo: wp.array2d(dtype=wp.float32),
    dz_hi: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    inv_one_minus_p: wp.float32,
    dz_cum: wp.array2d(dtype=wp.float64),
    dz_unapplied: wp.array2d(dtype=wp.float64),
):
    """Apply one interval's transport to the cumulative bed change.

    ``dz = -1/(1-p) * div(qs_int) / dx`` over the four bounding faces, in the same
    sign convention continuity uses (net export lowers the cell). Because the flux
    lives on faces, the divergence telescopes across the domain and the update is
    conservative by construction -- the shared face is the same number for both
    neighbours, exactly as it is for water.

    The face values are widened to float64 **before** subtracting: the difference of
    two nearly-equal integrals is the one new cancellation site the milestone
    introduces, and widening it is free.

    ``dz_lo``/``dz_hi`` bound the *cumulative* change, so a structure cell
    (``lo = hi = 0``) and an alluvium floor (``lo = -thickness``) are one mechanism;
    deposition can always lift a floored cell back off its limit. What the bounds
    refuse is accumulated in metres into ``dz_unapplied`` rather than discarded --
    the sediment ledger turns it into a volume at ``A*(1-p)``.
    """
    i, j = wp.tid()
    div_x = wp.float64(qs_int_x[i, j + 1]) - wp.float64(qs_int_x[i, j])
    div_y = wp.float64(qs_int_y[i + 1, j]) - wp.float64(qs_int_y[i, j])
    want = dz_cum[i, j] - wp.float64(inv_one_minus_p) * (div_x + div_y) / wp.float64(dx)

    new = want
    lo = wp.float64(dz_lo[i, j])
    hi = wp.float64(dz_hi[i, j])
    if new < lo:
        new = lo
    if new > hi:
        new = hi
    if new != want:
        dz_unapplied[i, j] = dz_unapplied[i, j] + (want - new)
    dz_cum[i, j] = new


@wp.kernel
def rebuild_bed(
    z0: wp.array2d(dtype=wp.float32),
    dz_cum: wp.array2d(dtype=wp.float64),
    z: wp.array2d(dtype=wp.float32),
):
    """``z = float32(z0 + dz_cum)`` -- recomputed from the pristine bed, never ``+=``.

    ``z0`` is the bed the run started stepping with, *after* barriers: rebuilding
    from a pre-barrier bed would delete every dam at the first activation.
    """
    i, j = wp.tid()
    z[i, j] = wp.float32(wp.float64(z0[i, j]) + dz_cum[i, j])


def accumulate_transport(state, dt: float) -> None:
    """Integrate ``q_s*dt`` onto every interior face -- the whole fast-step cost of M7.

    **Where this is called from is not a free choice** (see the module docstring):
    :func:`solver.core.local_inertial.step` calls it after ``limit_qx``/``limit_qy``
    and before ``update_h``, because that is the only point where the face discharge
    is the water that actually moved *and* ``eta`` is still the surface those fluxes
    were computed from. It is driven off ``state.sediment`` exactly as the scheme
    drives its other optional physics off ``state.channels`` / ``state.rain`` /
    ``state.infil``, so an unarmed run launches nothing and stays bitwise identical.

    Dispatch mirrors the momentum update one-for-one: with channels armed each face
    carries a channel component (its own flow depth and hydraulic radius ``A/P``) plus
    the M1 floodplain component, recombined by area fraction. The interior-face
    ``nx > 1`` / ``ny > 1`` guards the kernels need are owed by this function, and a
    single-row fixture (``ny == 1``) therefore leaves the y integral allocated and
    zero rather than being special-cased anywhere else.
    """
    sed = state.sediment
    if sed is None:
        raise SedimentError("accumulate_transport requires arm_sediment(); nothing is armed")
    g = state.grid
    chan = state.channels
    dtf = float(dt)
    if chan is None:
        if g.nx > 1:
            wp.launch(
                accumulate_qs_x,
                dim=(g.ny, g.nx - 1),
                inputs=[
                    state.qx, state.eta, state.z, state.n, sed.d50, dtf,
                    sed.qs_int_x, sed.qs_comp_x,
                ],
                device=state.device,
            )  # fmt: skip
        if g.ny > 1:
            wp.launch(
                accumulate_qs_y,
                dim=(g.ny - 1, g.nx),
                inputs=[
                    state.qy, state.eta, state.z, state.n, sed.d50, dtf,
                    sed.qs_int_y, sed.qs_comp_y,
                ],
                device=state.device,
            )  # fmt: skip
    else:
        dxf = float(g.dx)
        if g.nx > 1:
            wp.launch(
                accumulate_qs_x_channels,
                dim=(g.ny, g.nx - 1),
                inputs=[
                    chan.qx_ch, chan.qx_fp, state.eta, state.z, state.n, sed.d50,
                    chan.w, chan.d, chan.n, dxf, dtf, sed.qs_int_x, sed.qs_comp_x,
                ],
                device=state.device,
            )  # fmt: skip
        if g.ny > 1:
            wp.launch(
                accumulate_qs_y_channels,
                dim=(g.ny - 1, g.nx),
                inputs=[
                    chan.qy_ch, chan.qy_fp, state.eta, state.z, state.n, sed.d50,
                    chan.w, chan.d, chan.n, dxf, dtf, sed.qs_int_y, sed.qs_comp_y,
                ],
                device=state.device,
            )  # fmt: skip


def clear_transport_integral(qs_int_x: wp.array, qs_int_y: wp.array) -> None:
    """Zero the transport integrals after an activation has consumed them.

    **The compensation terms are deliberately not zeroed.** They hold increment bits
    that were never added to the integral, so they were never applied to the bed
    either -- carrying them into the next interval pays them back, exactly as the
    rain kernels keep launching after the storm stops. Zeroing them instead would
    discard up to half an ulp per face per interval, systematically, which is the
    drift the compensation exists to remove.
    """
    qs_int_x.zero_()
    qs_int_y.zero_()


# --- What a morphology run carries, and where it lives -------------------------
# The kernels above fix every shape and dtype here, so arming is read off them
# rather than off prose: `accumulate_qs_x` writes `qs_int[i, jj+1]` over
# `(ny, nx-1)`, so the face accumulators are the grid's own `qx_shape`/`qy_shape`;
# `exner_update` and `rebuild_bed` declare their dtypes outright.
#
# **The split between what lives here and what lives on the process is by clock.**
# `d50` is read by the transport kernel every *fast* step, so it is state, exactly
# as M6's channel geometry is. `interval_s`, the alluvium thickness and the
# `dz_lo`/`dz_hi` bounds derived from it are read only at an *activation*, so they
# belong to :mod:`solver.processes.morphology`. That is not tidiness: the bounds
# must stay something a scenario can hand in whole, since `alluvium_thickness = 0`
# pins the floor and leaves the ceiling open and nothing in the `[sediment]` table
# can hold an outlet cell *down* -- which is exactly what the celerity fixture's
# pinned ends need (M7 plan §2 step 5, §4).


class SedimentError(ValueError):
    """Sediment inputs are inconsistent with the grid or with themselves."""


@dataclass
class SedimentState:
    """Device-side morphology accumulators, plus the grain size the fast loop reads.

    Two accumulators with opposite arithmetic decisions, for the reason the module
    docstring gives: ``qs_int_*`` is **float32 + Kahan** because its increments are
    the same order as the interval sum, and ``dz_cum`` is **float64** because a
    sub-millimetre increment onto an O(100 m) elevation is below ``eps(z)`` and a
    float32 add would delete it outright rather than merely drift.

    ``dz_unapplied`` banks in **metres** what the bounds refused; the sediment
    ledger (M7 build step 6) converts it once at ``A*(1-p)``, the same conversion
    :meth:`solid_volume` makes for the applied part.
    """

    d50: wp.array  # (ny, nx) float32 median grain size, m
    porosity: float  # bed porosity p; dz -> solid volume is A*(1-p)*dz
    z0: wp.array  # (ny, nx) float32 pristine bed -- see `arm_sediment` on *when*
    dz_cum: wp.array  # (ny, nx) float64 cumulative bed change, m
    dz_unapplied: wp.array  # (ny, nx) float64 bed change the bounds refused, m
    qs_int_x: wp.array  # (ny, nx+1) float32 transport integral, m^2
    qs_comp_x: wp.array  # (ny, nx+1) float32 its Kahan compensation
    qs_int_y: wp.array  # (ny+1, nx)
    qs_comp_y: wp.array  # (ny+1, nx)
    d50_min: float  # host-side diagnostics, validated at arm time
    d50_max: float
    zero_d50_cells: int  # cells with no grain size -- NOT immobile cells, see below

    @property
    def inv_one_minus_p(self) -> float:
        """Exner's ``1/(1-p)``, the factor :func:`exner_update` takes."""
        return 1.0 / (1.0 - self.porosity)

    def bed_change_numpy(self) -> np.ndarray:
        """Cumulative bed change as a host ``(ny, nx)`` float64 array (metres)."""
        return self.dz_cum.numpy()

    def solid_volume(self, cell_area: float) -> float:
        """Net solid volume the bed has gained (m^3), float64-summed.

        Negative when the domain has net-eroded. The sediment ledger's headline
        term; mirrors :meth:`solver.core.state.State.loss_volume` for water.
        """
        return float(self.dz_cum.numpy().sum()) * cell_area * (1.0 - self.porosity)

    def banked_volume(self, cell_area: float) -> float:
        """Solid volume (m^3) the bounds refused to apply -- banked, never discarded."""
        return float(self.dz_unapplied.numpy().sum()) * cell_area * (1.0 - self.porosity)

    def clear_integral(self) -> None:
        """Zero the transport integrals, keeping the compensation debt."""
        clear_transport_integral(self.qs_int_x, self.qs_int_y)

    def summary(self) -> str:
        grain = (
            f"d50 = {1000.0 * self.d50_min:.2f}..{1000.0 * self.d50_max:.2f} mm"
            if self.d50_min != self.d50_max
            else f"d50 = {1000.0 * self.d50_max:.2f} mm"
        )
        zeros = f", {self.zero_d50_cells} cells with no grain size" if self.zero_d50_cells else ""
        return f"{grain}, porosity {self.porosity:.2f}{zeros}"

    def as_attrs(self) -> dict:
        return {
            "d50_min_m": self.d50_min,
            "d50_max_m": self.d50_max,
            "porosity": self.porosity,
            "zero_d50_cells": self.zero_d50_cells,
        }


def validate_grain_size(d50: np.ndarray | float, grid) -> np.ndarray:
    """Check and broadcast the grain-size field against the grid.

    A scalar broadcasts to a uniform field, which keeps the face mean
    ``0.5*(d50 + d50)`` bit-exact -- the ``n`` idiom, so a uniform grain size costs
    nothing in accuracy for the generality.

    Rejects what cannot mean anything (negative, non-finite). A **zero is accepted
    and counted, and it does not make that cell immobile** -- ``d50`` is
    *face-averaged*, so a lone zero cell's faces carry half its neighbours' grain
    size, transport normally, and move its bed like any other. Worse, the error has
    a sign: ``theta`` goes as ``1/d50``, so an isolated zero reads as **more**
    mobile than its neighbours, not less. Only the interior faces of a *contiguous*
    zero region see ``d50 == 0`` on both sides and carry nothing, and even there the
    region's fringe stays mobile. ``[sediment] d50`` refuses a zero scalar for a
    related reason (there is no ``d50`` that means "off"); a field can carry zeros,
    ``zero_d50_cells`` counts them and :meth:`SedimentState.summary` says how many,
    named for what they are rather than for what they look like they do.

    The way to spell "the bed here cannot move" is ``alluvium_thickness = 0``, whose
    bound holds the cell at zero *and banks what it refused* into the sediment
    ledger -- rather than quietly transporting at a grain size nobody chose.
    ``test_a_zero_grain_size_does_not_make_a_cell_immobile`` pins the real behaviour
    so the label cannot drift back into a claim.
    """
    ny, nx = grid.shape
    if np.isscalar(d50):
        arr = np.full((ny, nx), float(d50), dtype=np.float32)
    else:
        arr = np.ascontiguousarray(d50, dtype=np.float32).copy()
        if arr.shape != (ny, nx):
            raise SedimentError(f"d50 shape {arr.shape} != grid {(ny, nx)}")
    if not np.isfinite(arr).all():
        raise SedimentError("d50 field contains non-finite values")
    if (arr < 0).any():
        raise SedimentError("d50 must be >= 0 everywhere (0 = an inert cell)")
    return arr


def arm_sediment(state, d50: np.ndarray | float, porosity: float) -> SedimentState:
    """Attach the morphology accumulators to ``state`` and return them.

    **Call this after the bed is final, not before.** ``z0`` is captured from
    ``state.z`` here and the bed is *rebuilt* from it at every activation
    (:func:`rebuild_bed`), so arming before
    :func:`solver.processes.reservoir.apply_barriers` would delete every dam at the
    first activation. The run loop owes this ordering; it is not something the
    kernels can check.

    Arming is what makes a run a morphology run: unarmed, nothing launches a
    transport kernel and the bed is static, so every pre-M7 scenario is bitwise
    unchanged. All accumulators start at zero, so the initial bed *is* ``z0`` and
    the stored ``bed`` and ``bed_change`` agree by construction rather than by
    accumulation (M7 plan §1.1).

    Arming is **not** idempotent and refuses to run twice, unlike
    :func:`solver.core.hllc.arm_hllc`. Re-arming would recapture ``z0`` from a bed
    that has already moved and zero the cumulative change that moved it -- a silent
    ledger reset, and the bed would then look pristine while carrying its whole
    history. Better a loud error than a run whose ``bed_change`` is a lie.
    """
    if getattr(state, "sediment", None) is not None:
        raise SedimentError(
            "state is already armed for morphology; re-arming would recapture z0 from "
            "a moved bed and discard the cumulative change that moved it"
        )
    # The same open interval `[sediment] porosity` enforces, so a state armed
    # directly (a fixture, a test) cannot be legal where a scenario would be refused.
    # Zero is arithmetically harmless -- 1/(1-0) = 1 -- and excluded anyway, because
    # a bed with no pore space is not a bed Exner's 1/(1-p) was written for.
    if not 0.0 < porosity < 1.0:
        raise SedimentError(f"porosity must be in (0, 1), got {porosity}")
    g = state.grid
    grain = validate_grain_size(d50, g)
    dev = state.device
    sed = SedimentState(
        d50=wp.array(grain, dtype=wp.float32, device=dev),
        porosity=float(porosity),
        z0=wp.clone(state.z),
        dz_cum=wp.zeros(g.shape, dtype=wp.float64, device=dev),
        dz_unapplied=wp.zeros(g.shape, dtype=wp.float64, device=dev),
        qs_int_x=wp.zeros(g.qx_shape, dtype=wp.float32, device=dev),
        qs_comp_x=wp.zeros(g.qx_shape, dtype=wp.float32, device=dev),
        qs_int_y=wp.zeros(g.qy_shape, dtype=wp.float32, device=dev),
        qs_comp_y=wp.zeros(g.qy_shape, dtype=wp.float32, device=dev),
        d50_min=float(grain.min()),
        d50_max=float(grain.max()),
        zero_d50_cells=int((grain == 0).sum()),
    )
    state.sediment = sed
    return sed


# --- Host-side reference + the celerity the gates are written against ----------


def _as_arrays(*vals: np.ndarray | float) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(v, dtype=np.float64) for v in vals)


def shields_from_flow(
    q: np.ndarray | float,
    h: np.ndarray | float,
    n: np.ndarray | float,
    d50: np.ndarray | float,
    radius: np.ndarray | float | None = None,
) -> np.ndarray:
    """Host reference for :func:`shields_number` (``radius`` defaults to ``h``)."""
    q, h, n, d50 = _as_arrays(q, h, n, d50)
    r = h if radius is None else np.asarray(radius, dtype=np.float64)
    tau_over_rho = GRAVITY * n**2 * q**2 / (h**2 * np.cbrt(r))
    return tau_over_rho / (SUBMERGED_SG * GRAVITY * d50)


def capacity_from_flow(
    q: np.ndarray | float,
    h: np.ndarray | float,
    n: np.ndarray | float,
    d50: np.ndarray | float,
    radius: np.ndarray | float | None = None,
) -> np.ndarray:
    """Host reference for the MPM capacity (magnitude, m^2/s) the kernels compute."""
    theta = shields_from_flow(q, h, n, d50, radius)
    d50 = np.asarray(d50, dtype=np.float64)
    excess = np.maximum(theta - SHIELDS_CRITICAL, 0.0)
    return MPM_COEFFICIENT * excess**MPM_EXPONENT * np.sqrt(SUBMERGED_SG * GRAVITY * d50**3)


def bed_celerity(
    q: np.ndarray | float,
    h: np.ndarray | float,
    n: np.ndarray | float,
    d50: np.ndarray | float,
    porosity: float,
    width: np.ndarray | float | None = None,
    dx: float | None = None,
) -> np.ndarray:
    """Speed a low bed wave migrates at (m/s), analytically, for the law shipped here.

    Built once and used twice (M7 plan §2): it is what the bed-wave celerity gate
    compares the numerical migration against, and what the pre-run morphological-CFL
    print is computed from. Neither is free without it.

    Linearising Exner about a uniform flow at fixed unit discharge ``q``, a rise in
    the bed thins the flow, raises the shear and so raises the capacity, giving
    ``dz/dt + c_b * dz/dx = 0`` with ``c_b = (1/(1-p)) * dq_s/dz`` and
    ``dq_s/dz = -dq_s/dh``. For MPM::

        dtheta/dh = -kappa * theta / h,   kappa = 2 + R'/(3R) * h  ->  7/3 wide
        c_b = 12 * kappa * (theta/h) * (theta - theta_c)^0.5 * sqrt(s' g d50^3) / (1-p)

    ``kappa`` is ``7/3`` for a wide section (``R = h``) and tends to ``2`` for a
    channel far narrower than it is deep; passing ``width`` uses the channel radius
    ``A/P`` throughout. Passing ``dx`` as well scales by the conveyance fraction
    ``w/dx``: a sub-grid channel's flux is per *channel* width while Exner spreads
    the bed change over the whole cell, so the cell's bed wave is that much slower.

    Quasi-steady and small-amplitude, like every analytical bed-wave celerity -- it
    is a gate and a warning threshold, not a prediction.
    """
    q, h, n, d50 = _as_arrays(q, h, n, d50)
    if width is None:
        r = h
        kappa = 7.0 / 3.0
        frac = 1.0
    else:
        w = np.asarray(width, dtype=np.float64)
        r = w * h / (w + 2.0 * h)
        # dln(R)/dln(h) = w/(w + 2h), and dln(theta)/dln(h) = -2 - (1/3)dln(R)/dln(h)
        kappa = 2.0 + w / (3.0 * (w + 2.0 * h))
        frac = 1.0 if dx is None else w / float(dx)
    theta = shields_from_flow(q, h, n, d50, r)
    excess = np.maximum(theta - SHIELDS_CRITICAL, 0.0)
    # dq_s/dtheta = 8 * 1.5 * sqrt(theta - theta_c) * sqrt(s' g d50^3); dtheta/dz = +kappa theta/h
    dqs_dz = (
        MPM_COEFFICIENT
        * MPM_EXPONENT
        * kappa
        * (theta / h)
        * np.sqrt(excess)
        * np.sqrt(SUBMERGED_SG * GRAVITY * d50**3)
    )
    return frac * dqs_dz / (1.0 - float(porosity))


def morphological_courant(celerity: float, interval_s: float, dx: float) -> float:
    """``c_b * interval_s / dx`` -- cells a bed wave crosses in one slow interval.

    A bed wave that travels more than a cell per activation is a splitting artefact,
    not a result: the exact analogue of M5's *"54,000 m^3 into one 40 m cell is a
    34 m column"*. The run loop prints this before stepping, from the scenario's own
    numbers, in the codebase's habit of saying the dangerous ratio out loud before
    it bites.
    """
    return float(celerity) * float(interval_s) / float(dx)


def celerity_field(state) -> np.ndarray:
    """Per-cell bed-wave celerity (m/s) from the flow the state is carrying *now*.

    :func:`bed_celerity` evaluated cell by cell, so a morphology run can report the
    morphological Courant number it **achieved** over an interval
    (:mod:`solver.processes.morphology`) rather than one assumed from a nominal flow.
    There is no useful pre-run number to compute instead: at ``t = 0`` a scenario
    usually has no flow at all, and a Courant number derived from a flow that has not
    happened would be a reassurance rather than a warning.

    Host-side numpy over a few field copies, called once per *activation* (900 s of
    simulated time by default), so it is noise beside the fast loop -- and it is
    deliberately not a kernel, because nothing in the physics reads it.

    **A channel cell is evaluated as a channel**, with the channel component's own
    discharge, its column depth, its roughness and the ``w/dx`` conveyance fraction
    that spreads a per-channel-width flux over the whole cell. Reading those cells
    with the floodplain form would under-report the celerity by orders of magnitude
    exactly where the transport is, which for a warning diagnostic is the dangerous
    direction to be wrong in.

    Dry cells, cells with no grain size and cells with no flow return exactly zero.
    """
    sed = state.sediment
    if sed is None:
        raise SedimentError("celerity_field requires arm_sediment(); nothing is armed")
    g = state.grid
    h = state.h.numpy().astype(np.float64)
    n = state.n.numpy().astype(np.float64)
    d50 = sed.d50.numpy().astype(np.float64)
    p = sed.porosity

    def centred(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
        """Cell-centred flow magnitude from the two pairs of bounding faces."""
        return np.hypot(0.5 * (qa[:, :-1] + qa[:, 1:]), 0.5 * (qb[:-1, :] + qb[1:, :]))

    # Guards first, so nothing below divides by a dry depth or a zero grain size; the
    # mask decides what survives and the substituted values never reach the output.
    grain_ok = d50 > 0.0
    d50_safe = np.where(grain_ok, d50, 1.0)
    q = centred(state.qx.numpy().astype(np.float64), state.qy.numpy().astype(np.float64))
    wet = (h >= H_DRY) & grain_ok & (q > 0.0)
    c_fp = np.where(
        wet, bed_celerity(np.where(wet, q, 0.0), np.where(wet, h, 1.0), n, d50_safe, p), 0.0
    )

    chan = state.channels
    if chan is None:
        return c_fp

    w = chan.w.numpy().astype(np.float64)
    d = chan.d.numpy().astype(np.float64)
    has = (w > 0.0) & (d > 0.0)
    w_safe = np.where(has, w, 1.0)
    # The M6 storage curve, host-side (solver.core.channels.column_depth): below bank
    # full the cell's water is all in the channel and stands dx/w deeper than the mean.
    h_bf = w_safe * d / g.dx
    h_col = np.where(h <= h_bf, h * g.dx / w_safe, d + (h - h_bf))
    q_ch = centred(chan.qx_ch.numpy().astype(np.float64), chan.qy_ch.numpy().astype(np.float64))
    wet_ch = has & (h_col >= H_DRY) & grain_ok & (q_ch > 0.0)
    c_ch = bed_celerity(
        np.where(wet_ch, q_ch, 0.0),
        np.where(wet_ch, h_col, 1.0),
        chan.n.numpy().astype(np.float64),
        d50_safe,
        p,
        width=w_safe,
        dx=g.dx,
    )
    # The **larger** of the two components, not the channel one selected: a channel
    # cell flowing overbank (``h > h_bf``) carries a real floodplain flux across its
    # faces too, and above bank full that component is most of the cell's width. A
    # hard select would report exactly zero for a channel cell whose channel happens
    # to be still, which for a warning diagnostic is the wrong way to be wrong.
    return np.maximum(np.where(has & wet_ch, c_ch, 0.0), c_fp)
