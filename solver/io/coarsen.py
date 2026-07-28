"""Resolution choice: conservative coarsening of a domain (M6, plan §1.1).

Reach is bought by **choosing the resolution** and putting the lost river back as
a sub-grid channel (:mod:`solver.core.channels`), rather than by nesting a fine
grid inside a coarse one. That choice is what makes M6's conservation story
simple: one uniform grid per run, aggregated **once, before any water moves**, so
continuity stays a pure flux divergence and there is no resolution interface for
mass to leak across (HANDOFF §12 names that interface as the highest risk in the
project; M6 does not build it, and says so).

``[grid] coarsen = k`` runs a ``k``x coarser grid: ``dx' = k·dx`` and every input
field is aggregated over ``k x k`` blocks by the rule that suits it:

===================  ==========  ===============================================
field                rule        why
===================  ==========  ===============================================
bed elevation        **mean**    volume-preserving: the storage of a block of
                                 cells is its mean elevation times its area
Manning n            mean        an area-weighted roughness for the coarse cell
infiltration, rain   mean        rates per unit area -- the mean preserves volume
channel width/depth  **max**     a river passes *through* a block; averaging its
                                 width with the dry cells beside it would thin it
                                 away, which is the failure mode sub-grid channels
                                 exist to prevent
===================  ==========  ===============================================

Cell **indices** in the scenario (``[[inflow]] cell``, ``[[structures]] cells``,
``pool``, ``outlet``) are authored in the *assembled-domain* frame -- after
``[grid] window``, before ``coarsen`` -- and mapped here by ``i // k``, so changing
the run resolution never silently moves an inflow or a dam.

**Documented limitation.** A block mean removes barriers: a one-cell ridge or levee
crest inside a 4x4 block is averaged away, and the coarse run will not hold water
the fine run holds. Engineered barriers survive because ``[[structures]]`` are
applied *after* coarsening at their authored crest elevation; natural ridges do
not. That is a property of coarsening, not a bug -- state it wherever a coarse
run's numbers are reported.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from solver.io.config import Scenario


class CoarsenError(ValueError):
    """A coarsening factor cannot be applied to this domain or scenario."""


def crop_to_multiple(field: np.ndarray, k: int) -> np.ndarray:
    """Trim the trailing rows/columns that do not fill a whole ``k x k`` block.

    Cropping (rather than padding) keeps every coarse cell backed by real data:
    an invented half-block would be terrain nobody surveyed.
    """
    ny, nx = field.shape
    return field[: ny - ny % k, : nx - nx % k]


def block_reduce(field: np.ndarray, k: int, how: str = "mean") -> np.ndarray:
    """Aggregate a 2-D field over ``k x k`` blocks (``mean`` or ``max``)."""
    if k < 1:
        raise CoarsenError(f"coarsen factor must be >= 1, got {k}")
    if k == 1:
        return field
    cropped = crop_to_multiple(field, k)
    ny, nx = cropped.shape
    if ny == 0 or nx == 0:
        raise CoarsenError(
            f"coarsen={k} leaves no whole blocks in a {field.shape[0]}x{field.shape[1]} domain"
        )
    blocks = cropped.reshape(ny // k, k, nx // k, k)
    if how == "mean":
        # float64 accumulation: a block mean is the *definition* of the coarse cell's
        # storage, so it should not inherit float32 summation error.
        out = blocks.astype(np.float64).mean(axis=(1, 3))
    elif how == "max":
        out = blocks.max(axis=(1, 3))
    else:  # pragma: no cover - programmer error
        raise CoarsenError(f"unknown block rule {how!r}")
    return np.ascontiguousarray(out, dtype=field.dtype)


def coarsened_shape(shape: tuple[int, int], k: int) -> tuple[int, int]:
    """Shape after cropping to whole blocks and reducing by ``k``."""
    return (shape[0] // k, shape[1] // k)


def crop_report(shape: tuple[int, int], k: int) -> str | None:
    """Human-readable note about rows/cols dropped by cropping, or ``None``."""
    dr, dc = shape[0] % k, shape[1] % k
    if not (dr or dc):
        return None
    return f"coarsen={k} trimmed {dr} row(s) and {dc} column(s) to whole blocks"


def _map_cell(cell: tuple[int, int], k: int) -> tuple[int, int]:
    return (cell[0] // k, cell[1] // k)


def coarsen_scenario(scenario: Scenario, k: int) -> Scenario:
    """Map a scenario's cell indices into the coarsened grid (``i // k``).

    Returns the scenario unchanged when ``k == 1``, so the default path is not
    merely equivalent but *identical*.
    """
    if k == 1:
        return scenario
    inflows = [replace(inf, cell=_map_cell(inf.cell, k)) for inf in scenario.inflows]
    structures = []
    for s in scenario.structures:
        box = (
            None
            if s.pool is None
            else (s.pool[0] // k, s.pool[1] // k, s.pool[2] // k, s.pool[3] // k)
        )
        out = (
            None
            if s.outlet is None
            else (s.outlet[0] // k, s.outlet[1] // k, s.outlet[2] // k, s.outlet[3] // k)
        )
        structures.append(
            replace(
                s,
                cells=sorted({_map_cell(c, k) for c in s.cells}),
                pool=box,
                outlet=out,
            )
        )
    return replace(scenario, inflows=inflows, structures=structures)


def check_indices(scenario: Scenario, shape: tuple[int, int]) -> None:
    """Fail loudly on any scenario cell index outside the resolved domain.

    Windowing and coarsening both change what a ``(row, col)`` means; a scenario
    that points outside the domain it is being run on is a mistake worth stopping
    for, not a source clamped silently to the edge.
    """
    ny, nx = shape

    def _bad(i: int, j: int) -> bool:
        return not (0 <= i < ny and 0 <= j < nx)

    for n, inf in enumerate(scenario.inflows):
        if _bad(*inf.cell):
            raise CoarsenError(
                f"[[inflow]] #{n} cell {list(inf.cell)} is outside the resolved "
                f"{ny}x{nx} domain (check [grid] window / coarsen)"
            )
    for s in scenario.structures:
        for cell in s.cells:
            if _bad(*cell):
                raise CoarsenError(
                    f"structure '{s.name}' cell {list(cell)} is outside the resolved "
                    f"{ny}x{nx} domain (check [grid] window / coarsen)"
                )
        for name, box in (("pool", s.pool), ("outlet", s.outlet)):
            if box is not None and (_bad(box[0], box[1]) or _bad(box[2], box[3])):
                raise CoarsenError(
                    f"structure '{s.name}' {name} {list(box)} is outside the resolved "
                    f"{ny}x{nx} domain (check [grid] window / coarsen)"
                )
