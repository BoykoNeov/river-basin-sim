"""Canonical Zarr result store (M1, HANDOFF §7.2).

The solver's *only* output contract for M1: a Zarr group with dims
``(time, y, x)``, written incrementally one timestep per chunk so the store never
has to hold the whole time-series in memory (frames are appended as the run
produces them). The viewer's lean per-frame stream (§7.3) is deliberately **not**
built here -- that lands in M2 when the loop closes.

Layout::

    results.zarr/
      .zattrs   crs, dx, units, scheme, run_hash, mass_* (from the ledger)
      time       (T,)      float64  simulated seconds
      depth      (T, Y, X) float32  water depth h
      u, v       (T, Y, X) float32  cell-centred velocity
      bed        (Y, X)    float32  bed elevation z -- always the *initial* bed
      bed_change (T, Y, X) float32  cumulative bed change, when morphology ran (M7)

``dimension_names`` is set on every array (the zarr v3 metadata xarray needs) so
the store opens directly with ``xarray.open_zarr(..., consolidated=False)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from solver.core.grid import Grid


class ZarrWriter:
    """Incremental writer for the canonical result store.

    Preallocates ``n_frames`` along the time axis (the run knows its output count
    up front: ``end_time / output_every + 1``) and fills frames by index.
    """

    def __init__(
        self,
        path: str | Path,
        grid: Grid,
        n_frames: int,
        attrs: dict,
        *,
        bed_change: bool = False,
    ):
        self.path = Path(path)
        self.grid = grid
        self.n_frames = n_frames
        self._i = 0

        self.root = zarr.open_group(self.path, mode="w")
        ny, nx = grid.shape
        tyx = ("time", "y", "x")
        chunk = (1, ny, nx)  # one timestep per chunk -> cheap frame streaming
        self._depth = self.root.create_array(
            "depth", shape=(n_frames, ny, nx), chunks=chunk, dtype="f4", dimension_names=tyx
        )
        self._u = self.root.create_array(
            "u", shape=(n_frames, ny, nx), chunks=chunk, dtype="f4", dimension_names=tyx
        )
        self._v = self.root.create_array(
            "v", shape=(n_frames, ny, nx), chunks=chunk, dtype="f4", dimension_names=tyx
        )
        self._time = self.root.create_array(
            "time", shape=(n_frames,), dtype="f8", dimension_names=("time",)
        )
        self._bed = self.root.create_array(
            "bed", shape=(ny, nx), dtype="f4", dimension_names=("y", "x")
        )
        # Morphology (M7 §1.7): `bed` stays the *initial* bed and the moving part
        # ships beside it, rather than promoting `bed` to (T, Y, X). Two reasons:
        # every existing reader keeps working untouched, and a bed *change* is a
        # small quantity, so float32 stores it at full relative precision where
        # float32 elevations would not (§1.1 -- the same conditioning that made
        # `dz_cum` float64 in the first place). Created only when a run actually
        # has morphology, so an unarmed run's store is byte-identical to M6's.
        self._bed_change = None
        if bed_change:
            self._bed_change = self.root.create_array(
                "bed_change",
                shape=(n_frames, ny, nx),
                chunks=chunk,
                dtype="f4",
                dimension_names=tyx,
            )
        self.root.attrs.update(attrs)

    def write_bed(self, bed: np.ndarray) -> None:
        """Write the static bed field once (call any time before finalize)."""
        self._bed[:] = np.ascontiguousarray(bed, dtype=np.float32)

    def write_static(self, name: str, field: np.ndarray) -> None:
        """Write an extra static ``(Y, X)`` field into the store (§7.2 extensible).

        Used by M6 for sub-grid channel geometry (``channel_width``,
        ``channel_depth``): the bed alone no longer describes the terrain the run
        actually stepped on, so a stored result that used channels carries them.
        """
        ny, nx = self.grid.shape
        arr = self.root.create_array(name, shape=(ny, nx), dtype="f4", dimension_names=("y", "x"))
        arr[:] = np.ascontiguousarray(field, dtype=np.float32)

    def append(
        self,
        time: float,
        depth: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        bed_change: np.ndarray | None = None,
    ) -> None:
        """Append one output frame at the next time index.

        ``bed_change`` is required exactly when the store was opened with
        ``bed_change=True``, and refused otherwise -- **both** directions raise. A
        preallocated frame is all zeros and zero is a perfectly legal bed change, so
        a caller that forgot to pass one would not leave a hole a reader could see:
        it would leave a store that says the bed did not move. That is the same class
        of quiet lie as keeping depth non-negative with ``max(h, 0)``.
        """
        if self._i >= self.n_frames:
            raise IndexError(f"more frames than preallocated ({self.n_frames})")
        if (bed_change is None) != (self._bed_change is None):
            raise ValueError(
                "bed_change must be passed exactly when the store carries it "
                f"(store: {'yes' if self._bed_change is not None else 'no'}, "
                f"frame: {'yes' if bed_change is not None else 'no'}); an omitted "
                "frame would store zeros, which reads as 'the bed did not move'"
            )
        i = self._i
        self._time[i] = float(time)
        self._depth[i] = np.ascontiguousarray(depth, dtype=np.float32)
        self._u[i] = np.ascontiguousarray(u, dtype=np.float32)
        self._v[i] = np.ascontiguousarray(v, dtype=np.float32)
        if self._bed_change is not None:
            # float64 -> float32 on the way out: what is *balanced* is `dz_cum` in
            # f64 (the sediment ledger reads that, never this array), and what is
            # stored is a rendering of it for pictures and analysis.
            self._bed_change[i] = np.ascontiguousarray(bed_change, dtype=np.float32)
        self._i += 1

    def finalize(self, extra_attrs: dict | None = None) -> None:
        """Merge final attributes (e.g. the mass-balance series) and close out.

        If fewer frames were written than preallocated (an early stop), the time
        axis is trimmed so trailing zero frames don't masquerade as real output.
        """
        if extra_attrs:
            self.root.attrs.update(extra_attrs)
        self.root.attrs["n_frames"] = self._i
        if self._i < self.n_frames:
            self.root.attrs["n_frames_allocated"] = self.n_frames
