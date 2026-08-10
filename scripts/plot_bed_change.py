"""Render a morphological run's bed change to a PNG (M7 build step 7).

M7's evidence is a **figure, not a scrub**. The canonical store carries
``bed_change (T, Y, X)`` beside the static ``bed`` (§7.2), but the viewer
deliberately does not animate terrain this milestone (M7 plan §1.7): re-fitting the
height map per frame is a viewer milestone, and the shader still lifts water as
``bed + depth`` rather than through the sub-grid storage curve, so a moving bed
would move that mis-lift with it. Fix the lift first, then animate. Until then this
is how a morphological run is looked at.

Three panels, which is the whole story a bed-change run has to tell:

  1. **the bed it started on** -- what the viewer renders, and what panel 2 is a
     correction to;
  2. **the cumulative bed change** at the last written frame, on a symmetric
     diverging scale so scour and fill are the same size of colour;
  3. **the sediment ledger's own series** -- gross displaced volume against time,
     with the balance's worst relative residual against its gate.

**The numbers come from ``.zattrs``, never from the stored field.** What the ledger
balances is the float64 ``dz_cum`` (build step 6); ``bed_change`` in the store is a
float32 *rendering* of it for pictures, and re-deriving a volume from the picture
would report the rendering's error as the run's. Panel 2 is the picture; panels 1
and 3 are the record.

Run::

    uv run python scripts/plot_bed_change.py data/results/demo.zarr
    uv run python scripts/plot_bed_change.py <store> --out bed_change.png

Writes ``<store>.bed_change.png`` by default -- beside the store, the same place
and for the same reason as ``<store>.provenance.json``. Needs matplotlib, which
lives in the ``geo`` extra: ``uv sync --extra geo``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def _load(zarr_path: Path) -> xr.Dataset:
    """Open the store and refuse one that has no morphology to draw."""
    ds = xr.open_zarr(zarr_path, consolidated=False)
    if "bed_change" not in ds:
        raise SystemExit(
            f"{zarr_path} has no bed_change: it was run without a [sediment] table, "
            "so nothing moved the bed. Add [sediment] to the scenario and re-run."
        )
    return ds


def _last_frame(ds: xr.Dataset) -> int:
    """Index of the last frame actually written.

    ``ZarrWriter.finalize`` records ``n_frames`` but never resizes the time axis, so
    an early-stopped run leaves preallocated zero frames at the end. For ``depth``
    that reads as a dry domain; for ``bed_change`` **zero is a perfectly legal
    value**, so a trailing frame is indistinguishable from "the bed did not move"
    -- read the count, never the axis length.
    """
    n = int(ds.attrs.get("n_frames", ds.sizes["time"]))
    return max(0, min(n, int(ds.sizes["time"])) - 1)


def _series(ds: xr.Dataset) -> tuple[list[float], list[float], list[float]]:
    """Times, gross displaced volume and banked volume from the sediment ledger."""
    recs = ds.attrs.get("sediment_balance_series", [])
    return (
        [float(r["time"]) for r in recs],
        [float(r["gross_volume"]) for r in recs],
        [float(r["banked_volume"]) for r in recs],
    )


def plot(zarr_path: Path, out_path: Path) -> Path:
    """Draw the three panels and write the PNG; returns where it went."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: this is a batch tool, there is no window
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - environment, not logic
        raise SystemExit(
            "matplotlib is needed for this figure and lives in the optional 'geo' "
            "extra: run `uv sync --extra geo`"
        ) from e

    ds = _load(zarr_path)
    i = _last_frame(ds)
    bed = ds["bed"].values
    dz = ds["bed_change"].isel(time=i).values
    t_end = float(ds["time"].values[i])
    dx = float(ds.attrs.get("dx", 1.0))
    span = (0.0, bed.shape[1] * dx / 1000.0, bed.shape[0] * dx / 1000.0, 0.0)  # km, row 0 top

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.suptitle(
        f"{ds.attrs.get('scenario', zarr_path.stem)} -- bed change at t = {t_end:.0f} s "
        f"({bed.shape[0]}x{bed.shape[1]} @ {dx:.1f} m)"
    )

    im = axes[0].imshow(bed, cmap="terrain", extent=span, origin="upper")
    axes[0].set_title("initial bed (what the viewer renders)")
    fig.colorbar(im, ax=axes[0], label="elevation (m)", shrink=0.85)

    # Symmetric limits so the zero of a diverging map is the zero of the physics:
    # an off-centre scale makes deposition look like scour at a glance.
    lim = float(np.abs(dz).max())
    im = axes[1].imshow(
        dz, cmap="RdBu", extent=span, origin="upper", vmin=-lim, vmax=lim if lim > 0 else 1e-12
    )
    axes[1].set_title(f"cumulative bed change ({dz.min():+.3f} .. {dz.max():+.3f} m)")
    fig.colorbar(im, ax=axes[1], label="scour (-) / fill (+) (m)", shrink=0.85)
    for ax in axes[:2]:
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

    times, gross, banked = _series(ds)
    ax = axes[2]
    if times:
        ax.plot(times, gross, marker="o", ms=3, color="#8c510a", label="gross displaced")
        if any(b != 0.0 for b in banked):
            # A bound refused this much and the ledger holds it; a flat zero line
            # would be noise, a nonzero one is the run telling on itself.
            ax.plot(times, np.abs(banked), marker="s", ms=3, color="#01665e", label="|banked|")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("solid volume (m3)")
    gate = float(ds.attrs.get("sediment_gate", 0.0))
    err = float(ds.attrs.get("sediment_max_rel_error", 0.0))
    ax.set_title(f"sediment ledger: max rel err {err:.1e} (gate {gate:.0e})")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a run's bed change (M7).")
    p.add_argument("zarr", nargs="?", default="data/results/demo.zarr", help="canonical store")
    p.add_argument("--out", default=None, help="PNG path (default: <store>.bed_change.png)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    zarr_path = Path(args.zarr)
    out = Path(args.out) if args.out else zarr_path.with_name(zarr_path.name + ".bed_change.png")
    ds = _load(zarr_path)
    i = _last_frame(ds)
    dz = ds["bed_change"].isel(time=i).values
    acts = ds.attrs.get("morphology", [])
    print(f"bed change over {len(acts)} activations, {dz.min():+.4f} .. {dz.max():+.4f} m")
    if acts:
        print(f"  peak morphological courant : {max(a['courant'] for a in acts):.2f}")
    print(
        f"  sediment max rel error     : {float(ds.attrs.get('sediment_max_rel_error', 0.0)):.2e}"
    )
    print(f"figure: {plot(zarr_path, out)}")


if __name__ == "__main__":
    sys.exit(main())
