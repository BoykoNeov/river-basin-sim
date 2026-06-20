"""Toolchain smoke test (HANDOFF §11.4).

Proves the pieces the rest of the project stands on actually work *on this machine*,
not just in theory:

  1. NVIDIA Warp initializes and sees the CUDA GPU.
  2. A tiny Warp kernel JIT-compiles for this GPU's architecture and runs correctly
     (the real test on Blackwell / sm_120 — classifiers can't confirm this).
  3. Results round-trip through the canonical store: numpy -> Zarr -> xarray.

Run with:  uv run python scripts/smoke_test.py
Exit code 0 = all green.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import warp as wp
import xarray as xr
import zarr


@wp.kernel
def fill_kernel(out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.float32(i) * 2.0


def main() -> int:
    print("=" * 60)
    print("River Basin Simulator - toolchain smoke test")
    print("=" * 60)

    # 1. Warp init + device discovery -----------------------------------
    wp.init()
    print(f"Warp version: {wp.config.version}")

    cuda_devices = wp.get_cuda_devices()
    if not cuda_devices:
        print("FAIL: Warp sees no CUDA device. Check NVIDIA driver / GPU.")
        return 1

    dev = cuda_devices[0]
    # device.arch is the compute capability as an int, e.g. 120 == sm_120 (Blackwell)
    print(f"CUDA device: {dev.name}  (sm_{dev.arch})")
    print(f"  is_cuda={dev.is_cuda}  pci={getattr(dev, 'pci_bus_id', '?')}")

    # 2. Tiny kernel — forces JIT for this exact architecture ----------
    n = 16
    a = wp.zeros(n, dtype=wp.float32, device=dev)
    wp.launch(fill_kernel, dim=n, inputs=[a], device=dev)
    wp.synchronize_device(dev)

    result = a.numpy()
    expected = np.arange(n, dtype=np.float32) * 2.0
    if not np.array_equal(result, expected):
        print(f"FAIL: kernel result mismatch.\n  got={result}\n  want={expected}")
        return 1
    print(f"Kernel JIT + launch OK: result[:4]={result[:4]}  (sm_{dev.arch} compiled)")

    # 3. numpy -> Zarr -> xarray round-trip ----------------------------
    grid = result.reshape(4, 4)
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "smoke.zarr"
        ds = xr.DataArray(grid, dims=("y", "x"), name="depth").to_dataset()
        ds.to_zarr(store, mode="w", consolidated=False)
        back = xr.open_zarr(store, consolidated=False)["depth"].values
        if not np.array_equal(back, grid):
            print("FAIL: Zarr round-trip mismatch.")
            return 1
    print(f"Zarr round-trip OK: shape={grid.shape}, dtype={grid.dtype} via zarr {zarr.__version__}")

    print("-" * 60)
    print("ALL GREEN - toolchain is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
