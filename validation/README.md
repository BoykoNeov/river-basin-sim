# Validation harness

Headless benchmark + regression tests. For a solo project the batch driver doubles
as the regression harness, so **these gate every milestone** (HANDOFF §10).

Designed to run on Warp's **CPU backend** so they execute in CI without a GPU.

| Test | Checks | Milestone |
|---|---|---|
| **Global mass balance** | float64/Kahan `inflow − outflow − Δstored` relative error below a fixed threshold | always (every run) |
| **Dam-break** | shock speed + wave shape vs the analytical Stoker/Ritter solution | M1 |
| **Lake-at-rest** | flat surface over arbitrary bed stays flat, zero velocity (well-balancedness) | M4 |
| **UK EA 2D benchmark suite** | realistic flood behaviour vs the standard battery | M4 |

Run with `uv run pytest`. A run whose mass-balance error exceeds threshold is a
**bug, not a warning** — treat exceedance as a failing test.
