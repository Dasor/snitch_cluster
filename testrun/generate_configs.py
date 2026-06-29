#!/usr/bin/env python3
"""
Generate N random tile configurations for a given GEMM size (M, N, K),
respecting all hardware constraints, and output features in the same CSV
format produced by myrtle/scripts/tile_metrics.py.

Usage:
    python generate_configs.py M N K N_configs [--seed SEED] [--out PATH]
                               [--allow-boundary]

The output CSV is ready to feed directly into:
  - testrun/run_iteration1.sh  (via compile step)
  - testrun/train/boost.py predict --model ...

Hardware constraints enforced (default — exact-divisor mode):
  M % m == 0  (m must exactly divide M — no remainder tiles)
  N % n == 0  (n must exactly divide N — no remainder tiles)
  n is a multiple of 8  (SSR unroll-and-jam factor)
  K % k == 0  (k must exactly divide K — no remainder tiles)
  k >= 8 if K >= 8, else k >= 3  (HW loop prologue/epilogue minimum)

With --allow-boundary (boundary/remainder-tile mode, requires gemm_boundary):
  m can be any value >= 8 (remainder M-tile is handled by the boundary kernel)
  n is still a multiple of 8; if N % n != 0 the N-remainder must also be a
    multiple of 8  (SSR width constraint on the boundary tile)
  k can be any value >= 8; if K % k != 0 the K-remainder must be >= 3
    (HW loop prologue/epilogue minimum on the boundary tile)
  remainderTiles tag is set to e.g. "M0K" to indicate which dims have a
    boundary tile (used by prepareParams.py with gemm_boundary gemmDir)
"""

import argparse
import csv
import os
import random
import sys

# Reach the myrtle library the same way tile_metrics.py does
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../myrtle/myrtle"))

from tile_static_analysis.TSA_C_Remainder import TSA_C_Remainder
from tile_static_analysis.remainder_utils import TilingScheme

# Hardware parameters (matches TSG_C_Remainder defaults)
L1_BYTES = 100_000          # usable heap in TCDM scratchpad (used for feature columns)
BYTES_PER_ELEM = 8          # FP64
MAX_ATTEMPTS = 100_000      # give up if we can't find enough valid configs


def _valid_n_values(N: int, boundary: bool) -> list[int]:
    """Multiples of 8 that divide N exactly, or (with boundary) where N % n is also a multiple of 8."""
    if boundary:
        return [n for n in range(8, N + 1, 8) if N % n == 0 or N % n % 8 == 0]
    return [n for n in range(8, N + 1, 8) if N % n == 0]


def _valid_m_values(M: int, boundary: bool) -> list[int]:
    """Exact divisors of M ≥ 8, or (with boundary) all values ≥ 8."""
    min_m = 8 if M >= 8 else 1
    if boundary:
        return list(range(min_m, M + 1))
    return [m for m in range(min_m, M + 1) if M % m == 0]


def _valid_k_values(K: int, boundary: bool) -> list[int]:
    """Exact divisors of K ≥ 8 (or ≥ 3 when K < 8), or (with boundary) any value where the
    K-remainder is either 0 or ≥ 3 (HW loop prologue/epilogue minimum on the boundary tile)."""
    min_k = 8 if K >= 8 else 3
    if boundary:
        return [k for k in range(min_k, K + 1) if K % k == 0 or K % k >= 3]
    return [k for k in range(min_k, K + 1) if K % k == 0]


def _is_valid(m: int, n: int, k: int) -> bool:
    return True


def _remainder_tag(M: int, N: int, K: int, m: int, n: int, k: int) -> str:
    """Return the remainder-tile tag string, e.g. 'M0K', '000', 'MNK'."""
    return (
        ("M" if M % m else "0") +
        ("N" if N % n else "0") +
        ("K" if K % k else "0")
    )


def generate(M: int, N: int, N_configs: int, K: int, rng: random.Random,
             boundary: bool = False) -> list[dict]:
    valid_n = _valid_n_values(N, boundary)
    valid_m = _valid_m_values(M, boundary)
    valid_k = _valid_k_values(K, boundary)

    if not valid_n:
        sys.exit(f"No valid n values for N={N}. N must be ≥ 8 and have at least one divisor that is a multiple of 8.")
    if not valid_m:
        sys.exit(f"No valid m values for M={M}.")
    if not valid_k:
        sys.exit(f"No valid k values for K={K}.")

    tsa = TSA_C_Remainder(unrollAndJamFactor=8, degreeOfParallelism=8)
    seen = set()
    rows = []
    attempts = 0

    while len(rows) < N_configs and attempts < MAX_ATTEMPTS:
        attempts += 1
        m = rng.choice(valid_m)
        n = rng.choice(valid_n)
        k = rng.choice(valid_k)

        if not _is_valid(m, n, k):
            continue
        key = (m, n, k)
        if key in seen:
            continue
        seen.add(key)

        remainder = _remainder_tag(M, N, K, m, n, k)
        ts = TilingScheme(M, N, K, m, n, k, u=8, p=8, remainderTiles=remainder)
        metrics = tsa.tilingSchemeMetrics(ts)

        m_prime = m // 8
        row = {
            "FakeNN JSON Name": f"{M}x{N}x{K}w{m}-{n}-{k}",
            "M": M, "N": N, "K": K,
            "m": m, "n": n, "k": k,
            "JSON Name": f"{m}-{n}-{k}",
            "remainderTiles": remainder,
            "Space Needed in L1": 2 * (m*k + n*k + m*n) * BYTES_PER_ELEM,
            "Row Dim": n,
            "tileB_cc": n * k * BYTES_PER_ELEM,
            "Reduction Dim": k,
            "tileA": m * k * BYTES_PER_ELEM,
            "tileB": n * k * BYTES_PER_ELEM,
            "tileC_cc": m_prime * n * BYTES_PER_ELEM,
            "tileA_cc": m_prime * k * BYTES_PER_ELEM,
            "Space Remaining": L1_BYTES - 2 * (m*k + n*k + m*n) * BYTES_PER_ELEM,
            "tileC": m * n * BYTES_PER_ELEM,
            "m Dim": m,
            "Weight Matrix Tile Size": 2 * n * k * BYTES_PER_ELEM,
        }
        row.update(metrics)
        rows.append(row)

    if len(rows) < N_configs:
        print(
            f"Warning: only found {len(rows)} unique valid configs after "
            f"{attempts} attempts (requested {N_configs}). "
            "The valid search space may be smaller than N_configs.",
            file=sys.stderr,
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate random valid tile configurations for a GEMM size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("M", type=int, help="GEMM M dimension")
    parser.add_argument("N", type=int, help="GEMM N dimension")
    parser.add_argument("K", type=int, help="GEMM K dimension")
    parser.add_argument("N_configs", type=int, help="number of configurations to generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducibility")
    parser.add_argument("--out", default="-", metavar="PATH",
                        help="output CSV path (default: stdout)")
    parser.add_argument("--allow-boundary", action="store_true",
                        help="include boundary/remainder-tile configurations (requires gemm_boundary kernel)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = generate(args.M, args.N, args.N_configs, args.K, rng,
                    boundary=args.allow_boundary)
    rows.sort(key=lambda r: r["Space Needed in L1"], reverse=True)

    if not rows:
        sys.exit("No valid configurations found.")

    out = open(args.out, "w", newline="") if args.out != "-" else sys.stdout
    writer = csv.DictWriter(out, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    if args.out != "-":
        out.close()
        print(f"Wrote {len(rows)} configurations to {args.out}")


if __name__ == "__main__":
    main()
