#!/usr/bin/env python3
"""
Repair for tile-size GA individuals.

Given an (m, n, k) triple, repair() snaps each dimension to the nearest valid
value (divisibility, SSR alignment, HW loop minimum).

Constraint definitions are imported from generate_configs.py.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from generate_configs import (
    _valid_k_values,
    _valid_m_values,
    _valid_n_values,
)


def _snap(val: int, sorted_valid: list[int]) -> int:
    return min(sorted_valid, key=lambda x: abs(x - val))


def repair(
    m: int,
    n: int,
    k: int,
    M: int,
    N: int,
    K: int,
    boundary: bool = True,
) -> tuple[int, int, int] | None:
    """
    Return the nearest valid (m, n, k) to the given triple, or None if
    no valid configuration exists for the given (M, N, K).

    Snaps each dimension independently to its closest valid value.
    """
    vm = sorted(_valid_m_values(M, boundary))
    vn = sorted(_valid_n_values(N, boundary))
    vk = sorted(_valid_k_values(K, boundary))

    if not vm or not vn or not vk:
        return None

    return _snap(m, vm), _snap(n, vn), _snap(k, vk)


if __name__ == "__main__":
    # Quick sanity checks
    cases = [
        # (m, n, k, M, N, K, boundary, expect_valid)
        (30, 32, 32, 128, 128, 128, True, True),
        (999, 999, 999, 128, 128, 128, True, True),   # must scale down
        (1, 1, 1, 128, 128, 128, True, True),          # must scale up (snap)
        (16, 16, 16, 128, 128, 128, False, True),      # exact mode
        (15, 15, 15, 128, 128, 128, False, True),      # odd — must snap
    ]
    from generate_configs import _is_valid as chk
    ok = True
    for m, n, k, M, N, K, bnd, expect in cases:
        r = repair(m, n, k, M, N, K, boundary=bnd)
        valid = r is not None and chk(*r)
        status = "OK" if valid == expect else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {status}  repair({m},{n},{k}, {M},{N},{K}, bnd={bnd}) → {r}  valid={valid}")
    raise SystemExit(0 if ok else 1)
