#!/usr/bin/env python3
"""Demonstration of correlation.nearest() on realistic monitoring data.

Scenarios
---------
1. Equity-sector monitoring matrix: pairwise complete-case correlations
   computed by an analyst from return series with missing observations.
   The sample correlation matrix is not PSD (the pairwise-complete /
   missing-data problem), so joint portfolio analysis (which requires a
   PSD covariance) would fail without correction.

2. Generic entry point: any valid matrix supplied as JSON on the command
   line or via a file, e.g.
       python3 -I demo.py '[[1,0.9],[0.9,1]]'
       python3 -I demo.py --file matrix.json

Run:  python3 -I demo.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import correlation  # noqa: E402


def _print_matrix(title, M, digits=6):
    print(title)
    width = digits + 4
    for row in M:
        print("  " + "  ".join(f"{v: {width}.{digits}f}" for v in row))
    print()


def _eig_min(M):
    vals, _ = correlation.jacobi_eigh(M)
    return min(vals)


def scenario_equity_sectors():
    """Pairwise-complete correlation matrix from equity-sector returns.

    Entries emulate an analyst's workbook: correlations were estimated on
    overlapping but non-identical observation windows (holidays, halted
    names, vendor gaps), so the resulting matrix fails positive
    semidefiniteness even though every diagonal equals 1 and every entry
    lies in [-1, 1].  This is the common "barely infeasible" case: the
    violation is a small negative eigenvalue rather than an extreme entry.
    """
    labels = ["OilProducerA", "OilProducerB", "Airline", "Utilities",
              "Healthcare", "Technology"]
    # Symmetric pairwise-complete sample correlations (unit diagonal).
    A = [
        [1.00, 0.95, -0.90, 0.25, -0.05, 0.30],
        [0.95, 1.00, -0.90, 0.30, 0.05, 0.35],
        [-0.90, -0.90, 1.00, 0.10, 0.20, -0.10],
        [0.25, 0.30, 0.10, 1.00, 0.22, 0.18],
        [-0.05, 0.05, 0.20, 0.22, 1.00, 0.45],
        [0.30, 0.35, -0.10, 0.18, 0.45, 1.00],
    ]

    print("=" * 72)
    print("Scenario 1: equity-sector monitoring (pairwise-complete sample)")
    print("=" * 72)
    _print_matrix("Analyst-supplied correlation matrix A:", A)
    lam_min = _eig_min(A)
    status = "PSD, no correction needed" if lam_min >= -1e-12 else "NOT PSD"
    print(f"Smallest eigenvalue of A: {lam_min:.6f}  -> {status}")
    print()

    result = correlation.nearest(A)
    X, y = result["matrix"], result["dual"]
    _print_matrix("Nearest valid correlation matrix X (Frobenius projection):", X)
    print(f"Smallest eigenvalue of X: {_eig_min(X):.3e} (>= -1e-8 required)")

    # Independent lower bound recomputed from the dual (see verify.py for
    # the standalone implementation; here we restate the formula inline).
    B = [row[:] for row in A]
    for i in range(len(B)):
        B[i][i] -= y[i]
    Bp, vals_b, _, bp_fro = correlation._psd_project(B)
    fro = lambda M: sum(v * v for row in M for v in row)
    F = 0.5 * fro([[X[i][j] - A[i][j] for j in range(len(A))]
                   for i in range(len(A))])
    D = 0.5 * fro(A) - 0.5 * bp_fro - sum(y)
    print(f"Dual vector y: {[round(v, 6) for v in y]}")
    print(f"Eigenvalues of B = A - diag(y): {[round(v, 6) for v in vals_b]}")
    print(f"Primal objective F = ||X-A||^2/2          = {F:.10f}")
    print(f"Dual lower bound D (recomputed from y)   = {D:.10f}")
    print(f"Optimality gap F - D                     = {F - D:.3e}")
    consistency = max(abs(Bp[i][j] - X[i][j])
                      for i in range(len(A)) for j in range(len(A)))
    print(f"B+ equals X (max absolute difference)    = {consistency:.3e}")
    max_change = max(abs(X[i][j] - A[i][j])
                     for i in range(len(A)) for j in range(len(A))
                     if i != j)
    print(f"Largest off-diagonal adjustment          = {max_change:.6f}")
    print("Sectors, in order:", ", ".join(labels))
    print()


def scenario_bad_pair():
    """A small two-name example where the raw pair correlation is infeasible
    in a larger system (2x2 alone is always PSD; embed it in a 4-name book).
    """
    print("=" * 72)
    print("Scenario 2: four-name credit book with an inconsistent block")
    print("=" * 72)
    # Leading 3x3 block is the canonical -0.75 equicorrelation triple
    # (eigenvalues -0.5, 1.25, 1.25), which guarantees non-PSD structure;
    # a fourth, weakly linked name is added.
    A = [
        [1.00, -0.75, -0.75, 0.20],
        [-0.75, 1.00, -0.75, 0.15],
        [-0.75, -0.75, 1.00, -0.10],
        [0.20, 0.15, -0.10, 1.00],
    ]
    _print_matrix("Raw pairwise correlations A:", A)
    lam_min = _eig_min(A)
    status = "PSD" if lam_min >= -1e-12 else "NOT PSD"
    print(f"Smallest eigenvalue of A: {lam_min:.6f} -> {status}")
    result = correlation.nearest(A)
    X = result["matrix"]
    _print_matrix("Corrected matrix X:", X)
    print(f"Smallest eigenvalue of X: {_eig_min(X):.3e}")
    fro = lambda M: sum(v * v for row in M for v in row)
    F = 0.5 * fro([[X[i][j] - A[i][j] for j in range(4)] for i in range(4)])
    print(f"Minimum squared correction F = {F:.10f}")
    print()


def run_user_matrix(text):
    A = json.loads(text)
    result = correlation.nearest(A)
    _print_matrix("Input A:", A)
    _print_matrix("Nearest correlation matrix:", result["matrix"])
    print("Dual y:", result["dual"])


def main(argv):
    if len(argv) == 2 and not argv[1].startswith("--"):
        run_user_matrix(argv[1])
        return 0
    if len(argv) == 3 and argv[1] == "--file":
        with open(argv[2], "r", encoding="utf-8") as fh:
            run_user_matrix(fh.read())
        return 0

    scenario_equity_sectors()
    scenario_bad_pair()
    print("For custom input: python3 -I demo.py '<json matrix>' "
          "or python3 -I demo.py --file matrix.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
