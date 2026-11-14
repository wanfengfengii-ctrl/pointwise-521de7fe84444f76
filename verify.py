#!/usr/bin/env python3
"""Independent verification for correlation.nearest().

Everything needed to re-check a result is re-implemented here from scratch
(Jacobi eigendecomposition, PSD projection, dual lower bound, an independent
Dykstra solver and a naive eigenvalue-truncation reference) so the check
does not rely on the internals of the module under test.

Run:  python3 -I verify.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import correlation  # noqa: E402


# --------------------------------------------------------------------------- #
# Independent linear-algebra primitives
# --------------------------------------------------------------------------- #

def eigh(A):
    """Standalone cyclic Jacobi eigendecomposition (rows = eigenvectors)."""
    n = len(A)
    S = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scale = math.sqrt(sum(x * x for row in S for x in row)) or 1.0
    for _ in range(150):
        worst = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = S[p][q]
                worst = max(worst, abs(apq))
                if abs(apq) <= 1e-15 * scale:
                    continue
                app, aqq = S[p][p], S[q][q]
                t_theta = (aqq - app) / (2.0 * apq)
                root = math.sqrt(1.0 + t_theta * t_theta)
                t = 1.0 / (t_theta + root) if t_theta >= 0 else 1.0 / (t_theta - root)
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                S[p][p] = app - t * apq
                S[q][q] = aqq + t * apq
                S[p][q] = S[q][p] = 0.0
                for k in range(n):
                    if k in (p, q):
                        continue
                    akp, akq = S[k][p], S[k][q]
                    S[k][p] = S[p][k] = c * akp - s * akq
                    S[k][q] = S[q][k] = s * akp + c * akq
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
        if worst <= 1e-14 * scale:
            break
    vals = [S[i][i] for i in range(n)]
    vecs = [[V[i][k] for i in range(n)] for k in range(n)]
    return vals, vecs


def psd_part(B):
    """B+ : zero negative eigenvalues. Returns (Bplus, frobenius_sq(B+))."""
    n = len(B)
    vals, vecs = eigh(B)
    P = [[0.0] * n for _ in range(n)]
    fro = 0.0
    for lam, v in zip(vals, vecs):
        if lam <= 0.0:
            continue
        fro += lam * lam
        for i in range(n):
            for j in range(n):
                P[i][j] += lam * v[i] * v[j]
    return P, fro


def fro_sq(M):
    return sum(x * x for row in M for x in row)


def dual_lower_bound(A, y):
    """Independent recomputation of D = ||A||^2/2 - ||B+||^2/2 - sum(y)."""
    B = [row[:] for row in A]
    for i in range(len(B)):
        B[i][i] -= y[i]
    _, bp_fro = psd_part(B)
    return 0.5 * fro_sq(A) - 0.5 * bp_fro - sum(y)


def dykstra_reference(A, iters=3000, tol=1e-12):
    """Independent Dykstra projection, written differently from the module.

    Two-increment formulation with a full-matrix increment for the PSD-cone
    projection and a diagonal-only increment for the unit-diagonal set.
    """
    n = len(A)
    X = [row[:] for row in A]
    P = [[0.0] * n for _ in range(n)]
    q = [0.0] * n
    for _ in range(iters):
        Y = [[X[i][j] + P[i][j] for j in range(n)] for i in range(n)]
        Z, _ = psd_part(Y)
        for i in range(n):
            for j in range(n):
                P[i][j] = Y[i][j] - Z[i][j]
        err = 0.0
        for i in range(n):
            err = max(err, abs(Z[i][i] - 1.0))
            q[i] = Z[i][i] + q[i] - 1.0
            Z[i][i] = 1.0
        X = Z
        if err <= tol:
            break
    return X


def naive_truncate_once(A):
    """The deliberately insufficient reference: truncate negatives once,
    then rescale rows/columns once to restore the unit diagonal."""
    P, _ = psd_part(A)
    n = len(A)
    out = [[0.0] * n for _ in range(n)]
    d = [math.sqrt(max(P[i][i], 1e-300)) for i in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = P[i][j] / (d[i] * d[j])
        out[i][i] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

FAILURES = []


def check(condition, message):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def check_solution(A, result, label, permutation_check=True):
    X, y = result["matrix"], result["dual"]
    n = len(A)
    print(f"\n=== {label} (n={n}) ===")

    # Structure constraints.
    check(isinstance(X, list) and len(X) == n and all(len(r) == n for r in X),
          "matrix has the requested shape")
    check(all(abs(X[i][j] - X[j][i]) <= 1e-8 for i in range(n) for j in range(n)),
          "matrix symmetric within 1e-8")
    check(all(abs(X[i][i] - 1.0) <= 1e-8 for i in range(n)),
          "unit diagonal within 1e-8")
    vals, _ = eigh(X)
    check(min(vals) >= -1e-8, f"PSD: min eigenvalue = {min(vals):.3e} >= -1e-8")
    check(all(abs(x) <= 1.0 + 1e-8 for row in X for x in row),
          "entries stay within [-1, 1]")
    check(len(y) == n and all(math.isfinite(v) for v in y),
          "dual is a finite vector of order n")

    # Objectives and the independently recomputed lower bound.
    F = 0.5 * fro_sq([[X[i][j] - A[i][j] for j in range(n)] for i in range(n)])
    D = dual_lower_bound(A, y)
    scale = max(1.0, fro_sq(A))
    gap = F - D
    print(f"  F = {F:.12e}   D = {D:.12e}   gap F-D = {gap:.3e}")
    check(-1e-8 <= gap, f"gap not below -1e-8 (got {gap:.2e})")
    check(gap <= 1e-7 * scale, f"gap <= 1e-7*max(1,||A||^2) = {1e-7 * scale:.2e}")

    # B+ must equal the returned matrix (primal-dual consistency).
    B = [row[:] for row in A]
    for i in range(n):
        B[i][i] -= y[i]
    Bp, _ = psd_part(B)
    resid = max(abs(Bp[i][j] - X[i][j]) for i in range(n) for j in range(n))
    check(resid <= 1e-7, f"B+ equals the returned matrix (residual {resid:.2e})")

    # Independent Dykstra cross-check (small/medium orders only).
    if n <= 6:
        Xref = dykstra_reference(A)
        dref = max(abs(Xref[i][j] - X[i][j]) for i in range(n) for j in range(n))
        check(dref <= 1e-7, f"matches independent Dykstra solution (diff {dref:.2e})")

    # Simultaneous row+column permutation equivariance.
    if permutation_check:
        perm = list(range(n))
        random.Random(1234 + n).shuffle(perm)
        Ap = [[A[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
        Xp = correlation.nearest(Ap)["matrix"]
        err = max(abs(Xp[i][j] - X[perm[i]][perm[j]])
                  for i in range(n) for j in range(n))
        check(err <= 1e-7, f"equivariant under simultaneous permutation (err {err:.2e})")


def main():
    random.seed(20261114)

    # 1. The specification's canonical example.
    A3 = [[1.0, -0.75, -0.75],
          [-0.75, 1.0, -0.75],
          [-0.75, -0.75, 1.0]]
    saved = [row[:] for row in A3]
    r3 = correlation.nearest(A3)
    check(A3 == saved, "input A is not modified")
    off = r3["matrix"][0][1]
    check(abs(off - (-0.5)) <= 1e-8, f"3x3 equicorrelation: off-diagonal = {off:.6f} == -0.5")
    check_solution(A3, r3, "canonical 3x3, all off-diagonals -0.75")

    # 2. Trivial 2x2 cases (every 2x2 unit-diagonal matrix is PSD).
    A2 = [[1.0, 0.3], [0.3, 1.0]]
    r2 = correlation.nearest(A2)
    check(max(abs(r2["matrix"][i][j] - A2[i][j]) for i in range(2) for j in range(2))
          <= 1e-8, "feasible 2x2 returned unchanged")
    check_solution(A2, r2, "feasible 2x2", permutation_check=False)

    # 3. PSD inputs of several orders must be preserved within 1e-8.
    for n in (3, 5, 10):
        G = [[random.uniform(-1.0, 1.0) for _ in range(n + 2)] for _ in range(n)]
        M = [[sum(G[i][k] * G[j][k] for k in range(n + 2)) for j in range(n)]
             for i in range(n)]
        d = [math.sqrt(M[i][i]) for i in range(n)]
        Apsd = [[M[i][j] / (d[i] * d[j]) for j in range(n)] for i in range(n)]
        for i in range(n):
            Apsd[i][i] = 1.0
        rpsd = correlation.nearest(Apsd)
        err = max(abs(rpsd["matrix"][i][j] - Apsd[i][j])
                  for i in range(n) for j in range(n))
        check(err <= 1e-8, f"PSD input preserved within 1e-8 (n={n}, err {err:.2e})")
        check_solution(Apsd, rpsd, f"already-PSD random matrix n={n}")

    # 4. Random non-PSD matrices across the full order range.
    nonpsd_found = 0
    for n in range(2, 21):
        for trial in range(3):
            A = [[0.0] * n for _ in range(n)]
            for i in range(n):
                A[i][i] = 1.0
                for j in range(i):
                    v = random.uniform(-1.0, 1.0)
                    A[i][j] = A[j][i] = v
            vals, _ = eigh(A)
            if min(vals) >= -1e-10:
                continue
            nonpsd_found += 1
            r = correlation.nearest(A)
            check_solution(A, r, f"random non-PSD n={n} trial={trial}",
                           permutation_check=(n <= 12))
    check(nonpsd_found >= 20, f"exercised {nonpsd_found} genuinely non-PSD random cases")

    # 5. Singular optimum / repeated eigenvalues: equicorrelation family.
    for rho in (-0.9, -0.6, -0.5, 0.9):
        n = 6
        Ae = [[1.0 if i == j else rho for j in range(n)] for i in range(n)]
        re = correlation.nearest(Ae)
        check_solution(Ae, re, f"equicorrelation rho={rho} (repeated eigenvalues)")

    # A genuinely non-PSD block-structured matrix (block 1 violates PSD);
    # exercises a solution with a rank-deficient positive part.
    n = 5
    # Leading 3x3 block is the canonical non-PSD -0.75 equicorrelation
    # triple; the remaining two names form a PSD block linked by weak
    # cross-correlations, producing a rank-deficient positive part.
    Ab = [
        [1.00, -0.75, -0.75, 0.20, 0.10],
        [-0.75, 1.00, -0.75, 0.15, 0.05],
        [-0.75, -0.75, 1.00, -0.10, 0.00],
        [0.20, 0.15, -0.10, 1.00, 0.60],
        [0.10, 0.05, 0.00, 0.60, 1.00],
    ]
    vals0, _ = eigh(Ab)
    check(min(vals0) < -1e-10, f"block case is genuinely non-PSD "
                               f"(min eig {min(vals0):.3f})")
    rb = correlation.nearest(Ab)
    check_solution(Ab, rb, "non-PSD block structure (rank-deficient optimum)")

    # 6. The naive one-shot truncation must be demonstrably suboptimal.
    beaten = 0
    for n in (4, 6, 8):
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            A[i][i] = 1.0
            for j in range(i):
                v = random.uniform(-1.0, 1.0)
                A[i][j] = A[j][i] = v
        vals, _ = eigh(A)
        if min(vals) >= -1e-10:
            continue
        X = correlation.nearest(A)["matrix"]
        Xn = naive_truncate_once(A)
        F_opt = 0.5 * fro_sq([[X[i][j] - A[i][j] for j in range(n)]
                              for i in range(n)])
        F_naive = 0.5 * fro_sq([[Xn[i][j] - A[i][j] for j in range(n)]
                                for i in range(n)])
        if F_naive - F_opt > 1e-6:
            beaten += 1
    check(beaten >= 1, f"one-shot truncate+renormalize is suboptimal on {beaten} case(s)")

    # 7. Published literature example (Qi & Sun 2006, 3x3).
    Aq = [[1.0, 1.0, 0.0],
          [1.0, 1.0, 0.95],
          [0.0, 0.95, 1.0]]
    rq = correlation.nearest(Aq)
    check_solution(Aq, rq, "Qi & Sun (2006) 3x3 example")

    # 8. Input validation.
    bad_inputs = [
        [[1.0, 0.5], [0.5]],                       # not square
        [[1.0]],                                   # order 1
        [[1.0] * 21 for _ in range(21)],           # order 21
        [[1.0, 2.0], [2.0, 1.0]],                  # entry out of range
        [[1.0, 0.5], [0.5, 0.0]],                  # bad diagonal
    ]
    for bad in bad_inputs:
        try:
            correlation.nearest(bad)
            check(False, f"invalid input rejected: {bad[0][:2]}...")
        except (ValueError, TypeError):
            check(True, "invalid input rejected")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED")
        for msg in FAILURES:
            print("  -", msg)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
