"""Nearest correlation matrix solver (Python standard library only).

Given a symmetric matrix ``A`` with unit diagonal, solve the projection

    minimize   F(X) = 0.5 * ||X - A||_F^2
    subject to X = X^T,  diag(X) = 1,  X is positive semidefinite.

The problem is solved through its dual.  With dual vector ``y`` and

    B  = A - diag(y),   B+ = P_psd(B)  (negative eigenvalues set to zero),

the concave dual objective is

    g(y) = 0.5*||A||_F^2 - 0.5*||B+||_F^2 - sum(y),

whose value is the independent optimality lower bound ``D`` requested in the
specification.  Strong duality holds (Slater's condition: the identity
matrix is strictly feasible), so max g = min F.

We maximize g with a guarded (semismooth) Newton method on the gradient
    h(y) = 1 - diag(B+).
A Dykstra alternating-projection routine is used as a fallback for singular
degenerate cases and also serves as the polishing engine.

Only the Python standard library is used.
"""

import math

__all__ = ["nearest"]


# --------------------------------------------------------------------------- #
# Basic linear algebra (symmetric matrices stored as lists of lists)
# --------------------------------------------------------------------------- #

def _copy_matrix(A):
    return [row[:] for row in A]


def _symmetrize(A):
    """Return an exactly-symmetric copy of A (averaging both triangles)."""
    n = len(A)
    return [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]


def _frobenius_sq(A):
    total = 0.0
    for i, row in enumerate(A):
        for j, a in enumerate(row):
            total += a * a
    return total


def jacobi_eigh(A, tol=1e-15, max_sweeps=120):
    """Eigen-decomposition of a real symmetric matrix.

    Returns ``(eigvals, eigvecs)`` where ``eigvecs[k]`` is the normalized
    eigenvector for ``eigvals[k]`` (rows are eigenvectors), so that

        A = sum_k eigvals[k] * v_k v_k^T.

    Cyclic-by-row Jacobi rotations; repeated eigenvalues are handled
    naturally since no eigenvalue splitting or gap logic is required.
    """
    n = len(A)
    S = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    scale = math.sqrt(_frobenius_sq(S))
    if scale == 0.0:
        return [0.0] * n, [row[:] for row in V]
    eps = tol * scale

    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = S[p][q]
                if abs(apq) <= eps:
                    continue
                off = max(off, abs(apq))
                app = S[p][p]
                aqq = S[q][q]
                theta = (aqq - app) / (2.0 * apq)
                if theta >= 0.0:
                    t = 1.0 / (theta + math.sqrt(1.0 + theta * theta))
                else:
                    t = 1.0 / (theta - math.sqrt(1.0 + theta * theta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                # Apply rotation to S: S <- J^T S J
                S[p][p] = app - t * apq
                S[q][q] = aqq + t * apq
                S[p][q] = 0.0
                S[q][p] = 0.0
                for k in range(n):
                    if k == p or k == q:
                        continue
                    akp = S[k][p]
                    akq = S[k][q]
                    S[k][p] = c * akp - s * akq
                    S[p][k] = S[k][p]
                    S[k][q] = s * akp + c * akq
                    S[q][k] = S[k][q]

                # Accumulate eigenvectors: V <- V J
                for k in range(n):
                    vkp = V[k][p]
                    vkq = V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
        if off <= eps:
            break

    eigvals = [S[i][i] for i in range(n)]
    eigvecs = [[V[i][k] for i in range(n)] for k in range(n)]
    return eigvals, eigvecs


def _psd_project(B):
    """Positive part of a symmetric matrix: zero out negative eigenvalues."""
    n = len(B)
    eigvals, eigvecs = jacobi_eigh(B)
    P = [[0.0] * n for _ in range(n)]
    trace_p = 0.0
    fro_sq = 0.0
    for k, lam in enumerate(eigvals):
        if lam <= 0.0:
            continue
        v = eigvecs[k]
        for i in range(n):
            vi = lam * v[i]
            Pi = P[i]
            for j in range(n):
                Pi[j] += vi * v[j]
        trace_p += lam
        fro_sq += lam * lam
    # Force exact symmetry against round-off.
    for i in range(n):
        for j in range(i + 1, n):
            avg = 0.5 * (P[i][j] + P[j][i])
            P[i][j] = P[j][i] = avg
    return P, eigvals, eigvecs, fro_sq


def _solve_linear(M, b):
    """Gaussian elimination with partial pivoting for a symmetric system.

    Raises ZeroDivisionError when the matrix is numerically singular.
    """
    n = len(M)
    U = [row[:] for row in M]
    rhs = b[:]
    for k in range(n):
        pivot = k
        best = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > best:
                best = abs(U[i][k])
                pivot = i
        if best < 1e-14:
            raise ZeroDivisionError("singular Jacobian")
        if pivot != k:
            U[k], U[pivot] = U[pivot], U[k]
            rhs[k], rhs[pivot] = rhs[pivot], rhs[k]
        pivot_val = U[k][k]
        for i in range(k + 1, n):
            factor = U[i][k] / pivot_val
            if factor == 0.0:
                continue
            U[i][k] = 0.0
            Ui = U[i]
            Uk = U[k]
            for j in range(k + 1, n):
                Ui[j] -= factor * Uk[j]
            rhs[i] -= factor * rhs[k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        Ui = U[i]
        for j in range(i + 1, n):
            s -= Ui[j] * x[j]
        x[i] = s / Ui[i]
    return x


# --------------------------------------------------------------------------- #
# Dykstra alternating projections (fallback / polishing / independent method)
# --------------------------------------------------------------------------- #

def _dykstra(A, tol=1e-12, max_iter=20000, X0=None):
    """Dykstra's alternating projections between

        C1 = {symmetric PSD matrices}
        C2 = {symmetric matrices with unit diagonal}.

    Standard two-increment formulation (P for C1, q for C2).  The C2
    projection only touches diagonal entries, so its increment q is
    diagonal.  ``X0`` may be a near-feasible matrix for a warm start.
    Returns ``(X, y)``: at the fixed point
    ``A = X - P - diag(q)`` with P in the normal cone of the PSD cone;
    comparing with the KKT identity  A = X + diag(y) - N_psd(X)  gives
    ``y = q`` and  X = (A - diag(y))_+.
    """
    n = len(A)
    X = _copy_matrix(X0) if X0 is not None else _copy_matrix(A)
    P = [[0.0] * n for _ in range(n)]  # C1 (PSD cone) increment
    q = [0.0] * n                      # C2 (unit diagonal) increment

    for _ in range(max_iter):
        # Projection onto C1.
        Y = [[X[i][j] + P[i][j] for j in range(n)] for i in range(n)]
        Z, _, _, _ = _psd_project(Y)
        for i in range(n):
            for j in range(n):
                P[i][j] = Y[i][j] - Z[i][j]

        # Projection onto C2: reset the diagonal to 1; off-diagonals untouched.
        diag_err = 0.0
        for i in range(n):
            err = abs(Z[i][i] - 1.0)
            if err > diag_err:
                diag_err = err
            val = Z[i][i] + q[i]
            q[i] = val - 1.0
            Z[i][i] = 1.0
        X = Z

        if diag_err <= tol:
            break

    # Dual recovery: the C2 increment q equals the dual vector y.
    y = q[:]
    return X, y


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def nearest(A):
    """Nearest correlation matrix to ``A``.

    Parameters
    ----------
    A : sequence of sequences of real numbers
        Symmetric matrix of order 2..20, with unit diagonal and all entries
        in [-1, 1].  The input is never modified.

    Returns
    -------
    dict
        ``{"matrix": X, "dual": y}`` where ``X`` is symmetric, has unit
        diagonal, is positive semidefinite and minimizes
        ``0.5*||X-A||_F^2`` under those constraints; ``y`` is the dual
        vector such that with ``B = A - diag(y)`` the primal optimum equals
        ``B+`` (negative eigenvalues of B zeroed).
    """
    # -- validation ------------------------------------------------------- #
    if not isinstance(A, (list, tuple)):
        raise TypeError("A must be a list or tuple of rows")
    n = len(A)
    if not 2 <= n <= 20:
        raise ValueError("matrix order must be between 2 and 20")
    for row in A:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise ValueError("A must be a square matrix")
        for a in row:
            if not isinstance(a, (int, float)) or not math.isfinite(a):
                raise ValueError("all entries must be finite real numbers")

    work = _symmetrize([[float(a) for a in row] for row in A])
    for i in range(n):
        if abs(work[i][i] - 1.0) > 1e-8:
            raise ValueError("diagonal entries must equal 1")
        for j in range(n):
            if work[i][j] < -1.0 - 1e-8 or work[i][j] > 1.0 + 1e-8:
                raise ValueError("all entries must lie in [-1, 1]")
        work[i][i] = 1.0

    norm_A_sq = _frobenius_sq(work)
    y = [0.0] * n

    X, eigvals, _, _ = _psd_project(work)

    # Fast path: A itself is feasible.
    if min(eigvals) >= -1e-12 and max(abs(X[i][i] - 1.0) for i in range(n)) <= 1e-12:
        X = _copy_matrix(work)
        for i in range(n):
            X[i][i] = 1.0
        return {"matrix": X, "dual": [0.0] * n}

    # -- semismooth Newton on the dual ------------------------------------ #
    converged = False
    max_newton = 60
    for _iteration in range(max_newton):
        # B = A - diag(y)
        B = [work[i][:] for i in range(n)]
        for i in range(n):
            B[i][i] = work[i][i] - y[i]

        Bplus, eigvals, eigvecs, p_fro_sq = _psd_project(B)
        g = 0.5 * (norm_A_sq - p_fro_sq) - sum(y)

        h = [1.0 - Bplus[i][i] for i in range(n)]
        hnorm = math.sqrt(sum(hi * hi for hi in h))
        if hnorm <= 1e-10:
            X = Bplus
            converged = True
            break

        # Semismooth Newton step.  The dual gradient is grad g = -h (since
        # dg/dy_i = -(1-(B+)_ii)).  Maximizing g therefore solves, with the
        # PSD generalized Jacobian Q of h,
        #     Q dy = -h.
        # For the spectral positive-part map phi(t)=max(t,0) the divided
        # difference equals 1 for every pair of *positive* eigenvalues
        # (repeated ones included).  Writing P_alpha for the projector onto
        # the positive eigenspace and using dB = -diag(dy),
        #     d(B+)_ii/dy_j = -(P_alpha)_ij^2,
        # so Q = P_alpha (.) P_alpha (entrywise square), which is PSD by the
        # Schur product theorem and needs no special-casing for repeats.
        pos = [k for k, lam in enumerate(eigvals) if lam > 0.0]
        projector = [[0.0] * n for _ in range(n)]
        for k in pos:
            v = eigvecs[k]
            for i in range(n):
                vi = v[i]
                pi = projector[i]
                for j in range(n):
                    pi[j] += vi * v[j]
        Q = [[projector[i][j] * projector[i][j] for j in range(n)]
             for i in range(n)]
        rhs = [-hi for hi in h]

        # Newton step: Q dy = -h, regularized if singular.
        dy = None
        for reg in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
            try:
                if reg:
                    Qr = [Q[i][:] for i in range(n)]
                    for i in range(n):
                        Qr[i][i] += reg
                    dy = _solve_linear(Qr, rhs)
                else:
                    dy = _solve_linear(Q, rhs)
                break
            except ZeroDivisionError:
                continue
        if dy is None:
            # Steepest-ascent fallback on the concave dual: dy = grad g = -h.
            dy = rhs[:]

        # Monotone backtracking line search on the concave (nonsmooth) dual.
        # Inside a region where the active set is fixed, g can be locally
        # linear (or flat), so a classical Armijo quadratic test only accepts
        # vanishingly small steps and the iteration never crosses an active
        # boundary.  Use the Qi-Sun style tolerance that is linear in step.
        step = 1.0
        accepted = False
        directional = sum((-h[i]) * dy[i] for i in range(n))
        if directional < 0.0:
            # Defensive: Q is PSD so this should not happen; use gradient.
            dy = rhs[:]
            directional = hnorm * hnorm
        tol_scale = 1e-12 * (1.0 + abs(g))
        g_try = g
        for _ls in range(80):
            y_try = [y[i] + step * dy[i] for i in range(n)]
            B_try = [work[i][:] for i in range(n)]
            for i in range(n):
                B_try[i][i] = work[i][i] - y_try[i]
            _, _, _, fro_try = _psd_project(B_try)
            g_try = 0.5 * (norm_A_sq - fro_try) - sum(y_try)
            if g_try >= g - tol_scale * step:
                accepted = True
                break
            step *= 0.5
        if not accepted or any(not math.isfinite(v) for v in y_try):
            # Line search failed: switch directly to the robust fallback.
            break
        y = y_try

    # ------------------------------------------------------------------ #
    # Robust solve / polish with Dykstra alternating projections.
    # Warm-started from the Newton iterate when one is available.
    # ------------------------------------------------------------------ #
    if converged:
        X, y = _dykstra(work, tol=1e-13, max_iter=3000, X0=X)
    else:
        X, y = _dykstra(work, tol=1e-13, max_iter=20000)

    # Final exact symmetry and unit diagonal guarantees.
    for i in range(n):
        for j in range(i + 1, n):
            avg = 0.5 * (X[i][j] + X[j][i])
            X[i][j] = X[j][i] = avg
        X[i][i] = 1.0

    # Ensure the dual is consistent with the returned matrix: B+ must equal X.
    B = [work[i][:] for i in range(n)]
    for i in range(n):
        B[i][i] = work[i][i] - y[i]
    Bp, _, _, _ = _psd_project(B)
    if max(abs(Bp[i][j] - X[i][j]) for i in range(n) for j in range(n)) > 1e-9:
        X, y = _dykstra(work, tol=1e-13, max_iter=20000)
        for i in range(n):
            for j in range(i + 1, n):
                avg = 0.5 * (X[i][j] + X[j][i])
                X[i][j] = X[j][i] = avg
            X[i][i] = 1.0

    return {"matrix": X, "dual": y}
