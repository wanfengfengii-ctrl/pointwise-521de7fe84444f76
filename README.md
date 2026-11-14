# nearest correlation matrix — `correlation.nearest(A)`

Pure-Python (standard library only) correction of a pairwise-complete sample
correlation matrix that has lost positive semidefiniteness.

## Problem

Given a symmetric matrix `A` of order 2–20 with unit diagonal and entries in
`[-1, 1]`, solve

```
minimize   F(X) = 0.5 * ||X - A||_F^2
subject to X = X^T,  diag(X) = 1,  X is positive semidefinite (PSD).
```

This is the projection onto the intersection of the PSD cone and the affine
space of unit-diagonal symmetric matrices.  It is **not** enough to truncate
the negative eigenvalues of `A` and renormalize the diagonal once: `verify.py`
demonstrates numerically that this shortcut is strictly suboptimal.

Example (the specification case): for the 3×3 matrix whose off-diagonal
entries are all `-0.75`, the optimal corrected off-diagonal is exactly `-0.5`.

## Method

Dual formulation.  With dual vector `y`, `B = A - diag(y)` and `B+` the
spectral positive part of `B` (negative eigenvalues set to zero), the concave
dual objective is

```
g(y) = 0.5*||A||_F^2 - 0.5*||B+||_F^2 - sum(y).
```

Strong duality holds (the identity matrix is a strictly feasible point), so
`max_y g(y) = min_X F(X)`.  The requested independent lower bound is
`D = g(y)`.

1. **Semismooth Newton on the dual.**  Stationarity is `h(y) = 1 - diag(B+) = 0`.
   Since the gradient of the dual is `∇g = -h`, each step solves
   `Q dy = -h`, where for the positive-part map the generalized Jacobian is
   `Q = P_α ∘ P_α` (entrywise square of the projector onto the positive
   eigenspace of `B`).  `Q` is PSD by the Schur product theorem, so repeated
   eigenvalues need no special treatment.  A monotone backtracking line
   search with a Qi–Sun style tolerance handles the non-smooth active-set
   changes; the linear system uses Gaussian elimination with partial pivoting
   and small diagonal regularization when singular.
2. **Dykstra alternating projections** between the PSD cone and the
   unit-diagonal affine set (standard two-increment formulation), warm-started
   from the Newton iterate, provide the robust solve/fallback and recover the
   dual vector `y` (the diagonal increment equals `y`).
3. All eigen-decompositions use a self-contained cyclic **Jacobi eigensolver**
   — no third-party or NumPy dependency.

## Files

| File | Purpose |
|---|---|
| `correlation.py` | The deliverable: `nearest(A)` plus the internal solver primitives |
| `verify.py` | Standalone independent verification (its own Jacobi solver, PSD projection, lower-bound recomputation and Dykstra reference) |
| `demo.py` | Realistic monitoring-data demonstrations and a JSON/CLI entry point |
| `.pointwise-runtime.json` | Declared verification and demonstration commands |

## Usage

```python
import correlation

A = [[1.0, -0.75, -0.75],
     [-0.75, 1.0, -0.75],
     [-0.75, -0.75, 1.0]]

result = correlation.nearest(A)
result["matrix"]   # corrected PSD correlation matrix (unit diagonal)
result["dual"]     # dual vector y; B = A - diag(y), result["matrix"] == B+
```

`A` is never modified.  Invalid inputs (wrong order/shape, non-unit diagonal,
entries outside `[-1, 1]`, non-finite values) raise `ValueError`/`TypeError`.

## Running

```bash
python3 -I verify.py          # full verification suite (exit 0 = all pass)
python3 -I demo.py            # realistic-data demonstration
python3 -I demo.py '<json>'   # custom matrix, e.g. '[[1,-0.75,-0.75],...]'
python3 -I demo.py --file matrix.json
```

## What is verified

- symmetry and unit-diagonal errors ≤ `1e-8`;
- smallest eigenvalue ≥ `-1e-8`;
- `F - D` within the required band, with `D` recomputed independently from
  the returned `y` (reconstructing `B`, `B+` and the dual formula from
  scratch);
- primal–dual consistency `B+ == matrix`;
- agreement with an independently written Dykstra reference;
- equivariance under simultaneous row/column permutations;
- already-PSD inputs preserved to within `1e-8`;
- repeated-eigenvalue (equicorrelation) families, singular/block optima,
  random non-PSD matrices of every order 2–20, and invalid-input rejection;
- the naive one-shot truncate-and-renormalize approach shown to be suboptimal.

Typical runtime is well under one second for orders up to 20 with the standard
CPython interpreter (all code is standard-library Python).
