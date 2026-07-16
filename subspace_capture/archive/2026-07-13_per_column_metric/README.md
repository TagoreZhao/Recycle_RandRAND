# Archived: capture results with the per-column (basis-dependent) metric

Archived 2026-07-13. These are the original outputs of the three capture
studies (`run_subspace_capture.m`, `run_krylov_capture.m`,
`run_inverse_subspace_iter.m`) computed with the OLD metric in
`+src/+precond/subspace_capture.m`:

    residual_per_vec(i) = ||v_i - P_comp v_i|| / ||v_i||

and its aggregates (`max_residual`, `mean_residual`, `capture_frac_1pct`).

## Why archived

The per-column residual depends on the specific eigenvector basis returned
by `eigs`. On the sphere the spectrum has near-degenerate clusters, so the
basis within a cluster is arbitrary and the per-column numbers are not
basis-invariant. In addition, all columns passing the 1% threshold does NOT
imply the subspace as a whole is captured to 1%:

    max_residual <= ||(I - P_comp) Q_true||_2 <= sqrt(k) * max_residual.

The studies were rerun with basis-invariant directed principal-angle
metrics (`eigspace_err_2 = ||(I - P_comp) Q_true||_2`, `eigspace_err_fro`,
and angle-based capture counts) implemented in
`subspace_capture/subspace_capture_directed.m`.

## Contents

- `output/`                — polynomial-filter ablation results + plots and
  the (metric-independent) spectrum/spy diagnostics.
- `output_krylov_capture/` — (P)CG Krylov-subspace capture study.
- `output_inverse_iter/`   — exact-inverse subspace-iteration study.

The eigendecomposition caches were NOT archived; they are metric-independent
and remain in `subspace_capture/output/cache/` for reuse by the reruns.
