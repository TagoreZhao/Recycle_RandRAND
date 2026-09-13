# Promoted variable-viscosity Stokes rotor iteration benchmark

This sibling benchmark applies the existing
[`stokes_varvisc_rotor`](../stokes_varvisc_rotor/README.md) solver machinery to
the two physical configurations promoted by the verified upgrade. The focused
production experiment compares refreshed ILDL against stale rank-20 deflation
with fixed and dynamically selected coarse weights. The earlier 11-solver run
is retained separately and is not overwritten.

The cases are:

- `current_channel_ar4`: a 4:1 through-flow channel, three-point rotating bar,
  and smooth moving 50:1 viscosity field (`0.04` to `2.0`).
- `mixer_circle_four_blade`: a closed unit disk, nine-marker rotating cross,
  and asymmetric moving 50:1 viscosity field (`0.02` to `1.0`).

Both use `h=0.05`, backward Euler with `dt=0.02` for 60 solves through
`Tmax=1.2`, and radius-`0.12` finite-radius immersed constraints. These values
come from the verified upgraded parameter bundle.

In the focused experiment, no-fill ILDL is rebuilt for every system while the
exact rank-20 small-edge deflation basis is built only at step 1 and kept stale
in physical coordinates. For each current KKT system, the dynamic arm uses

```text
cutoff_n = abs(lambda_21(C_n^{-1} K_n C_n^{-T}))
tau_n    = cutoff_n^2
```

The fixed control uses the same rank-20 basis with `tau=0.5`, so the only
difference between the two deflation arms is tau selection.

## Run

From this directory in MATLAB:

```matlab
test_upgraded_varvisc_components
test_dynamic_tau_deflation
SMOKE_TEST = true; run_upgraded_varvisc_dynamic_tau_benchmark
validate_dynamic_tau_varvisc_results('benchmark_varvisc_upgraded_dynamic_tau_smoke')
clear SMOKE_TEST; run_upgraded_varvisc_dynamic_tau_benchmark
validate_dynamic_tau_varvisc_results
replot_dynamic_tau_varvisc_benchmark
```

The full focused run writes `benchmark_varvisc_upgraded_dynamic_tau/`, including
`all_results.csv`, `dynamic_tau_summary.{csv,txt}`,
`run_config.{mat,json}`, per-solver CSVs, and the requested figures:

```text
benchmark_varvisc_upgraded_dynamic_tau/
  iteration_vs_timestep/
    current_channel_ar4_linear.png
    current_channel_ar4_log.png
    mixer_circle_four_blade_linear.png
    mixer_circle_four_blade_log.png
  summary_plots/
    all_cases_comparison_linear.png
    all_cases_comparison_log.png
```

Files ending in `_linear.png` use an ordinary linear iteration axis with integer
ticks; the matching `_log.png` files are companions for scale comparison. Exact
iterations, dynamic tau, spectral cutoff, true residual, solution error, and
end-to-end timing are retained in the CSV files.

The legacy `run_upgraded_varvisc_benchmark` entry point still reproduces the
archived 11-solver comparison in `benchmark_varvisc_upgraded/`.

Generated full and smoke result directories are retained locally but ignored
by Git.
