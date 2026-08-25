# Promoted variable-viscosity Stokes rotor iteration benchmark

This sibling benchmark applies the existing
[`stokes_varvisc_rotor`](../stokes_varvisc_rotor/README.md) 11-solver
iterations-versus-timestep comparison to the two physical configurations
promoted by the verified upgrade. It intentionally does not copy the upgrade's
spectral screening or validation pipeline.

The cases are:

- `current_channel_ar4`: a 4:1 through-flow channel, three-point rotating bar,
  and smooth moving 50:1 viscosity field (`0.04` to `2.0`).
- `mixer_circle_four_blade`: a closed unit disk, nine-marker rotating cross,
  and asymmetric moving 50:1 viscosity field (`0.02` to `1.0`).

Both use `h=0.05`, backward Euler with `dt=0.02` for 60 solves through
`Tmax=1.2`, and radius-`0.12` finite-radius immersed constraints. The solver
settings come from `varvisc_default_benchmark_params` in the parent benchmark.

## Run

From this directory in MATLAB:

```matlab
test_upgraded_varvisc_components
SMOKE_TEST = true; run_upgraded_varvisc_benchmark
clear SMOKE_TEST; run_upgraded_varvisc_benchmark
validate_upgraded_varvisc_results
replot_upgraded_varvisc_benchmark
```

The full run writes `benchmark_varvisc_upgraded/`, including
`all_results.csv`, `run_config.{mat,json}`, summary tables, per-solver plots,
and the requested figures:

```text
benchmark_varvisc_upgraded/
  iteration_vs_timestep/
    current_channel_ar4.png
    current_channel_ar4_linear.png
    mixer_circle_four_blade.png
    mixer_circle_four_blade_linear.png
  summary_plots/
    all_cases_comparison.png
    all_cases_comparison_linear.png
```

Files ending in `_linear.png` use an ordinary linear iteration axis; the
unsuffixed figures retain the logarithmic axis for viewing all solver scales.

Generated full and smoke result directories are retained locally but ignored
by Git.
