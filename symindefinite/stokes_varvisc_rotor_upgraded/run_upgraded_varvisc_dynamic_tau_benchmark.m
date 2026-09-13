% RUN_UPGRADED_VARVISC_DYNAMIC_TAU_BENCHMARK Focused ILDL/deflation study.
%
% Keeps the verified upgraded physical cases and KKT assembly unchanged.  The
% exact rank-20 basis is built at step 1 and kept stale in physical coordinates;
% ILDL is refreshed every step.  The dynamic arm chooses
% tau_n = |lambda_21(C_n^-1 K_n C_n^-T)|^2 for every current system.

%% Setup and focused production parameters
thisFileDir = fileparts(mfilename('fullpath'));
parentDir = fullfile(fileparts(thisFileDir), 'stokes_varvisc_rotor');
repoRoot = fileparts(fileparts(thisFileDir));
addpath(repoRoot, parentDir, thisFileDir);
import src.stokes.*
rng(1);

params = varvisc_default_benchmark_params();
params.h0 = 0.05;
params.dt = 0.02;
params.Tstep = 61;
params.DEFLAT_SM_EIG = 20;
params.DEFLAT_LG_EIG = 0;
params.DEFLAT_TAU = 0.5;
params.ILDL_MODE = 'nofill';
params.ILDL_PREC_REFRESH = 1;
params.DEFLAT_PREC_REFRESH = Inf;
params.SOLVER_PROFILE = 'dynamic_tau';
physical_Tmax = params.dt * (params.Tstep - 1);

is_smoke = evalin('base', ...
    'exist(''SMOKE_TEST'',''var'') && logical(SMOKE_TEST)');
if is_smoke
    fprintf('[SMOKE_TEST] dynamic tau, two upgraded cases, two solves each.\n');
    params.h0 = 0.20;
    params.Tstep = 3;
    params.SOLVER_MAXIT = 500;
    results_root = fullfile(thisFileDir, ...
        'benchmark_varvisc_upgraded_dynamic_tau_smoke');
else
    results_root = fullfile(thisFileDir, ...
        'benchmark_varvisc_upgraded_dynamic_tau');
end
params.solvers = varvisc_define_solver_list(params);

case_names = {'current_channel_ar4', 'mixer_circle_four_blade'};
geometry = 'stokes_varvisc_rotor_upgraded';
if ~exist(results_root, 'dir'), mkdir(results_root); end

%% Solve both verified upgraded configurations
all_stats = cell(numel(case_names), 1);
case_descriptors = cell(numel(case_names), 1);
for k = 1:numel(case_names)
    cname = case_names{k};
    [cfg, descriptor] = varvisc_build_upgraded_case( ...
        cname, params.h0, params.dt, physical_Tmax);
    case_descriptors{k} = descriptor;
    fprintf('\n========== Dynamic-tau case %d/%d: %s ==========\n', ...
        k, numel(case_names), cname);
    fprintf(['  geometry=%s, nodes=%d, h0=%.3f, viscosity=%.3g..%.3g, ' ...
        'coupling radius=%.2f\n'], descriptor.geometry, cfg.mesh.N, ...
        params.h0, descriptor.nu_lo, descriptor.nu_hi, ...
        descriptor.coupling_radius);

    run_dir = fullfile(results_root, cname);
    if ~exist(run_dir, 'dir'), mkdir(run_dir); end
    st = solve_stokes_varvisc(cfg, params, run_dir);
    st.case_name = cname;
    st.geometry = geometry;
    st.dt = params.dt;
    all_stats{k} = st;
end

%% Immutable result table, focused summaries, and both plot scales
solver_keys = all_stats{1}.solver_keys;
solver_labels = all_stats{1}.solver_labels;
varvisc_write_all_results_csv(results_root, all_stats, geometry, solver_keys);
figopts_log = varvisc_fig_defaults(struct('yscale', 'log'));
figopts_linear = varvisc_fig_defaults(struct('yscale', 'linear'));
for k = 1:numel(all_stats)
    st = all_stats{k};
    case_dir = fullfile(results_root, st.case_name);
    varvisc_write_case_csvs(case_dir, st);
    varvisc_write_case_figures(case_dir, st, figopts_log);
end

ivt_dir = fullfile(results_root, 'iteration_vs_timestep');
if ~exist(ivt_dir, 'dir'), mkdir(ivt_dir); end
for k = 1:numel(all_stats)
    varvisc_write_iteration_vs_timestep( ...
        ivt_dir, all_stats{k}, figopts_linear, '_linear');
    varvisc_write_iteration_vs_timestep( ...
        ivt_dir, all_stats{k}, figopts_log, '_log');
end
summary_dir = fullfile(results_root, 'summary_plots');
varvisc_write_all_cases_comparison(summary_dir, all_stats, ...
    figopts_linear, 'all_cases_comparison_linear.png');
varvisc_write_all_cases_comparison(summary_dir, all_stats, ...
    figopts_log, 'all_cases_comparison_log.png');
varvisc_write_dynamic_tau_summary(results_root, all_stats, geometry);

params_save = rmfield(params, 'solvers');
cfg_out = struct('params', params_save, 'physical_Tmax', physical_Tmax, ...
    'geometry', geometry, 'case_names', {case_names}, ...
    'case_descriptors', {case_descriptors}, 'solver_keys', {solver_keys}, ...
    'solver_labels', {solver_labels}, 'random_seed', 1, ...
    'deflation_basis', 'exact rank-20, built at step 1, stale physical span', ...
    'tau_rule', 'tau_n = abs(lambda_21(C_n^{-1} K_n C_n^{-T}))^2', ...
    'source_parameter_bundle', ...
        ['TimeMarchingSolverBenchmark/bench_upgrade/stokes_varvisc_rotor/' ...
         'spectral_upgrade/verified']);
save(fullfile(results_root, 'run_config.mat'), 'cfg_out');
fid = fopen(fullfile(results_root, 'run_config.json'), 'w');
if fid < 0
    error('run_upgraded_varvisc_dynamic_tau_benchmark:configWrite', ...
        'Could not create run_config.json in %s.', results_root);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(cfg_out, 'PrettyPrint', true));

fprintf('\n[stokes_varvisc_rotor_upgraded dynamic tau] done. Output in %s\n', ...
    results_root);
