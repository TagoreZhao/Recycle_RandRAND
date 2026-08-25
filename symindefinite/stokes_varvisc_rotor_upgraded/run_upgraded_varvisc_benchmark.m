% RUN_UPGRADED_VARVISC_BENCHMARK  Iterations-vs-timestep comparison on the
% promoted channel and circular-mixer geometries.

%% Setup and shared production parameters
thisFileDir = fileparts(mfilename('fullpath'));
parentDir = fullfile(fileparts(thisFileDir), 'stokes_varvisc_rotor');
repoRoot = fileparts(fileparts(thisFileDir));
addpath(repoRoot, parentDir, thisFileDir);
import src.stokes.*
rng(1);

params = varvisc_default_benchmark_params();
physical_Tmax = params.dt * (params.Tstep - 1);
is_smoke = evalin('base', ...
    'exist(''SMOKE_TEST'',''var'') && logical(SMOKE_TEST)');
if is_smoke
    fprintf('[SMOKE_TEST] two promoted cases, coarse mesh, two solves each.\n');
    params.h0 = 0.20;
    params.Tstep = 3;
    params.DEFLAT_SM_EIG = 12;
    params.GMRES_MAXIT = 50;
    results_root = fullfile(thisFileDir, 'benchmark_varvisc_upgraded_smoke');
else
    results_root = fullfile(thisFileDir, 'benchmark_varvisc_upgraded');
end
params.solvers = varvisc_define_solver_list(params);

case_names = {'current_channel_ar4', 'mixer_circle_four_blade'};
geometry = 'stokes_varvisc_rotor_upgraded';
if ~exist(results_root, 'dir'), mkdir(results_root); end

%% Run both promoted configurations with the current solver registry
all_stats = cell(numel(case_names), 1);
case_descriptors = cell(numel(case_names), 1);
for k = 1:numel(case_names)
    cname = case_names{k};
    [cfg, descriptor] = varvisc_build_upgraded_case( ...
        cname, params.h0, params.dt, physical_Tmax);
    case_descriptors{k} = descriptor;
    fprintf('\n========== Promoted case %d/%d: %s ==========\n', ...
        k, numel(case_names), cname);
    fprintf('  geometry=%s, nodes=%d, h0=%.3f, coupling radius=%.2f\n', ...
        descriptor.geometry, cfg.mesh.N, params.h0, descriptor.coupling_radius);

    run_dir = fullfile(results_root, cname);
    if ~exist(run_dir, 'dir'), mkdir(run_dir); end
    st = solve_stokes_varvisc(cfg, params, run_dir);
    st.case_name = cname;
    st.geometry = geometry;
    st.dt = params.dt;
    all_stats{k} = st;
    write_coefficient_movie(cfg.mesh, cfg.nu_fun, cfg.motion_fun, ...
        params.dt, physical_Tmax, run_dir, is_smoke);
end

%% Tables and figures use the existing benchmark's artifact contract
solver_keys = all_stats{1}.solver_keys;
solver_labels = all_stats{1}.solver_labels;
varvisc_write_all_results_csv(results_root, all_stats, geometry, solver_keys);
figopts = varvisc_fig_defaults();
for k = 1:numel(all_stats)
    st = all_stats{k};
    case_dir = fullfile(results_root, st.case_name);
    varvisc_write_case_csvs(case_dir, st);
    varvisc_write_case_figures(case_dir, st, figopts);
end

ivt_dir = fullfile(results_root, 'iteration_vs_timestep');
if ~exist(ivt_dir, 'dir'), mkdir(ivt_dir); end
for k = 1:numel(all_stats)
    varvisc_write_iteration_vs_timestep(ivt_dir, all_stats{k}, figopts);
end
varvisc_write_all_cases_comparison( ...
    fullfile(results_root, 'summary_plots'), all_stats, figopts);
varvisc_write_upgraded_linear_iteration_figures(results_root, all_stats);
write_speedup_summary(results_root, all_stats, geometry);

params_save = rmfield(params, 'solvers');
cfg_out = struct('params', params_save, 'physical_Tmax', physical_Tmax, ...
    'geometry', geometry, 'case_names', {case_names}, ...
    'case_descriptors', {case_descriptors}, 'solver_keys', {solver_keys}, ...
    'solver_labels', {solver_labels}, 'random_seed', 1, ...
    'source_parameter_bundle', ...
        'TimeMarchingSolverBenchmark/bench_upgrade/stokes_varvisc_rotor/spectral_upgrade/verified');
save(fullfile(results_root, 'run_config.mat'), 'cfg_out');
fid = fopen(fullfile(results_root, 'run_config.json'), 'w');
if fid < 0
    error('run_upgraded_varvisc_benchmark:configWrite', ...
        'Could not create run_config.json in %s.', results_root);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(cfg_out, 'PrettyPrint', true));
varvisc_make_paper_summary_table(results_root);

fprintf('\n[stokes_varvisc_rotor_upgraded] done. Output in %s\n', results_root);

%=========================================================================
function write_speedup_summary(results_root, all_stats, geometry)
    keys = all_stats{1}.solver_keys;
    base = 'minres_unprec';
    if ~ismember(base, keys), base = keys{1}; end
    others = keys(~strcmp(keys, base));
    geometry_col = cell(numel(all_stats), 1);
    case_col = geometry_col;
    maxdiff = repmat({zeros(numel(all_stats), 1)}, numel(others), 1);
    maxfactor = repmat({zeros(numel(all_stats), 1)}, numel(others), 1);
    for k = 1:numel(all_stats)
        st = all_stats{k};
        geometry_col{k} = geometry;
        case_col{k} = st.case_name;
        baseline = st.solver_its.(base)(:);
        for j = 1:numel(others)
            candidate = st.solver_its.(others{j})(:);
            maxdiff{j}(k) = max(baseline - candidate);
            maxfactor{j}(k) = max(baseline ./ max(candidate, 1));
        end
    end
    T = table(geometry_col, case_col, ...
        'VariableNames', {'geometry', 'case_name'});
    for j = 1:numel(others)
        T.([others{j} '_vs_' base '_maxdiff']) = maxdiff{j};
        T.([others{j} '_vs_' base '_maxfactor']) = maxfactor{j};
    end
    writetable(T, fullfile(results_root, 'speedup_summary.csv'));
end

function write_coefficient_movie(msh, nu_fun, motion_fun, dt, Tmax, run_dir, smoke)
    movie_dir = fullfile(run_dir, 'coefficient_movie');
    if ~exist(movie_dir, 'dir'), mkdir(movie_dir); end
    if smoke, frame_count = 2; else, frame_count = 8; end
    times = linspace(dt, Tmax, frame_count);
    for i = 1:frame_count
        nu_e = nu_fun(msh.cent(:, 1), msh.cent(:, 2), times(i));
        mot = motion_fun(times(i));
        fh = figure('Visible', 'off', 'Position', [100 100 800 420]);
        patch('Faces', msh.t, 'Vertices', msh.p, ...
            'FaceVertexCData', log10(nu_e), 'FaceColor', 'flat', ...
            'EdgeColor', 'none');
        hold on;
        plot(mot.X(:, 1), mot.X(:, 2), 'r.', 'MarkerSize', 10);
        axis equal tight; xlabel('x'); ylabel('y'); colorbar;
        title(sprintf('log_{10} nu(x,t) and immersed markers, t = %.3f', times(i)));
        saveas(fh, fullfile(movie_dir, sprintf('frame_%02d.png', i)));
        close(fh);
    end
end
