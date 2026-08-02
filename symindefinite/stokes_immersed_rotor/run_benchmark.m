% RUN_BENCHMARK  Stokes-immersed-rotor benchmark (simplified deal.II step-70).
%
% Backward-Euler unsteady Stokes in a 2D channel with a moving immersed rigid
% solid enforced by distributed Lagrange multipliers.  Each time step solves a
% SYMMETRIC INDEFINITE saddle-point (KKT) system whose coupling block C(t_n)
% changes because the solid moves.  Per step the system is solved by backslash
% (ground truth) and by MINRES for every solver in the registry
% (define_solver_list): the unpreconditioned solve, the SPD block-diagonal
% "block Jacobi" preconditioner, incomplete-LDL, the two-level deflation family
% (L^-T P L^-1, with exact / gaussian / sjlt / polynomial coarse spaces), and the
% Krylov-recycling variant two_level_krylov.
% Method knobs and per-preconditioner refresh cadences are set in the params
% block below.  Add a preconditioner by appending one struct to
% define_solver_list.m — CSV columns, plots and the summary table pick it up
% automatically.
%
% Krylov recycling (two_level_krylov).  Consecutive KKT systems differ only through
% the moving coupling block C(t_n), so the directions MINRES converged slowly on at
% step n-1 are the ones it will converge slowly on at step n.  MINRES runs on the
% SPLIT operator, so the vector it hands to its preconditioner each iteration already
% IS the ILDL-preconditioned residual: the last DEFLAT_RECYCLE_K of them are captured
% as a free side effect of the ordinary minres call (no extra matvec, iteration count
% unchanged), then appended to the SAME gaussian coarse space two_level_gaussian
% uses.  Refreshed every step; at step 1 nothing is recycled yet, so the two solvers
% are identical there and differ only by the recycled columns afterwards.
%
% Output (mirrors report/naca0012/benchmark_final), written to benchmark_final/:
%   all_results.csv, speedup_summary.csv, paper_summary_table.csv,
%   run_config.{mat,json}, summary_plots/, iteration_vs_timestep/, and a
%   per-case subdir with per-solver iteration CSV+PNG, all_solvers_comparison,
%   relative_step_to_step_change and accuracy plots.
%
% NOTE: this benchmark intentionally departs from the suite's SPD contract.
% The PCG/ICHOL/AMG/deflation zoo (solve_deflate_M_P, RAND_EIGS) does NOT apply
% to an indefinite system, so the iteration columns are MINRES solvers
% (minres_unprec / block_jacobi / ...) rather than ichol/amg/chol.

%% ===================== 1. Setup / params ==================================
thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
addpath(repoRoot);
addpath(thisFileDir);
import src.discretization.*
import src.stokes.*
rng(1);

params.dt          = 0.02;
params.Tstep       = 61;        % Tmax = 1.2
params.SOLVER_TOL  = 1e-8;
params.SOLVER_MAXIT = 4000;
params.h0          = 0.05;

% Per-preconditioner refresh cadences (rebuild every N steps; Inf = build once).
% Mirrors report/solve_deflate_M_P's *_PREC_REFRESH knobs — one per component.
params.BLOCKJAC_PREC_REFRESH = Inf;   % block-Jacobi ichol factor
params.ILDL_PREC_REFRESH     = 1;     % incomplete-LDL factor C
params.DEFLAT_PREC_REFRESH   = Inf;   % deflation subspace V
params.DINVERSE_PREC_REFRESH = Inf;   % exact A^{-1} factor (sketched V methods)

% Two-level / deflation method parameters (shared by all two-level V methods;
% consumed by define_solver_list -> build_deflation_V).  Defaults mirror
% report/ball_surface/run_benchmark.m.
params.DEFLAT_SM_EIG       = 500;       % # smallest-|lambda| deflation vectors (report sm_eig)
params.DEFLAT_LG_EIG       = 0;         % # largest-|lambda| deflation vectors (report lg_eig; 0=off)
params.DEFLAT_Q            = 2;         % sketch power-iteration rounds (gaussian/sjlt V)
params.DEFLAT_TAU          = 0.5;       % deflation coarse-correction weight tau
params.DEFLAT_CHEB_DEGREE  = 4;         % Chebyshev degree (polynomial V; exact eigs band)
params.DEFLAT_RECYCLE_K    = 200;        % # ILDL-preconditioned residuals recycled from the
                                        % previous step into two_level_krylov's coarse space
params.ILDL_MODE           = 'nofill';  % incomplete-LDL pattern: 'nofill' | 'droptol'
params.ILDL_DROPTOL        = 1e-3;      % drop tolerance when ILDL_MODE = 'droptol'

params.solvers     = define_solver_list(params);   % MINRES solver/preconditioner registry

geometry = 'stokes_immersed_rotor';

% Channel geometry
x1 = 0; x2 = 4; y1 = 0; y2 = 1;
Lyc = y2 - y1;
Uin = 1.0;                      % peak inflow velocity

%% ===================== 2. Mesh ============================================
fprintf('[stokes_immersed_rotor] building channel mesh (h0=%.3f) ...\n', params.h0);
msh = build_channel_mesh_pde(params.h0, x1, x2, y1, y2, {'rect_right'});
N = msh.N;
fprintf('  nodes: %d  velocity DOFs: %d  pressure DOFs: %d\n', N, 2*N, N);

%% ===================== 3. BCs + case list + SMOKE_TEST ====================
% Velocity Dirichlet: parabolic inflow on the left, no-slip on top/bottom,
% natural (do-nothing) outflow on the right.
left = find(msh.rect_left);
walls = unique([find(msh.rect_top); find(msh.rect_bottom)]);
bnodes = unique([left; walls]);
yv = msh.p(bnodes, 2);
uxv = zeros(numel(bnodes), 1);
isleft = ismember(bnodes, left);
uxv(isleft) = Uin * 4 .* yv(isleft) .* (Lyc - yv(isleft)) / Lyc^2;  % parabola, 0 at walls
veldofs = [bnodes; N + bnodes];
velvals = [uxv; zeros(numel(bnodes), 1)];
velbc_fun = @(t) struct('dofs', veldofs, 'vals', velvals);   % steady inflow

[~, pin_node] = max(msh.p(:, 1));   % pin pressure at the outflow corner

geo = struct('x1', x1, 'x2', x2, 'y1', y1, 'y2', y2, ...
             'xc', (x1+x2)/2, 'yc', (y1+y2)/2, ...
             'h0', params.h0, 'Tmax', params.dt * params.Tstep);

all_cases = define_motion_list(params.dt);
all_names = cellfun(@(c) c.name, all_cases, 'UniformOutput', false);
case_names = {'bar_rotating', 'disk_translating', 'disk_static'};

if evalin('base', 'exist(''SMOKE_TEST'',''var'') && logical(SMOKE_TEST)')
    fprintf('[SMOKE_TEST] Overriding params for fast end-to-end check.\n');
    params.Tstep = 3;
    case_names = case_names(1);   % single (stress) case
end

results_root = fullfile(thisFileDir, 'benchmark_final_no_recycle');
if ~exist(results_root, 'dir'), mkdir(results_root); end

%% ===================== 4. Loop over cases =================================
num_cases = numel(case_names);
all_stats = cell(num_cases, 1);
for k = 1:num_cases
    cname = case_names{k};
    idx   = find(strcmp(all_names, cname), 1);
    mcase = all_cases{idx}.factory(geo);

    cfg = struct();
    cfg.mesh       = msh;
    cfg.nu         = mcase.nu;
    cfg.h0         = params.h0;
    cfg.velbc_fun  = velbc_fun;
    cfg.motion_fun = mcase.motion_fun;
    cfg.pin_node   = pin_node;
    cfg.pin_val    = 0;
    cfg.case_name  = cname;
    cfg.geometry   = geometry;

    run_dir = fullfile(results_root, cname);
    if ~exist(run_dir, 'dir'), mkdir(run_dir); end

    fprintf('\n========== Case %d/%d: %s ==========\n', k, num_cases, cname);
    st = solve_stokes_immersed(cfg, params, run_dir);
    st.mean_nnz_per_row = nnz(assemble_stokes_blocks(msh).A2) / (2*N);
    st.case_name = cname;
    all_stats{k} = st;

    % --- coefficient movie for the stress case ---
    if mcase.is_stress
        write_coefficient_movie(msh, mcase.motion_fun, params, run_dir);
    end
end

solver_keys   = all_stats{1}.solver_keys;
solver_labels = all_stats{1}.solver_labels;

%% ===================== 5. CSVs + plots ====================================
% Root master table (one row per case-timestep, columns track the registry).
write_all_results_csv(results_root, all_stats, geometry, solver_keys);

% Per-case outputs (per-solver CSV+PNG, comparison, coupling change, accuracy).
for k = 1:num_cases
    st = all_stats{k};
    write_case_outputs(fullfile(results_root, st.case_name), st, params.dt);
end

% Root collections.
ivt_dir = fullfile(results_root, 'iteration_vs_timestep');
if ~exist(ivt_dir, 'dir'), mkdir(ivt_dir); end
for k = 1:num_cases
    write_iteration_vs_timestep(ivt_dir, all_stats{k});
end
write_all_cases_comparison(fullfile(results_root, 'summary_plots'), all_stats);
write_speedup_summary(results_root, all_stats, geometry);

% --- run config (strip non-serializable solver handles; keep keys/labels) ---
params_save = rmfield(params, 'solvers');
cfg_out.params        = params_save;
cfg_out.geometry      = geometry;
cfg_out.case_names    = case_names;
cfg_out.solver_keys   = solver_keys;
cfg_out.solver_labels = solver_labels;
save(fullfile(results_root, 'run_config.mat'), 'cfg_out');
jstr = jsonencode(cfg_out);
fid = fopen(fullfile(results_root, 'run_config.json'), 'w');
if fid > 0, fwrite(fid, jstr); fclose(fid); end

% --- per-(geometry,case) paper summary table ---
make_paper_summary_table(results_root);

fprintf('\n[stokes_immersed_rotor] done. Output in %s\n', results_root);

%==========================================================================
%  Local functions
%==========================================================================
function write_all_results_csv(results_root, all_stats, geometry, keys)
%WRITE_ALL_RESULTS_CSV  Master per-(case,timestep) table; solver columns follow
% the registry order (<key>_its / <key>_flag for each solver).
    nsolv  = numel(keys);
    case_col = {}; ts_col = []; relres = []; diffF = [];
    bs = []; constr = []; nCc = [];
    its = repmat({[]}, nsolv, 1); fl = repmat({[]}, nsolv, 1);
    for k = 1:numel(all_stats)
        st = all_stats{k};
        ns = numel(st.solver_its.(keys{1}));
        case_col = [case_col; repmat({st.case_name}, ns, 1)];   %#ok<AGROW>
        ts_col   = [ts_col;   (1:ns)'];                          %#ok<AGROW>
        for s = 1:nsolv
            its{s} = [its{s}; st.solver_its.(keys{s})(:)];
            fl{s}  = [fl{s};  st.solver_flag.(keys{s})(:)];
        end
        relres = [relres; st.solver_relres.(keys{end})(:)];      %#ok<AGROW>
        diffF  = [diffF;  st.coupling_change(:)];                %#ok<AGROW>
        bs     = [bs;     st.backslash_relres(:)];               %#ok<AGROW>
        constr = [constr; st.constraint_res(:)];                 %#ok<AGROW>
        nCc    = [nCc;    st.nC(:)];                             %#ok<AGROW>
    end
    geom_col = repmat({geometry}, numel(ts_col), 1);
    T = table(case_col, geom_col, ts_col, ...
        'VariableNames', {'case_name', 'geometry', 'timestep'});
    for s = 1:nsolv
        T.([keys{s} '_its'])  = its{s};
        T.([keys{s} '_flag']) = fl{s};
    end
    T.relres = relres; T.diffF = diffF; T.backslash_relres = bs;
    T.constraint_res = constr; T.nC = nCc;
    writetable(T, fullfile(results_root, 'all_results.csv'));
    fprintf('Wrote %s\n', fullfile(results_root, 'all_results.csv'));
end

function write_case_outputs(run_dir, st, dt)
%WRITE_CASE_OUTPUTS  Per-solver CSV+PNG, solver comparison, coupling change,
% and accuracy plots for one motion case.
    if ~exist(run_dir, 'dir'), mkdir(run_dir); end
    keys = st.solver_keys; labels = st.solver_labels;
    ns   = numel(st.solver_its.(keys{1}));
    tax  = (1:ns)' * dt;
    safe = @(v) max(v(:), 1);

    % per-solver iteration CSV + PNG
    for s = 1:numel(keys)
        itv = st.solver_its.(keys{s})(:);
        Tk = table((1:ns)', itv, 'VariableNames', {'timestep', 'iterations'});
        writetable(Tk, fullfile(run_dir, [keys{s} '_solver_iterations.csv']));
        fh = figure('Visible', 'off', 'Position', [100 100 700 380]);
        semilogy(tax, safe(itv), '-o', 'MarkerSize', 3, 'LineWidth', 1.2);
        grid on; xlabel('t'); ylabel('MINRES iterations');
        title(sprintf('%s: %s', st.case_name, labels{s}), 'Interpreter', 'none');
        saveas(fh, fullfile(run_dir, [keys{s} '_solver_iterations.png'])); close(fh);
    end

    % all-solvers comparison
    fh = figure('Visible', 'off', 'Position', [100 100 760 420]);
    plot_solver_curves(tax, st, 't');
    title(sprintf('%s: solver comparison', st.case_name), 'Interpreter', 'none');
    saveas(fh, fullfile(run_dir, 'all_solvers_comparison.png')); close(fh);

    % per-step coupling change
    fh = figure('Visible', 'off', 'Position', [100 100 700 380]);
    plot(tax, st.coupling_change(:), '.-', 'LineWidth', 1.2); grid on;
    xlabel('t'); ylabel('||\DeltaC||_F / ||C||_F');
    title(sprintf('%s: per-step coupling change', st.case_name), 'Interpreter', 'none');
    saveas(fh, fullfile(run_dir, 'relative_step_to_step_change.png')); close(fh);

    % accuracy: best-preconditioned error vs backslash + constraint residual
    fh = figure('Visible', 'off', 'Position', [100 100 700 380]);
    err = st.solver_err.(keys{end})(:);
    semilogy(tax, max(err, 1e-16), '-', 'LineWidth', 1.2); hold on;
    semilogy(tax, max(st.constraint_res(:), 1e-16), '--', 'LineWidth', 1.2);
    grid on; xlabel('t'); ylabel('relative');
    legend(sprintf('%s vs backslash', labels{end}), 'constraint ||Cu-g||/||g||', ...
           'Location', 'best', 'Interpreter', 'none');
    title(sprintf('%s: accuracy', st.case_name), 'Interpreter', 'none');
    saveas(fh, fullfile(run_dir, 'accuracy.png')); close(fh);
end

function write_iteration_vs_timestep(out_dir, st)
%WRITE_ITERATION_VS_TIMESTEP  All solvers' iterations vs time step for one case.
    fh = figure('Visible', 'off', 'Position', [100 100 760 420]);
    plot_solver_curves((1:numel(st.solver_its.(st.solver_keys{1})))', st, 'time step n');
    title(sprintf('%s: iterations vs time step', st.case_name), 'Interpreter', 'none');
    saveas(fh, fullfile(out_dir, [st.case_name '.png'])); close(fh);
end

function write_all_cases_comparison(out_dir, all_stats)
%WRITE_ALL_CASES_COMPARISON  One subplot per case, all solvers overlaid.
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    nc = numel(all_stats);
    fh = figure('Visible', 'off', 'Position', [100 100 max(420*nc, 480) 400]);
    for k = 1:nc
        st = all_stats{k};
        subplot(1, nc, k);
        plot_solver_curves((1:numel(st.solver_its.(st.solver_keys{1})))', st, 'time step');
        title(st.case_name, 'Interpreter', 'none');
    end
    saveas(fh, fullfile(out_dir, 'all_cases_comparison.png')); close(fh);
end

function plot_solver_curves(xax, st, xlab)
%PLOT_SOLVER_CURVES  Overlay every solver's iteration curve on a semilog axis.
    keys = st.solver_keys; labels = st.solver_labels;
    markers = {'-o', '-s', '-^', '-d', '-v', '-p'};
    safe = @(v) max(v(:), 1);
    hold on;
    for s = 1:numel(keys)
        semilogy(xax, safe(st.solver_its.(keys{s})), ...
            markers{mod(s-1, numel(markers)) + 1}, 'MarkerSize', 3, 'LineWidth', 1.2);
    end
    set(gca, 'YScale', 'log'); grid on;
    xlabel(xlab); ylabel('MINRES iterations');
    legend(labels, 'Location', 'best', 'Interpreter', 'none');
end

function write_speedup_summary(results_root, all_stats, geometry)
%WRITE_SPEEDUP_SUMMARY  Per-(geometry,case) max iteration difference and factor
% of the unpreconditioned baseline vs each other solver.
    keys   = all_stats{1}.solver_keys;
    base   = 'minres_unprec';
    if ~ismember(base, keys), base = keys{1}; end
    others = keys(~strcmp(keys, base));
    geom_col = {}; case_col = {};
    maxdiff = repmat({[]}, numel(others), 1);
    maxfac  = repmat({[]}, numel(others), 1);
    for k = 1:numel(all_stats)
        st = all_stats{k};
        geom_col = [geom_col; {geometry}];      %#ok<AGROW>
        case_col = [case_col; {st.case_name}];  %#ok<AGROW>
        bvals = st.solver_its.(base)(:);
        for j = 1:numel(others)
            ovals = st.solver_its.(others{j})(:);
            maxdiff{j} = [maxdiff{j}; max(bvals - ovals)];
            maxfac{j}  = [maxfac{j};  max(bvals ./ max(ovals, 1))];
        end
    end
    T = table(geom_col, case_col, 'VariableNames', {'geometry', 'case_name'});
    for j = 1:numel(others)
        T.([others{j} '_vs_' base '_maxdiff'])   = maxdiff{j};
        T.([others{j} '_vs_' base '_maxfactor']) = maxfac{j};
    end
    writetable(T, fullfile(results_root, 'speedup_summary.csv'));
    fprintf('Wrote %s\n', fullfile(results_root, 'speedup_summary.csv'));
end

function write_coefficient_movie(msh, motion_fun, params, run_dir)
    mdir = fullfile(run_dir, 'coefficient_movie');
    if ~exist(mdir, 'dir'), mkdir(mdir); end
    nframes = 8;
    Tmax = params.dt * params.Tstep;
    tt = linspace(params.dt, Tmax, nframes);
    for i = 1:nframes
        mot = motion_fun(tt(i));
        fh = figure('Visible', 'off', 'Position', [100 100 800 240]);
        triplot(msh.t, msh.p(:,1), msh.p(:,2), 'Color', [0.85 0.85 0.85]); hold on;
        plot(mot.X(:,1), mot.X(:,2), 'r.', 'MarkerSize', 8);
        axis equal tight; xlabel('x'); ylabel('y');
        title(sprintf('immersed solid, t = %.3f', tt(i)));
        saveas(fh, fullfile(mdir, sprintf('frame_%02d.png', i)));
        close(fh);
    end
end
