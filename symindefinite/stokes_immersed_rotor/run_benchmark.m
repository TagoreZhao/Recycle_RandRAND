% RUN_BENCHMARK  Stokes-immersed-rotor benchmark (simplified deal.II step-70).
%
% Backward-Euler unsteady Stokes in a 2D channel with a moving immersed rigid
% solid enforced by distributed Lagrange multipliers.  Each time step solves a
% SYMMETRIC INDEFINITE saddle-point (KKT) system whose coupling block C(t_n)
% changes because the solid moves.  Per step the system is solved by backslash
% (ground truth) and by MINRES for every solver in the registry
% (define_solver_list): the unpreconditioned solve, the SPD block-diagonal
% "block Jacobi" preconditioner, incomplete-LDL, the EXACT LDL factor of step 1
% frozen for the whole sequence (exact_ldl_frozen), the two-level deflation family
% (L^-T P L^-1, with exact / gaussian / sjlt / polynomial coarse spaces), the
% low-rank A^{-1}B sketch variant two_level_lowrank_sketch, the
% Krylov-recycling variant two_level_krylov, and one GMRES arm
% (gmres_exact_inv_frozen) that tests the low-rank finite-termination bound.
% Method knobs and per-preconditioner refresh cadences are set in the params
% block below.  Add a preconditioner by appending one struct to
% define_solver_list.m — CSV columns, plots and the summary table pick it up
% automatically.
%
% Low-rank finite termination (gmres_exact_inv_frozen).  The only non-MINRES arm, and
% it cannot be MINRES: its preconditioner is the exact SIGNED inverse of the step-1
% KKT matrix, which is indefinite.  Since K_n - K_1 is symmetric of rank 2*rank(C_n -
% C_1) <= 2*nC, the left-preconditioned operator K_1^-1 K_n is the IDENTITY plus a
% rank-r update, whose minimal polynomial has degree <= r+1 -- so unrestarted GMRES
% must terminate in at most 2*nC + 1 iterations (41 for bar_rotating, ~89 for the
% disks, and exactly 1 for disk_static where the coupling never moves).  This is a
% theorem, not a tuning knob: lowrank_bound.png plots the arm against 2*nC(n)+1 per
% case and states on the figure whether the claim held.  exact_ldl_frozen is the
% controlled contrast -- the SAME frozen factor, SPD-ified (M = |K_1|) so MINRES can
% use it at all, which costs the clean I + low-rank structure.
%
% Low-rank A^{-1}B sketch (two_level_lowrank_sketch).  The same two-level scheme as the
% rest of the deflation family, differing only in where the coarse space comes from:
% with A_1 = K_1 factored ONCE and frozen and A_2 = K_n the current system, the
% directions the update moves are the range of the nonsymmetric D = A_1^{-1}(A_2 - A_1),
% whose dominant left singular subspace is taken by randomized power iteration,
% V = orth(C_n' (D D')^q D Omega).  Since K_n - K_1 = U B U' with U = [dC, Sel] and B
% invertible, range(D) is EXACTLY K_1^{-1} range(U), of dimension <= 2*nC -- so at
% k >= 2*nC this coarse space contains every direction the operator update can have
% moved, at <= 2*nC effective columns rather than DEFLAT_SM_EIG = 500 of them, and
% with no refactorization after step 1.  Per step it costs (2q+1)*k batched backsolves against
% the frozen factor, (2q+1)*k sparse dK matvecs and one n-by-k pivoted QR.  What is
% recycled is the FACTORIZATION of A_1, not the subspace: V is rebuilt every step
% because dK changes.  gmres_exact_inv_frozen is the floor this arm is reaching for,
% and two_level_gaussian / two_level_exact are the incumbents it is cheaper than.
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
%   relative_step_to_step_change, accuracy and lowrank_bound plots.
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
params.EXACT_PREC_REFRESH    = Inf;   % EXACT LDL factor of the step-1 KKT (frozen)

% Two-level / deflation method parameters (shared by all two-level V methods;
% consumed by define_solver_list -> build_deflation_V).  Defaults mirror
% report/ball_surface/run_benchmark.m.
params.DEFLAT_SM_EIG       = 500;       % # smallest-|lambda| deflation vectors (report sm_eig)
params.DEFLAT_LG_EIG       = 0;         % # largest-|lambda| deflation vectors (report lg_eig; 0=off)
params.DEFLAT_Q            = 2;         % sketch power-iteration rounds (gaussian/sjlt V)
params.DEFLAT_TAU          = 0.5;       % deflation coarse-correction weight tau
params.DEFLAT_CHEB_DEGREE  = 4;         % Chebyshev degree (polynomial V; exact eigs band)
params.DEFLAT_RECYCLE_K    = 0;        % # ILDL-preconditioned residuals recycled from the
                                        % previous step into two_level_krylov's coarse space
params.ILDL_MODE           = 'nofill';  % incomplete-LDL pattern: 'nofill' | 'droptol'
params.ILDL_DROPTOL        = 1e-3;      % drop tolerance when ILDL_MODE = 'droptol'
% Low-rank A^{-1}B sketch (two_level_lowrank_sketch).  The sketch width is
% k = LOWRANK_OVERSAMPLE * LOWRANK_SM_EIG and V keeps all k columns.  k = 250 clears
% 2*nC (96 for the bar, 240 for the disks at this h0) for every case here, which is
% the regime the arm works in: below that rank the sketch keeps the directions the
% UPDATE moved most rather than the ones the OPERATOR is worst conditioned in, and
% measures WORSE than the smoother alone -- on disk_translating step 3, k = 100 took
% 2119 iterations against 1931 for no coarse space and 1184 at k = 250.
% test_lowrank_sketch_V T9 pins the boundary.  If nC changes (define_motion_list),
% RE-CHECK that 2*LOWRANK_SM_EIG still clears 2*nC.
params.LOWRANK_SM_EIG      = 125;       % # small eigendirections of interest
params.LOWRANK_OVERSAMPLE  = 2;         % oversampling FACTOR -> k = 2*125 = 250
params.LOWRANK_SKETCH_Q    = 2;         % power-iteration rounds q on D = K_1^{-1}(K_n-K_1)
params.LOWRANK_REF_REFRESH = Inf;       % frozen ldl of the reference system (Inf = once)

params.GMRES_MAXIT         = 300;       % iteration cap for gmres_exact_inv_frozen.
                                        % Full (unrestarted) GMRES, so this is a cap
                                        % on TOTAL iterations and on the Krylov basis
                                        % it stores.  It must stay above the predicted
                                        % 2*nC+1 (41 for the bar, ~89 for the disks) or
                                        % the arm reports the budget, not the claim.

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

results_root = fullfile(thisFileDir, 'benchmark_no_krylov_recycle_with_new');
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
    st.geometry  = geometry;   % carried so the plot writers need no extra args
    st.dt        = params.dt;
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
% The figure writers live in their own files so replot_benchmark can redraw
% everything from all_results.csv without re-solving.
figopts = benchmark_fig_defaults();
for k = 1:num_cases
    st = all_stats{k};
    case_dir = fullfile(results_root, st.case_name);
    write_case_csvs(case_dir, st);
    write_case_figures(case_dir, st, figopts);
end

% Root collections.
ivt_dir = fullfile(results_root, 'iteration_vs_timestep');
if ~exist(ivt_dir, 'dir'), mkdir(ivt_dir); end
for k = 1:num_cases
    write_iteration_vs_timestep(ivt_dir, all_stats{k}, figopts);
end
write_all_cases_comparison(fullfile(results_root, 'summary_plots'), all_stats, figopts);
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
