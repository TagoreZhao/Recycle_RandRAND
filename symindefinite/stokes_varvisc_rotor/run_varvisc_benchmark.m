% RUN_VARVISC_BENCHMARK  Variable-viscosity Stokes-immersed-rotor benchmark.
%
% Backward-Euler unsteady Stokes in a 2D channel with (a) a moving immersed
% rigid solid enforced by distributed Lagrange multipliers AND (b) a moving
% high-contrast viscosity field nu(x,t) stirred along with the rotor.  Each
% time step solves a SYMMETRIC INDEFINITE saddle-point (KKT) system.
%
% Unlike the parent stokes_immersed_rotor — where only the coupling border
% C(t) changes (a structured rank <= 2*nC update) — here the nu-scaled
% velocity stiffness A2(nu_e(t)) and the nu-weighted Brezzi-Pitkaranta
% stabilization block change at EVERY nonzero, EVERY step: the per-step
% update is dense in the sparsity pattern and numerically full-rank
% (r90 = O(N), certified by varvisc_convergence_test.m).  Every parent
% preconditioner arm is therefore re-measured under genuine full-rank
% operator drift: the frozen factorizations lose their rank-2nC safety net,
% the GMRES exact-inverse arm loses its 2nC+1 finite-termination bound, and
% the transported step-1 deflation spaces go honestly stale.
%
% Per step the system is solved by backslash (ground truth) and by one Krylov
% solve per registry entry (varvisc_define_solver_list): unpreconditioned
% MINRES, block Jacobi refreshed AND frozen (the source benchmark's headline
% frozen-vs-refreshed contrast), incomplete-LDL, the EXACT LDL factor of
% step 1 frozen for the whole sequence, the two-level deflation family
% (L^-T P L^-1 with sjlt / gaussian / polynomial / exact coarse spaces), one
% GMRES arm on the frozen exact signed inverse, and the low-rank
% K_1^{-1}(K_n-K_1) sketch variant.  There is NO Krylov-subspace-recycling
% arm in this benchmark.  Sketch parameters are UNIFIED: every randomized
% sketch (gaussian/sjlt deflation V and the low-rank D-sketch) draws
% SKETCH_OVERSAMPLE * DEFLAT_SM_EIG columns, runs DEFLAT_Q power rounds and
% keeps all columns (orthonormalize-only, no truncation).
%
% Add a preconditioner by appending one struct to varvisc_define_solver_list.m
% — CSV columns, plots and the summary table pick it up automatically.
%
% Output (written to benchmark_varvisc/):
%   all_results.csv, speedup_summary.csv, paper_summary_table.csv,
%   run_config.{mat,json}, summary_plots/, iteration_vs_timestep/, and a
%   per-case subdir with per-solver iteration CSV+PNG, all_solvers_comparison,
%   relative_step_to_step_change and accuracy plots.
%
% NOTE: this benchmark intentionally departs from the suite's SPD contract.
% The PCG/ICHOL/AMG/deflation zoo (solve_deflate_M_P, RAND_EIGS) does NOT
% apply to an indefinite system.

%% ===================== 1. Setup / params ==================================
thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
addpath(repoRoot);
addpath(thisFileDir);
import src.discretization.*
import src.stokes.*
rng(1);

params.dt           = 0.02;
params.Tstep        = 61;       % 61 time levels -> 60 solves, Tmax = 1.2
params.SOLVER_TOL   = 1e-8;
params.SOLVER_MAXIT = 4000;
params.h0           = 0.05;

% Per-preconditioner refresh cadences (rebuild every N steps; Inf = build once).
params.BLOCKJAC_PREC_REFRESH = 1;     % block-Jacobi ichol (fluid block moves every step;
                                      % the frozen contrast arm has its own Inf)
params.ILDL_PREC_REFRESH     = 1;     % incomplete-LDL factor C
params.DEFLAT_PREC_REFRESH   = Inf;   % deflation subspace V (transported step-1 space).
                                      % If ever set finite, set DINVERSE_PREC_REFRESH to
                                      % match — otherwise the inverse-power sketch mixes
                                      % a stale decomposition(K_1) with the current C.
params.DINVERSE_PREC_REFRESH = Inf;   % exact A^{-1} factor (sketched V methods)
params.EXACT_PREC_REFRESH    = Inf;   % EXACT LDL factor of the step-1 KKT (frozen)
params.LOWRANK_REF_REFRESH   = Inf;   % frozen ldl of the low-rank sketch reference

% Two-level / deflation method parameters (consumed by varvisc_define_solver_list).
% ONE sketch configuration for every randomized method: width
% SKETCH_OVERSAMPLE * DEFLAT_SM_EIG, DEFLAT_Q power rounds, orthonormalize-only
% (no truncation).  'exact'/'polynomial' V are deterministic and keep dimension
% DEFLAT_SM_EIG.
params.DEFLAT_SM_EIG       = 500;       % # smallest-|lambda| deflation vectors (report sm_eig)
params.DEFLAT_LG_EIG       = 0;         % # largest-|lambda| deflation vectors (0 = off)
params.DEFLAT_Q            = 2;         % power-iteration rounds, ALL randomized sketches
params.SKETCH_OVERSAMPLE   = 2;         % GLOBAL sketch-width factor -> 2*500 = 1000 columns
params.DEFLAT_TAU          = 0.5;       % deflation coarse-correction weight tau
params.DEFLAT_CHEB_DEGREE  = 4;         % Chebyshev degree (polynomial V; exact eigs band)
params.ILDL_MODE           = 'nofill';  % incomplete-LDL pattern: 'nofill' | 'droptol'
params.ILDL_DROPTOL        = 1e-3;      % drop tolerance when ILDL_MODE = 'droptol'

params.GMRES_MAXIT         = 300;       % cap for gmres_exact_inv_frozen (full/unrestarted).
                                        % No finite-termination bound exists here (the
                                        % update is full-rank), so hitting the cap on the
                                        % moving-nu cases is a finding, not a bug.

params.solvers     = varvisc_define_solver_list(params);   % solver/preconditioner registry

geometry = 'stokes_varvisc_rotor';

% Channel geometry
x1 = 0; x2 = 4; y1 = 0; y2 = 1;
Lyc = y2 - y1;
Uin = 1.0;                      % peak inflow velocity

%% ===================== 2. Mesh ============================================
fprintf('[stokes_varvisc_rotor] building channel mesh (h0=%.3f) ...\n', params.h0);
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
             'h0', params.h0, 'Tmax', params.dt * (params.Tstep - 1));

all_cases = varvisc_define_case_list(params.dt);
all_names = cellfun(@(c) c.name, all_cases, 'UniformOutput', false);
case_names = {'bar_rotating_nu_orbiting', 'disk_translating_nu_wake', 'disk_static_nu_const'};

if evalin('base', 'exist(''SMOKE_TEST'',''var'') && logical(SMOKE_TEST)')
    fprintf('[SMOKE_TEST] Overriding params for fast end-to-end check.\n');
    params.Tstep = 3;
    case_names = case_names(1);   % single (stress) case
end

results_root = fullfile(thisFileDir, 'benchmark_varvisc');
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
    cfg.nu_fun     = mcase.nu_fun;
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
    st = solve_stokes_varvisc(cfg, params, run_dir);
    st.case_name = cname;
    st.geometry  = geometry;   % carried so the plot writers need no extra args
    st.dt        = params.dt;
    all_stats{k} = st;

    % --- coefficient movie for the stress case: nu field + rotor points ---
    if mcase.is_stress
        write_coefficient_movie(msh, mcase.nu_fun, mcase.motion_fun, params, run_dir);
    end
end

solver_keys   = all_stats{1}.solver_keys;
solver_labels = all_stats{1}.solver_labels;

%% ===================== 5. CSVs + plots ====================================
% Root master table (one row per case-timestep, columns track the registry).
varvisc_write_all_results_csv(results_root, all_stats, geometry, solver_keys);

% Per-case outputs (per-solver CSV+PNG, comparison, matrix change, accuracy).
% The figure writers live in their own files so replot_varvisc_benchmark can
% redraw everything from all_results.csv without re-solving.
figopts = varvisc_fig_defaults();
for k = 1:num_cases
    st = all_stats{k};
    case_dir = fullfile(results_root, st.case_name);
    varvisc_write_case_csvs(case_dir, st);
    varvisc_write_case_figures(case_dir, st, figopts);
end

% Root collections.
ivt_dir = fullfile(results_root, 'iteration_vs_timestep');
if ~exist(ivt_dir, 'dir'), mkdir(ivt_dir); end
for k = 1:num_cases
    varvisc_write_iteration_vs_timestep(ivt_dir, all_stats{k}, figopts);
end
varvisc_write_all_cases_comparison(fullfile(results_root, 'summary_plots'), all_stats, figopts);
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
varvisc_make_paper_summary_table(results_root);

fprintf('\n[stokes_varvisc_rotor] done. Output in %s\n', results_root);

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

function write_coefficient_movie(msh, nu_fun, motion_fun, params, run_dir)
% Frames of log10(nu) at element centroids with the Lagrange points overlaid
% — both moving structures (viscosity blobs + rotor) in one picture.
    mdir = fullfile(run_dir, 'coefficient_movie');
    if ~exist(mdir, 'dir'), mkdir(mdir); end
    nframes = 8;
    Tmax = params.dt * (params.Tstep - 1);
    tt = linspace(params.dt, Tmax, nframes);
    for i = 1:nframes
        nu_e = nu_fun(msh.cent(:,1), msh.cent(:,2), tt(i));
        mot = motion_fun(tt(i));
        fh = figure('Visible','off','Position',[100 100 800 240]);
        patch('Faces', msh.t, 'Vertices', msh.p, ...
              'FaceVertexCData', log10(nu_e), 'FaceColor', 'flat', 'EdgeColor', 'none');
        hold on;
        plot(mot.X(:,1), mot.X(:,2), 'r.', 'MarkerSize', 8);
        axis equal tight; xlabel('x'); ylabel('y'); colorbar;
        title(sprintf('log_{10}\\nu(x,t) + immersed solid, t = %.3f', tt(i)));
        saveas(fh, fullfile(mdir, sprintf('frame_%02d.png', i)));
        close(fh);
    end
end
