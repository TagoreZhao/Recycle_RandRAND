% RUN_SCHUR_RECYCLE  Driver: does the Schur complement of the immersed-rotor
% KKT system behave like an ordinary SPD sequence?
%
% The sibling study symindefinite/stokes_immersed_rotor benchmarks the
% symmetric INDEFINITE KKT system
%
%     K(t_n) = [ Avel , B'     , C(t_n)' ]     Avel = M/dt + nu*A  (SPD, const)
%              [ B    , -eps*L , 0       ]     C(t) = the ONLY moving block
%              [ C(t) , 0      , 0       ]
%
% Eliminating the SPD velocity block leaves S(t_n) = D + G A^{-1} G' with
% G = [B; C(t)], which is SPD.  That turns the problem into a sequence of SPD
% solves.  S is DENSE (it contains Avel^{-1}), so no incomplete factorization
% applies to it; the arms are therefore an unpreconditioned floor, an exact
% dense chol of S_1 frozen at step 1 and RECYCLED for every later step, and two
% deflation arms whose coarse space is built ONCE at step 1 from S itself and
% then recycled verbatim -- one taking the exact smallest eigenvectors of S, one
% Gaussian-sketching S^{-1}, mirroring the V-selection switch of the indefinite
% sibling symindefinite/stokes_immersed_rotor.
%
% There is no split factor here, so the deflation basis lives in the PHYSICAL
% coordinates of S: unlike the sibling, no coordinate transport is needed.  Only
% the operator moves.
%
% What is structurally new here: the (p,p) block of S is EXACTLY constant in
% time and the step-to-step update has rank <= 2*nC (~40) inside a ~1960-
% dimensional operator.  Whether that friendly structure actually pays off is
% the question the run answers.
%
% Usage:
%     run_schur_recycle
%     SMOKE_TEST = 1; run_schur_recycle      % 3 steps, one case
%
% Outputs land in schur_recycle/ (gitignored).

% Read the smoke flag BEFORE clearing: a bare `clear` in a script wipes the
% base workspace, taking SMOKE_TEST with it and silently launching the full run.
is_smoke = evalin('base', ...
    'exist(''SMOKE_TEST'',''var'') && logical(SMOKE_TEST)');
clearvars -except is_smoke
clc;

paths = add_schur_paths();
assert_local_helpers();
import src.discretization.*
rng(1);

%% ===================== 1. Parameters ======================================
params = make_schur_params();

case_names = {'bar_rotating', 'disk_translating', 'disk_static'};

if is_smoke
    fprintf('[SMOKE_TEST] short run: 3 steps, stress case only.\n');
    % NOTE: trim max_steps, NOT Tstep -- Tstep sets Tmax and hence the rotor's
    % angular velocity, so shrinking it would change the geometry under test.
    params.max_steps = 3;
    params.h0        = 0.1;
    params.sm_eig    = 30;              % nS ~ 542 on this coarse mesh
    case_names = case_names(1);
    results_root = fullfile(paths.thisFileDir, 'schur_recycle_smoke');
else
    results_root = paths.outDir;
end
if ~exist(results_root, 'dir'), mkdir(results_root); end

%% ===================== 2. Mesh (built once) ===============================
fprintf('[schur_comp] building channel mesh (h0=%.3f) ...\n', params.h0);
[~, msh] = schur_make_cfg(case_names{1}, params, []);
fprintf('  nodes: %d   velocity DOFs: %d   pressure DOFs: %d\n', ...
        msh.N, 2*msh.N, msh.N);

%% ===================== 3. Case loop =======================================
figopts   = benchmark_fig_defaults();
all_stats = cell(numel(case_names), 1);

for k = 1:numel(case_names)
    cname = case_names{k};
    fprintf('\n========== Case %d/%d: %s ==========\n', k, numel(case_names), cname);

    rng(1);                               % identical streams across cases
    cfg = schur_make_cfg(cname, params, msh);

    run_dir = fullfile(results_root, cname);
    if ~exist(run_dir, 'dir'), mkdir(run_dir); end

    Astat = solve_schur_sequence(cfg, params, run_dir);
    all_stats{k} = Astat;

    write_schur_case_outputs(run_dir, Astat, figopts);

    fprintf('  [%s] median iterations:', cname);
    for i = 1:numel(Astat.solver_keys)
        fprintf('  %s=%g', Astat.solver_keys{i}, ...
                median(Astat.solver_its.(Astat.solver_keys{i})));
    end
    fprintf('\n');
end

%% ===================== 4. Roll-up CSVs ====================================
write_schur_all_results(results_root, all_stats);
write_schur_speedup(results_root, all_stats);

%% ===================== 5. Cross-case comparison figure ====================
write_schur_summary(results_root, all_stats, figopts);

%% ===================== 6. Provenance ======================================
cfg_dump = params;
if isfield(cfg_dump, 'standalone_variants')
    cfg_dump.variant_names = {params.standalone_variants.name};
    cfg_dump = rmfield(cfg_dump, 'standalone_variants');
end
cfg_dump.case_names    = case_names;
cfg_dump.solver_keys   = all_stats{1}.solver_keys;
cfg_dump.solver_labels = all_stats{1}.solver_labels;
% params.tau may be empty ("auto"); record the value actually used.
cfg_dump.tau_used      = all_stats{1}.tau;
cfg_dump.deflat_dim    = all_stats{1}.deflat_dim;
cfg_dump.mesh_N      = msh.N;
cfg_dump.matlab      = version;
cfg_dump.finished    = datestr(now, 'yyyy-mm-ddTHH:MM:SS'); %#ok<TNOW1,DATST>

save(fullfile(results_root, 'run_config.mat'), 'cfg_dump');
fid = fopen(fullfile(results_root, 'run_config.json'), 'w');
fprintf(fid, '%s', jsonencode(cfg_dump, 'PrettyPrint', true));
fclose(fid);

fprintf('\n[schur_comp] done -> %s\n', results_root);
