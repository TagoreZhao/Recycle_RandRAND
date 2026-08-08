% EXTRACT_KKT_EXAMPLES  Save the immersed-rotor KKT operator at TWO time steps.
%
% Writes, for each step in STEPS, one .mat holding the symmetric indefinite KKT
% matrix A = K(t_n), its right-hand side b, the backslash ground truth x_ref and
% a meta struct:
%
%     stokes_kkt_example_h0p03_step01.mat   ->  A, b, x_ref, meta
%     stokes_kkt_example_h0p03_step09.mat   ->  A, b, x_ref, meta
%
% The (A, b, meta) variable names match symindefinite/linear_solves/extract_system.m
% exactly, so any consumer of stokes_kkt_system.mat loads these unchanged.  The
% files are GITIGNORED and regenerable -- rerun this script.
%
% WHY h0 = 0.03 AND NOT THIS FOLDER'S BENCHMARK DEFAULT OF 0.05.  It is matched
% to the Schur twin's default (make_schur_params.m), so these artifacts and the
% ones from stokes_immersed_rotor_schur_comp/schur_extract_examples.m are the
% SAME system in two algebraic forms: S(t_n) is the Schur complement of this
% K(t_n).  meta carries a cross-artifact fingerprint so the two can be checked
% against each other (see CROSS-CHECK below).
%
% NO NEW ASSEMBLY CODE.  Everything is delegated to build_stokes_sequence, which
% reproduces +src/+stokes/solve_stokes_immersed.m block-for-block, advances the
% state with the same backslash solve, and asserts its low-rank identity per
% step.  (That helper lives under linear_solves/subspace_recycle/kernel/, which
% already addpaths THIS folder for define_motion_list -- so the dependency is
% mutual.  Harmless under MATLAB's flat path model, and the price of not adding
% a fifth copy of the KKT assembly to the repo.)
%
% CROSS-CHECK against the Schur twin:
%     k = load('stokes_kkt_example_h0p03_step09.mat', 'meta');
%     s = load('../stokes_immersed_rotor_schur_comp/schur_example_h0p03_step09.mat', 'meta');
%     [k.meta.normK_fro, s.meta.normK_fro]     % must agree to ~1e-12
%
% See also: build_stokes_sequence, seq_K, schur_extract_examples,
%           src.stokes.solve_stokes_immersed, extract_system.

clear; clc;

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
kernelDir   = fullfile(repoRoot, 'symindefinite', 'linear_solves', ...
                       'subspace_recycle', 'kernel');
addpath(repoRoot);
addpath(kernelDir);            % build_stokes_sequence calls add_recycle_paths itself
rng(1);

%==========================================================================
%  Configuration
%==========================================================================
CASE_NAME = 'bar_rotating';    % the stress case: the bar sweeps across the mesh
H0        = 0.03;              % set 0.1 for a seconds-long smoke run
DT        = 0.02;
TSTEP     = 61;                % sets Tmax = DT*TSTEP and hence the bar's omega
COMPUTE_SPECTRUM = false;      % eigs at n ~ 15.7k is slow; off by default

% WHICH TWO STEPS, AND WHY NOT 15/16 OR 30/31.
%   theta(n) = omega*n*dt = 0.20601*n rad = 11.803 deg * n   (2 revolutions
%   over Tmax = 1.22).  The bar's point set s = linspace(-Lb,Lb,nb) is symmetric
%   about 0, so for the involution j -> j' with s_j' = -s_j BOTH the points and
%   the velocities satisfy X_j(th+pi) = X_j'(th), V_j(th+pi) = V_j'(th).  Hence
%   C(th+pi) = P*C(th) exactly, for a permutation P: separation in theta is only
%   meaningful MOD PI, and K(th+pi) is permutation-similar to K(th).
%
%       step |  dtheta vs step 1  |  effective (mod pi)
%          8 |        82.6 deg    |     82.6
%          9 |        94.4 deg    |     85.6   <- maximum separation
%         10 |       106.2 deg    |     73.8
%         16 |       177.1 deg    |      2.9   <- near-permutation of step 1
%         31 |       354.1 deg    |      5.9   <- near-permutation of step 1
%
%   So the intuitive "step 1 vs step 30" is the WORST possible pair here.
%   Step 1 is also atypical in its RHS (u_prev = 0, so b's velocity block is
%   pure Dirichlet lifting).  Use STEPS = [5 13] instead for the same 85.6 deg
%   separation between two developed-flow steps.
STEPS = [1 9];

%==========================================================================
%  Build the sequence (all assembly happens in here)
%==========================================================================
fprintf('[extract_kkt_examples] building %s sequence, h0=%.3g, %d steps...\n', ...
        CASE_NAME, H0, max(STEPS));

Sq = build_stokes_sequence(struct( ...
        'case_name', CASE_NAME, 'h0', H0, 'dt', DT, 'Tstep', TSTEP, ...
        'nsteps',    max(STEPS), ...
        'verify',    true, ...        % asserts K_n == K0 + Cblk*Sel' + Sel*Cblk'
        'use_cache', false, ...       % see the Tmax guard below
        'quiet',     false));

% Tstep is NOT part of build_stokes_sequence's cache key, but it sets Tmax and
% hence omega -- so a stale cache at the same (case, h0, dt, nsteps) would
% silently hand back a DIFFERENT geometry.  Guard on the stored Tmax, which
% survives a cache hit.
assert(abs(Sq.Tmax - DT * TSTEP) < 1e-12, ...
       'Tmax mismatch: got %.6g, expected %.6g (stale cache?)', Sq.Tmax, DT*TSTEP);
assert(abs(Sq.h0 - H0) < 1e-12, 'h0 mismatch: got %.6g, expected %.6g', Sq.h0, H0);
assert(strcmp(Sq.case_name, CASE_NAME), 'case mismatch: got %s', Sq.case_name);
assert(max(STEPS) <= Sq.nsteps, 'sequence has %d steps, need %d', Sq.nsteps, max(STEPS));

nU = Sq.nU;
fprintf('[extract_kkt_examples] N=%d  nU=%d  nP=%d  nC=%d  n=%d\n', ...
        Sq.N, nU, Sq.nP, Sq.nC, Sq.n);

%==========================================================================
%  Extract, verify and save each requested step
%==========================================================================
tag   = ['h' strrep(num2str(H0), '.', 'p')];     % 0.03 -> h0p03, 0.1 -> h0p1
metas = cell(numel(STEPS), 1);

for k = 1:numel(STEPS)
    n     = STEPS(k);
    tcur  = n * DT;
    A     = seq_K(Sq, n);
    b     = Sq.b{n};
    x_ref = Sq.xref{n};
    Cn    = Sq.Ccpl{n};
    gn    = Sq.g{n};

    % --- verification -----------------------------------------------------
    sym_res = norm(A - A', 'fro') / max(norm(A, 'fro'), eps);
    relres  = norm(A * x_ref - b) / max(norm(b), eps);
    con_res = norm(Cn * x_ref(1:nU) - gn) / max(norm(gn), eps);
    assert(sym_res < 1e-12, 'step %d: K is not symmetric (%.3e)', n, sym_res);
    assert(relres  < 1e-10, 'step %d: K*x_ref ~= b (%.3e)', n, relres);

    % --- metadata ---------------------------------------------------------
    meta = struct( ...
        'n', Sq.n, 'nU', nU, 'nP', Sq.nP, 'nC', Sq.nC, 'N', Sq.N, ...
        'h0', Sq.h0, 'dt', Sq.dt, 'Tstep', TSTEP, 'Tmax', Sq.Tmax, ...
        'nu', Sq.nu, 'eps_stab', Sq.eps_stab, 'case_name', Sq.case_name, ...
        'step', n, 't_snap', tcur, ...
        'theta_rad', 2*pi*2*tcur/Sq.Tmax, ...
        'theta_deg', 360*2*tcur/Sq.Tmax, ...
        'pin_node', Sq.pin_node, 'sym_res', sym_res, ...
        'nnz_A', nnz(A), 'backslash_relres', relres, 'constraint_res', con_res, ...
        'coupling_change', Sq.coupling_change(n), 'lowrank_verified', true, ...
        ... % cross-artifact fingerprint -- must match the Schur twin's meta
        'normK_fro', norm(A, 'fro'), 'nnzK', nnz(A), ...
        'norm_b', norm(b), 'normC_fro', norm(Cn, 'fro'), ...
        'created', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
        'matlab_version', version);

    if COMPUTE_SPECTRUM
        % eigs takes NAME-VALUE options; a struct is silently ignored.
        meta.lambda_min = eigs(A, 1, 'smallestreal', 'MaxIterations', 500);  %#ok<UNRCH>
        meta.lambda_max = eigs(A, 1, 'largestreal',  'MaxIterations', 500);
    end

    % --- save -------------------------------------------------------------
    matName = sprintf('stokes_kkt_example_%s_step%02d.mat', tag, n);
    matFile = fullfile(thisFileDir, matName);
    save(matFile, 'A', 'b', 'x_ref', 'meta', '-v7.3');

    d = dir(matFile);
    fprintf(['[extract_kkt_examples] step %2d  t=%.3f  theta=%6.1f deg  ' ...
             'nnz=%d  symres=%.1e  relres=%.1e  conres=%.1e  -> %s (%.1f MB)\n'], ...
            n, tcur, meta.theta_deg, meta.nnz_A, sym_res, relres, con_res, ...
            matName, d.bytes/2^20);

    metas{k} = meta;
    clear A b x_ref;
end

%==========================================================================
%  The two steps must actually differ (the whole point of saving two)
%==========================================================================
if numel(STEPS) == 2
    C1 = Sq.Ccpl{STEPS(1)};
    C2 = Sq.Ccpl{STEPS(2)};
    dC = norm(C2 - C1, 'fro') / max(norm(C1, 'fro'), eps);
    fprintf('[extract_kkt_examples] ||C_%d - C_%d||_F / ||C_%d||_F = %.4f\n', ...
            STEPS(2), STEPS(1), STEPS(1), dC);
    assert(dC > 1e-3, ['steps %d and %d have near-identical coupling ' ...
           '(dC = %.2e) -- see the theta table above, they are probably a ' ...
           'multiple of pi apart'], STEPS(1), STEPS(2), dC);
end

fprintf('[extract_kkt_examples] done. Files are gitignored; rerun to regenerate.\n');
