% RUN_VARVISC_SPECTRUM_SPY  Time-resolved spectra of the variable-viscosity
% immersed-rotor KKT system before and after a recycled exact-LDL
% preconditioner.
%
% At time step n, the benchmark solves the symmetric-indefinite system K_n.
% This script factors K_1 once with the same exact, SPD-ified LDL convention as
% the benchmark's exact_ldl_frozen arm,
%
%       C_1 C_1' = |K_1|,
%
% and plots the symmetric split operator seen by MINRES and its update,
%
%       Khat_n = C_1^{-1} K_n C_1^{-T}.
%       E_n    = C_1^{-1} (K_n - K_1) C_1^{-T}.
%
% The factor C_1 is never refreshed.  Consequently Khat_1 has spectrum {+/-1}
% exactly, while later spectra show only the staleness caused by the moving,
% high-contrast viscosity field and the moving immersed coupling.
%
% Production snapshots are steps [1 15 30 45 60] of the stress case
% bar_rotating_nu_orbiting.  For K_n, Khat_n and E_n the script computes the
% smallest- and largest-magnitude spectral tails.  The smallest K_n/Khat_n tails
% use inverse operators.  E_n is structurally singular, so its exact zero
% multiplicity is inserted analytically and any remaining lower-tail slots come
% from paired, adaptively perturbed real shifts of the equivalent generalized
% pencil
%
%       (K_n - K_1) x = lambda (C_1 C_1') x
%
% on both sides of zero.  Exact zeros are plotted in neutral gray at a labeled
% floor because zero cannot appear on a logarithmic axis.
%
% Run from this directory (or with it on the MATLAB path):
%
%       run_varvisc_spectrum_spy
%
% Fast end-to-end validation:
%
%       SMOKE_TEST = true; run_varvisc_spectrum_spy
%
% Outputs -> output/spectrum_spy_varvisc/ (or ..._smoke/):
%   eig_spectrum_timepoints.{pdf,png}        smallest-|lambda| tails, 3 rows
%   eig_spectrum_large_timepoints.{pdf,png} largest-|lambda| tails, 3 rows
%   spy_timepoints.{pdf,png}                 first/last sparsity + signs
%   spectrum_summary.csv                     per-snapshot diagnostics

isSmoke = exist('SMOKE_TEST', 'var') && logical(SMOKE_TEST);
clearvars -except isSmoke;
clc;

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
kernelDir   = fullfile(fileparts(thisFileDir), 'linear_solves', ...
                       'subspace_recycle', 'kernel');
addpath(repoRoot);
addpath(thisFileDir);
addpath(kernelDir);                         % ildl_coordinate_map
rng(1);

% ---- experiment ----------------------------------------------------------
caseName     = 'bar_rotating_nu_orbiting';
dt           = 0.02;
Tmax         = 1.2;
h0           = 0.04;
snapshotStep = [1 15 30 45 60];
kEig         = 500;
eigTol       = 1e-5;
eigMaxit     = 2000;

if isSmoke
    h0           = 0.14;
    snapshotStep = [1 2 3];
    kEig         = 12;
end

if isSmoke
    test_update_small_tail();
end

outName = 'spectrum_spy_varvisc';
if isSmoke, outName = [outName '_smoke']; end
outDir = fullfile(thisFileDir, 'output', outName);
if ~exist(outDir, 'dir'), mkdir(outDir); end

fprintf('[varvisc_spectrum] case=%s  h0=%.3f  k=%d  steps=%s\n', ...
        caseName, h0, kEig, mat2str(snapshotStep));

%% ===================== 1. Shared mesh and case ===========================
base = make_assembly_context(h0, dt, Tmax, caseName);

nSnap = numel(snapshotStep);
spec = repmat(struct('step', [], 'time', [], 'K', [], 'n', [], 'nU', [], ...
                     'nP', [], 'nC', [], 'nnzK', [], 'nuMin', [], ...
                     'nuMax', [], 'symRes', [], 'relK1', [], ...
                     'rawSmall', [], 'rawLarge', [], ...
                     'preSmall', [], 'preLarge', [], ...
                     'updSmall', [], 'updLarge', [], ...
                     'updNullity', [], 'updSymRes', [], 'updIdentityRes', [], ...
                     'seconds', []), nSnap, 1);

% The exact reference factor is built from the first requested step, which is
% required to be production step 1 so it matches exact_ldl_frozen.
assert(snapshotStep(1) == 1, ...
       'The first snapshot must be step 1: it defines the recycled exact LDL.');

Kref = [];
Pref = [];
Cref = [];
Mref = [];
normKref = [];

%% ===================== 2. Snapshot spectra ===============================
for j = 1:nSnap
    step = snapshotStep(j);
    tcur = step * dt;
    fprintf('\n[varvisc_spectrum] snapshot %d/%d: step=%d, t=%.3f\n', ...
            j, nSnap, step, tcur);
    ticSnapshot = tic;

    S = assemble_varvisc_snapshot(base, tcur);
    fprintf('  n=%d (nU=%d, nP=%d, nC=%d), nnz(K)=%d, contrast=%.2f\n', ...
            S.n, S.nU, S.nP, S.nC, nnz(S.K), S.nuMax / S.nuMin);
    fprintf('  symmetry residual %.3e\n', S.symRes);
    assert(S.symRes < 1e-12, ...
           'KKT symmetry residual %.3e exceeds tolerance at step %d.', ...
           S.symRes, step);
    assert(kEig < S.n - 1, 'k=%d must be smaller than n-1=%d.', kEig, S.n-1);

    if j == 1
        Kref    = S.K;
        normKref = norm(Kref, 'fro');
        fprintf('  building the exact LDL split factor of K_1 once ...\n');
        Pref = src.precond.make_ildl_precond(Kref, struct('mode', 'exact'));
        [Cref, cinfo] = ildl_coordinate_map(Pref);
        Mref = Cref * Cref';
        Mref = sparse((Mref + Mref') / 2);
        splitRes = split_factor_residual(Kref, Pref);
        fprintf(['  exact factor: nnz(L)=%d, fill=%.2fx, nnz(C_1C_1'')=%d, ' ...
                 'split |lambda|-1 residual=%.3e\n'], ...
                cinfo.nnzL, cinfo.fill_ratio, nnz(Mref), splitRes);
        assert(splitRes < 1e-8, ...
               'Exact LDL split does not map K_1 to sign(D): residual %.3e.', splitRes);
    else
        assert(S.n == size(Kref, 1), ...
               ['System dimension changed from %d to %d at step %d; a fixed ' ...
                'recycled factor cannot be applied.'], size(Kref,1), S.n, step);
    end

    relK1 = norm(S.K - Kref, 'fro') / max(normKref, eps);

    % One current-system exact solve is used only by the spectral extraction:
    % it supplies K_n^{-1} for the two near-zero calculations.  The plotted
    % PRECONDITIONER remains C_1 from step 1 and is never refreshed.
    Kdec = decomposition((S.K + S.K') / 2, 'ldl');

    fprintf('  raw K_n: smallest/largest %d by |lambda| ...\n', kEig);
    rawLarge = forward_tail(@(x) S.K * x, S.n, kEig, eigTol, eigMaxit);
    rawSmall = inverse_tail(@(x) Kdec \ x, S.n, kEig, eigTol, eigMaxit);

    dK = S.K - Kref;
    dK = sparse((dK + dK') / 2);
    Ahat = @(x) Pref.applyCinv(S.K * Pref.applyCtinv(x));
    Jhat = @(x) Pref.applyCinv(Kref * Pref.applyCtinv(x));
    Ehat = @(x) Pref.applyCinv(dK * Pref.applyCtinv(x));

    if j == 1
        % Khat_1 is exactly sign(D), hence every eigenvalue has magnitude one.
        % Its spectrum has only two distinct values, so a single-vector Krylov
        % process cannot return k repeated Ritz values reliably.  Constructing
        % the exact invariant avoids presenting ARPACK multiplicity artifacts.
        [preSmall, preLarge] = exact_reference_samples(kEig);
    else
        fprintf('  recycled exact LDL: smallest/largest %d by |lambda| ...\n', kEig);
        AhatInv = @(x) Cref' * (Kdec \ (Cref * x));
        preLarge = forward_tail(Ahat,    S.n, kEig, eigTol, eigMaxit);
        preSmall = inverse_tail(AhatInv, S.n, kEig, eigTol, eigMaxit);
    end

    if j == 1
        updSmall   = zeros(kEig, 1);
        updLarge   = zeros(kEig, 1);
        updNullity = S.n;
    else
        fprintf('  scaled update E_n: smallest/largest %d by |lambda| ...\n', kEig);
        updLarge = forward_tail(Ehat, S.n, kEig, eigTol, eigMaxit);
        updRho   = max(abs(updLarge));
        [updSmall, updNullity] = update_small_tail( ...
            dK, Mref, kEig, eigTol, eigMaxit, updRho);
    end
    [updSymRes, updIdentityRes] = update_operator_residuals( ...
        Ahat, Jhat, Ehat, S.n);

    assert_real_finite(rawSmall, 'raw-small', step);
    assert_real_finite(rawLarge, 'raw-large', step);
    assert_real_finite(preSmall, 'preconditioned-small', step);
    assert_real_finite(preLarge, 'preconditioned-large', step);
    assert_real_finite(updSmall, 'update-small', step);
    assert_real_finite(updLarge, 'update-large', step);
    assert(any([rawSmall; rawLarge] < 0) && any([rawSmall; rawLarge] > 0), ...
           'Raw sampled spectrum did not expose both signs at step %d.', step);
    assert(any([preSmall; preLarge] < 0) && any([preSmall; preLarge] > 0), ...
           'Preconditioned sampled spectrum did not expose both signs at step %d.', step);

    fprintf(['  raw:  [% .3e, % .3e], min|lambda|=%.3e\n' ...
             '  prec: [% .3e, % .3e], min|lambda|=%.3e, ' ...
             'max ||lambda|-1|=%.3e\n' ...
             '  E:    [% .3e, % .3e], nullity=%d, ' ...
             'sym=%.3e, Khat-(J+E)=%.3e\n'], ...
            min([rawSmall;rawLarge]), max([rawSmall;rawLarge]), min(abs(rawSmall)), ...
            min([preSmall;preLarge]), max([preSmall;preLarge]), min(abs(preSmall)), ...
            max(abs(abs([preSmall;preLarge]) - 1)), ...
            min([updSmall;updLarge]), max([updSmall;updLarge]), updNullity, ...
            updSymRes, updIdentityRes);

    spec(j).step     = step;
    spec(j).time     = tcur;
    spec(j).K        = S.K;
    spec(j).n        = S.n;
    spec(j).nU       = S.nU;
    spec(j).nP       = S.nP;
    spec(j).nC       = S.nC;
    spec(j).nnzK     = nnz(S.K);
    spec(j).nuMin    = S.nuMin;
    spec(j).nuMax    = S.nuMax;
    spec(j).symRes   = S.symRes;
    spec(j).relK1    = relK1;
    spec(j).rawSmall = rawSmall;
    spec(j).rawLarge = rawLarge;
    spec(j).preSmall = preSmall;
    spec(j).preLarge = preLarge;
    spec(j).updSmall = updSmall;
    spec(j).updLarge = updLarge;
    spec(j).updNullity = updNullity;
    spec(j).updSymRes = updSymRes;
    spec(j).updIdentityRes = updIdentityRes;
    spec(j).seconds  = toc(ticSnapshot);
end

%% ===================== 3. Outputs ========================================
summary = make_summary_table(spec, caseName, h0, kEig);
summaryPath = fullfile(outDir, 'spectrum_summary.csv');
writetable(summary, summaryPath);
fprintf('\n  saved %s\n', summaryPath);

smallPdf = fullfile(outDir, 'eig_spectrum_timepoints.pdf');
plot_timepoint_spectra(spec, 'rawSmall', 'preSmall', 'updSmall', 'smallest-magnitude', ...
                       caseName, h0, kEig, smallPdf);
fprintf('  saved %s and PNG companion\n', smallPdf);

largePdf = fullfile(outDir, 'eig_spectrum_large_timepoints.pdf');
plot_timepoint_spectra(spec, 'rawLarge', 'preLarge', 'updLarge', 'largest-magnitude', ...
                       caseName, h0, kEig, largePdf);
fprintf('  saved %s and PNG companion\n', largePdf);

spyPdf = fullfile(outDir, 'spy_timepoints.pdf');
plot_spy_timepoints(spec([1 end]), spyPdf);
fprintf('  saved %s and PNG companion\n', spyPdf);

fprintf('\n[varvisc_spectrum] done. Output in %s\n', outDir);

%=========================================================================%
% Local functions
%=========================================================================%
function base = make_assembly_context(h0, dt, Tmax, caseName)
%MAKE_ASSEMBLY_CONTEXT  Time-independent pieces shared by all snapshots.
    x1 = 0; x2 = 4; y1 = 0; y2 = 1;

    msh = src.discretization.build_channel_mesh_pde( ...
        h0, x1, x2, y1, y2, {'rect_right'});
    N   = msh.N;
    blk = src.stokes.assemble_stokes_blocks(msh);

    left   = find(msh.rect_left);
    walls  = unique([find(msh.rect_top); find(msh.rect_bottom)]);
    bnodes = unique([left; walls]);
    veldofs = [bnodes; N + bnodes];
    [~, pinNode] = max(msh.p(:, 1));

    geo = struct('x1', x1, 'x2', x2, 'y1', y1, 'y2', y2, ...
                 'xc', (x1+x2)/2, 'yc', (y1+y2)/2, ...
                 'h0', h0, 'Tmax', Tmax);
    cases = varvisc_define_case_list(dt);
    names = cellfun(@(c) c.name, cases, 'UniformOutput', false);
    idx   = find(strcmp(names, caseName), 1);
    assert(~isempty(idx), 'Unknown variable-viscosity case "%s".', caseName);

    base = struct();
    base.msh       = msh;
    base.N         = N;
    base.nU        = 2 * N;
    base.nP        = N;
    base.Bdiv      = blk.B;
    base.Mdt       = blk.M2 / dt;
    base.TR        = triangulation(msh.t, msh.p);
    base.veldofs   = veldofs;
    base.pinNode   = pinNode;
    base.h0        = h0;
    base.mcase     = cases{idx}.factory(geo);
end

function S = assemble_varvisc_snapshot(base, tcur)
%ASSEMBLE_VARVISC_SNAPSHOT  Reproduce solve_stokes_varvisc's K_n assembly.
    msh = base.msh;
    N   = base.N;
    nU  = base.nU;
    nP  = base.nP;

    nu_e = base.mcase.nu_fun(msh.cent(:,1), msh.cent(:,2), tcur);
    K1nu = src.stokes.assemble_visc_stiffness(msh, nu_e);
    ZN   = sparse(N, N);
    Avel = base.Mdt + [K1nu, ZN; ZN, K1nu];

    eps_e  = base.h0^2 ./ (12 * nu_e);
    Lp_eps = src.stokes.assemble_visc_stiffness(msh, eps_e);

    mot = base.mcase.motion_fun(tcur);
    [C, ~, nC] = src.stokes.assemble_coupling(base.TR, N, mot.X, mot.V);
    Z = @(a,b) sparse(a,b);
    K = [ Avel,      base.Bdiv', C'; ...
          base.Bdiv, -Lp_eps,    Z(nP,nC); ...
          C,          Z(nC,nP),  Z(nC,nC) ];

    b = zeros(size(K,1), 1);
    [K, b] = src.stokes.apply_dirichlet_sym( ...
        K, b, base.veldofs, zeros(numel(base.veldofs),1));
    [K, ~] = src.stokes.apply_dirichlet_sym(K, b, nU + base.pinNode, 0);
    K = sparse(K);

    S = struct();
    S.K      = K;
    S.n      = size(K,1);
    S.nU     = nU;
    S.nP     = nP;
    S.nC     = nC;
    S.nuMin  = min(nu_e);
    S.nuMax  = max(nu_e);
    S.symRes = norm(K-K','fro') / max(norm(K,'fro'), eps);
end

function d = forward_tail(Afun, n, k, tol, maxit)
%FORWARD_TAIL  Largest-|lambda| eigenvalues of a symmetric operator.
    [~, D, flag] = eigs(Afun, n, k, 'largestabs', ...
                        'Tolerance', tol, 'MaxIterations', maxit, ...
                        'IsFunctionSymmetric', true);
    if flag ~= 0
        warning('run_varvisc_spectrum_spy:eigsForward', ...
                'eigs forward operator returned flag %d.', flag);
    end
    d = real(diag(D));
end

function d = inverse_tail(Ainv, n, k, tol, maxit)
%INVERSE_TAIL  Smallest-|lambda| eigenvalues via largest-|1/lambda|.
    [~, D, flag] = eigs(Ainv, n, k, 'largestabs', ...
                        'Tolerance', tol, 'MaxIterations', maxit, ...
                        'IsFunctionSymmetric', true);
    if flag ~= 0
        warning('run_varvisc_spectrum_spy:eigsInverse', ...
                'eigs inverse operator returned flag %d.', flag);
    end
    mu = real(diag(D));
    assert(all(abs(mu) > realmin), 'Inverse Ritz value underflowed to zero.');
    d = 1 ./ mu;
end

function [d, nullity] = update_small_tail(dK, Mref, k, tol, maxit, rho)
%UPDATE_SMALL_TAIL  Literal smallest-|lambda| tail of the singular update.
%
% E = C_1^{-1} dK C_1^{-T} and the generalized pencil (dK,Mref), with
% Mref=C_1*C_1', have the same eigenvalues.  Zero rows of dK give an exact
% nullspace whose multiplicity a single-vector Krylov process cannot reproduce,
% so those zeros are counted from the matrix structure and inserted explicitly.
% Any remaining slots come from shifts +delta and -delta.  Their candidate sets
% are merged and duplicate Ritz vectors are detected in the Mref inner product,
% preserving modes on both sides of zero without double-counting modes returned
% by both solves.  Delta is perturbed if a shifted factor is ill-conditioned.
    n = size(dK, 1);
    rowNnz = full(sum(spones(dK), 2));
    inactive = find(rowNnz == 0);
    assert(nnz(dK(:,inactive)) == 0, ...
           'Symmetric update has a zero row whose column is not zero.');
    active = find(rowNnz ~= 0);
    if isempty(active)
        nullity = n;
    else
        activeRank = sprank(dK(active,active));
        assert(activeRank == numel(active), ...
               ['The active update block has structural rank %d < %d; ' ...
                'the exact nullity needs a more general rank analysis.'], ...
               activeRank, numel(active));
        nullity = numel(inactive);
    end

    nzero = min(k, nullity);
    need  = k - nzero;
    d = zeros(nzero, 1);
    if need == 0
        return;
    end

    scale   = max(abs(rho), 1);
    delta   = 1e-5 * scale;
    zeroTol = max(1e-10 * scale, 100 * eps(scale));
    nreq    = min(n-2, max(need + 8, ceil(1.5 * need)));
    candidates = [];

    for attempt = 1:5
        try
            [Vp, Dp, fp] = eigs(dK, Mref, nreq,  delta, ...
                                'Tolerance', tol, 'MaxIterations', maxit);
            [Vm, Dm, fm] = eigs(dK, Mref, nreq, -delta, ...
                                'Tolerance', tol, 'MaxIterations', maxit);
        catch ME
            if attempt == 5
                rethrow(ME);
            end
            delta = sqrt(10) * delta;
            continue;
        end
        if fp ~= 0 || fm ~= 0
            warning('run_varvisc_spectrum_spy:eigsUpdateSmall', ...
                    'generalized update eigs returned flags [%d %d].', fp, fm);
        end

        dp = real(diag(Dp));
        dm = real(diag(Dm));
        kp = abs(dp) > zeroTol;
        km = abs(dm) > zeroTol;
        Vp = Vp(:,kp); dp = dp(kp);
        Vm = Vm(:,km); dm = dm(km);

        % EIGS returns B-orthonormal generalized Ritz vectors.  Keep all plus-
        % shift modes, then append only minus-shift vectors outside their span.
        candidates = dp;
        if isempty(Vp)
            Vkeep = zeros(n,0);
        else
            Vkeep = Vp;
        end
        for q = 1:numel(dm)
            if isempty(Vkeep)
                overlap = 0;
            else
                overlap = norm(Vkeep' * (Mref * Vm(:,q)));
            end
            if overlap < 0.9
                candidates(end+1,1) = dm(q); %#ok<AGROW>
                Vkeep(:,end+1) = Vm(:,q); %#ok<AGROW>
            end
        end
        [~, ord] = sort(abs(candidates), 'ascend');
        candidates = candidates(ord);
        if numel(candidates) >= need
            break;
        end
        if nreq >= n-2
            break;
        end
        nreq = min(n-2, max(nreq + need + 8, ceil(1.7*nreq)));
    end

    assert(numel(candidates) >= need, ...
           ['Only %d nonzero update eigenvalues were recovered, but %d are ' ...
            'needed after inserting %d exact zeros.'], ...
           numel(candidates), need, nzero);
    d = [candidates(1:need); d];
end

function [symRes, identityRes] = update_operator_residuals(Ahat, Jhat, Ehat, n)
%UPDATE_OPERATOR_RESIDUALS  Matrix-free symmetry and Khat_n=J+E checks.
    rs = rng;
    cleanup = onCleanup(@() rng(rs));
    rng(29);
    X = randn(n,3);
    Y = randn(n,3);
    EX = Ehat(X);
    EY = Ehat(Y);
    lhsSym = X' * EY;
    rhsSym = EX' * Y;
    symRes = norm(lhsSym-rhsSym,'fro') / ...
             max([norm(lhsSym,'fro'), norm(rhsSym,'fro'), eps]);
    lhs = Ahat(X);
    rhs = Jhat(X) + EX;
    identityRes = norm(lhs-rhs,'fro') / max(norm(lhs,'fro'),eps);
end

function test_update_small_tail()
%TEST_UPDATE_SMALL_TAIL  Smoke-only fixture with known zeros and lower tail.
    n = 60;
    ev = [zeros(3,1); 0.05; -0.2; linspace(1,10,n-5)'];
    mdiag = linspace(1,2,n)';
    M = spdiags(mdiag,0,n,n);
    A = spdiags(mdiag .* ev,0,n,n); % generalized eigenvalues are exactly ev
    [d, nullity] = update_small_tail(A, M, 5, 1e-11, 1000, max(abs(ev)));
    got = sort(abs(d));
    expected = [0; 0; 0; 0.05; 0.2];
    rel = norm(got-expected) / norm(expected);
    assert(nullity == 3 && rel < 1e-8, ...
           'Synthetic update-tail check failed: nullity=%d, relerr=%.3e.', ...
           nullity, rel);
    fprintf('[SMOKE_TEST] generalized update lower-tail fixture passed.\n');
end

function [dsmall, dlarge] = exact_reference_samples(k)
%EXACT_REFERENCE_SAMPLES  Representative samples of the exact {+1,-1} set.
    nneg = floor(k/2);
    vals = [-ones(nneg,1); ones(k-nneg,1)];
    dsmall = vals;
    dlarge = vals;
end

function r = split_factor_residual(K, P)
%SPLIT_FACTOR_RESIDUAL  Randomized check that |Khat_1| acts as identity.
% Khat_1 has eigenvalues +/-1, equivalently Khat_1^2 = I.
    rs = rng;
    cleanup = onCleanup(@() rng(rs));
    rng(17);
    X = randn(size(K,1), 3);
    Ahat = @(Y) P.applyCinv(K * P.applyCtinv(Y));
    r = norm(Ahat(Ahat(X)) - X, 'fro') / norm(X, 'fro');
end

function assert_real_finite(d, label, step)
    assert(all(isfinite(d)), '%s spectrum contains nonfinite values at step %d.', ...
           label, step);
    assert(isreal(d), '%s spectrum is complex at step %d.', label, step);
end

function T = make_summary_table(spec, caseName, h0, kEig)
%MAKE_SUMMARY_TABLE  One diagnostic row per selected time point.
    nrow = numel(spec);
    rows = repmat(struct(), nrow, 1);
    for j = 1:nrow
        sr = [spec(j).rawSmall; spec(j).rawLarge];
        sp = [spec(j).preSmall; spec(j).preLarge];
        se = [spec(j).updSmall; spec(j).updLarge];
        enz = abs(spec(j).updSmall(spec(j).updSmall ~= 0));
        rows(j).case_name = caseName;
        rows(j).step = spec(j).step;
        rows(j).time = spec(j).time;
        rows(j).h0 = h0;
        rows(j).k_per_tail = kEig;
        rows(j).n = spec(j).n;
        rows(j).nU = spec(j).nU;
        rows(j).nP = spec(j).nP;
        rows(j).nC = spec(j).nC;
        rows(j).nnzK = spec(j).nnzK;
        rows(j).nu_min = spec(j).nuMin;
        rows(j).nu_max = spec(j).nuMax;
        rows(j).nu_contrast = spec(j).nuMax / spec(j).nuMin;
        rows(j).symmetry_residual = spec(j).symRes;
        rows(j).relative_K_change_from_step1 = spec(j).relK1;
        rows(j).raw_lambda_min = min(sr);
        rows(j).raw_lambda_max = max(sr);
        rows(j).raw_min_abs_lambda = min(abs(spec(j).rawSmall));
        rows(j).raw_sampled_negative = sum(sr < 0);
        rows(j).raw_sampled_positive = sum(sr > 0);
        rows(j).pre_lambda_min = min(sp);
        rows(j).pre_lambda_max = max(sp);
        rows(j).pre_min_abs_lambda = min(abs(spec(j).preSmall));
        rows(j).pre_max_abs_lambda = max(abs(spec(j).preLarge));
        rows(j).pre_max_abs_deviation_from_one = max(abs(abs(sp)-1));
        rows(j).pre_sampled_negative = sum(sp < 0);
        rows(j).pre_sampled_positive = sum(sp > 0);
        rows(j).update_lambda_min = min(se);
        rows(j).update_lambda_max = max(se);
        rows(j).update_min_abs_lambda = min(abs(spec(j).updSmall));
        if isempty(enz)
            rows(j).update_min_nonzero_abs_lambda = NaN;
        else
            rows(j).update_min_nonzero_abs_lambda = min(enz);
        end
        rows(j).update_sampled_spectral_radius = max(abs(spec(j).updLarge));
        rows(j).update_structural_nullity = spec(j).updNullity;
        rows(j).update_small_sampled_zero = sum(spec(j).updSmall == 0);
        rows(j).update_sampled_negative = sum(se < 0);
        rows(j).update_sampled_positive = sum(se > 0);
        rows(j).update_symmetry_residual = spec(j).updSymRes;
        rows(j).update_split_identity_residual = spec(j).updIdentityRes;
        rows(j).elapsed_seconds = spec(j).seconds;
    end
    T = struct2table(rows);
end

function plot_timepoint_spectra(spec, rawField, preField, updField, tailName, ...
                                caseName, h0, kEig, pdfPath)
%PLOT_TIMEPOINT_SPECTRA  Raw, recycled-exact-LDL and update rows through time.
    POS = [0.85 0.40 0.32];
    NEG = [0.20 0.45 0.70];
    ZER = [0.45 0.45 0.45];
    ncol = numel(spec);

    rawAll = []; preAll = []; updAll = [];
    for j = 1:ncol
        rawAll = [rawAll; abs(spec(j).(rawField)(:))]; %#ok<AGROW>
        preAll = [preAll; abs(spec(j).(preField)(:))]; %#ok<AGROW>
        updAll = [updAll; abs(spec(j).(updField)(:))]; %#ok<AGROW>
    end
    rawLim = log_limits(rawAll);
    preLim = log_limits(preAll);
    [updLim, zeroFloor] = update_log_limits(updAll);

    width = max(12, 2.45*ncol);
    f = figure('Visible','off','Color','w','Units','inches', ...
               'Position',[0.5 0.5 width 8.4]);
    tl = tiledlayout(f, 3, ncol, 'Padding','compact', 'TileSpacing','compact');

    for j = 1:ncol
        ax = nexttile(tl, j);
        sign_spectrum(ax, spec(j).(rawField), POS, NEG, ZER, []);
        ylim(ax, rawLim);
        title(ax, sprintf('step %d, t = %.2f', spec(j).step, spec(j).time));
        if j == 1
            ylabel(ax, {'original K_n', '|\lambda|'});
            add_sign_legend(ax, POS, NEG, ZER);
        else
            set(ax, 'YTickLabel', []);
        end
        set(ax, 'XTickLabel', []);

        ax = nexttile(tl, ncol+j);
        sign_spectrum(ax, spec(j).(preField), POS, NEG, ZER, []);
        ylim(ax, preLim);
        if j == 1
            ylabel(ax, {'C_1^{-1}K_nC_1^{-T}', '|\lambda|'});
        else
            set(ax, 'YTickLabel', []);
        end
        set(ax, 'XTickLabel', []);

        ax = nexttile(tl, 2*ncol+j);
        eu = spec(j).(updField);
        sign_spectrum(ax, eu, POS, NEG, ZER, zeroFloor);
        ylim(ax, updLim);
        xlabel(ax, sprintf('rank (1:%d)', kEig));
        if j == 1
            ylabel(ax, {'C_1^{-1}(K_n-K_1)C_1^{-T}', '|\lambda|'});
        else
            set(ax, 'YTickLabel', []);
        end
        if spec(j).updNullity == spec(j).n
            note = sprintf('E_1 = 0; %d zeros at floor', sum(eu == 0));
        elseif any(eu == 0)
            note = sprintf('nullity %d; %d zeros at floor', ...
                           spec(j).updNullity, sum(eu == 0));
        else
            note = sprintf('structural nullity %d', spec(j).updNullity);
        end
        text(ax, 0.03, 0.07, note, 'Units','normalized', ...
             'FontSize',8, 'Color',ZER, 'Interpreter','tex');
    end

    title(tl, sprintf(['%s spectrum: original, recycled exact LDL, and scaled update' ...
                       '\n%s, h_0 = %.3f; warm = positive, cool = negative, gray = exact zero'], ...
                       tailName, strrep(caseName,'_','\_'), h0), ...
          'FontWeight','bold');
    exportgraphics(f, pdfPath, 'ContentType','vector');
    exportgraphics(f, strrep(pdfPath,'.pdf','.png'), 'Resolution',180);
    close(f);
end

function sign_spectrum(ax, lam, posColor, negColor, zeroColor, zeroFloor)
%SIGN_SPECTRUM  Rank by magnitude; encode sign by color and zeros at a floor.
    [~, ord] = sort(abs(lam(:)), 'descend');
    lam = lam(ord);
    av  = abs(lam);
    idx = (1:numel(lam))';
    pos = lam > 0;
    neg = lam < 0;
    zer = lam == 0;
    if any(zer) && isempty(zeroFloor)
        nz = av(av > 0);
        if isempty(nz), zeroFloor = 1e-16; else, zeroFloor = min(nz)/10; end
    end

    hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    yp = av; yp(~pos) = NaN;
    yn = av; yn(~neg) = NaN;
    if any(pos)
        semilogy(ax, idx, yp, '-', 'Color',posColor, 'LineWidth',1.2, ...
                 'Marker','.', 'MarkerSize',6);
    end
    if any(neg)
        semilogy(ax, idx, yn, '-', 'Color',negColor, 'LineWidth',1.2, ...
                 'Marker','.', 'MarkerSize',6);
    end
    if any(zer)
        semilogy(ax, idx(zer), repmat(zeroFloor,sum(zer),1), 'o', ...
                 'LineStyle','none', 'Color',zeroColor, ...
                 'MarkerFaceColor',zeroColor, 'MarkerSize',3);
    end
    set(ax,'YScale','log','FontSize',9);
    xlim(ax,[1 max(numel(lam),2)]);
end

function add_sign_legend(ax, POS, NEG, ZER)
    hp = plot(ax, NaN, NaN, '-', 'Color',POS, 'LineWidth',1.5);
    hn = plot(ax, NaN, NaN, '-', 'Color',NEG, 'LineWidth',1.5);
    hz = plot(ax, NaN, NaN, 'o', 'Color',ZER, 'MarkerFaceColor',ZER, ...
              'MarkerSize',3);
    legend(ax,[hp hn hz], {'\lambda > 0','\lambda < 0','\lambda = 0'}, ...
           'Location','best','FontSize',8,'Box','on');
end

function lim = log_limits(vals)
    vals = vals(isfinite(vals) & vals > 0);
    assert(~isempty(vals), 'No positive finite magnitudes available for plotting.');
    lo = min(vals); hi = max(vals);
    if hi / lo < 1.01
        lim = [lo/1.5, hi*1.5];
    else
        lim = [lo/1.25, hi*1.25];
    end
end

function [lim, zeroFloor] = update_log_limits(vals)
%UPDATE_LOG_LIMITS  Shared update-row limits with a visible exact-zero floor.
    vals = vals(isfinite(vals) & vals > 0);
    if isempty(vals)
        zeroFloor = 1e-16;
        lim = [zeroFloor/1.5, zeroFloor*20];
        return;
    end
    lo = min(vals); hi = max(vals);
    zeroFloor = 10^(floor(log10(lo))-1);
    lim = [zeroFloor/1.5, max(hi*1.25,zeroFloor*20)];
end

function plot_spy_timepoints(S, pdfPath)
%PLOT_SPY_TIMEPOINTS  Sparsity and sign patterns at the first/last snapshot.
    POS = [0.85 0.40 0.32];
    NEG = [0.20 0.45 0.70];
    f = figure('Visible','off','Color','w','Units','inches', ...
               'Position',[0.5 0.5 11.5 8.2]);
    tl = tiledlayout(f, numel(S), 2, 'Padding','compact','TileSpacing','compact');

    for j = 1:numel(S)
        K  = S(j).K;
        b1 = S(j).nU + 0.5;
        b2 = S(j).nU + S(j).nP + 0.5;

        ax = nexttile(tl, 2*j-1);
        spy(K); hold(ax,'on');
        block_lines(ax,b1,b2,[0.80 0.20 0.20]);
        title(ax,sprintf('step %d: sparsity (nnz = %d)',S(j).step,nnz(K)));
        xlabel(ax,sprintf('n_U=%d, n_P=%d, n_C=%d',S(j).nU,S(j).nP,S(j).nC));

        ax = nexttile(tl, 2*j);
        [ip,jp] = find(K > 0);
        [in,jn] = find(K < 0);
        plot(ax,jp,ip,'.','Color',POS,'MarkerSize',2); hold(ax,'on');
        plot(ax,jn,in,'.','Color',NEG,'MarkerSize',2);
        block_lines(ax,b1,b2,[0.55 0.55 0.55]);
        set(ax,'YDir','reverse'); axis(ax,'equal'); axis(ax,'tight'); box(ax,'on');
        xlim(ax,[0 size(K,1)]); ylim(ax,[0 size(K,1)]);
        title(ax,sprintf('step %d: sign pattern',S(j).step));
    end
    title(tl,'Variable-viscosity KKT matrix: first and last snapshots', ...
          'FontWeight','bold');
    exportgraphics(f,pdfPath,'ContentType','image','Resolution',200);
    exportgraphics(f,strrep(pdfPath,'.pdf','.png'),'Resolution',160);
    close(f);
end

function block_lines(ax, b1, b2, color)
    xline(ax,b1,'-','Color',color); xline(ax,b2,'-','Color',color);
    yline(ax,b1,'-','Color',color); yline(ax,b2,'-','Color',color);
end
