% TEST_TWO_LEVEL_RECYCLE  Krylov-subspace recycling for the two-level (ILDL
% smoother + P^{1/2} deflation) MINRES preconditioner on the symmetric-indefinite
% Stokes KKT system.
%
% Same setup as test_two_level_sketched.m (ILDL smoother M = C C', split operator
% Ahat = C^-1 A C^-T, MINRES solves Ahat y = C^-1 b and recovers x = C^-T y).  The
% new ingredient is RECYCLING across a pair of consecutive "time steps":
%
%   step 1   deflate with the Gaussian sketched coarse space Vbase, and RECORD the
%            ILDL-preconditioned residuals for free inside the preconditioner
%            handle (make_recording_pdef).  In the split space the vector MINRES
%            hands to its preconditioner IS the ILDL-preconditioned residual, so
%            the capture needs no transformation and costs no extra matvec.
%   step 2   a slightly perturbed system A2 (the immersed solid has moved),
%            deflated with Vbase alone vs the augmented space
%            augment_recycle_V(Vbase, W) = [Vbase, orth(W - Vbase Vbase' W)].
%
% This is the single-system stand-in for the two_level_krylov solver registered in
% stokes_immersed_rotor/define_solver_list.m, and it pins down the property the
% whole design rests on: the recording tap must NOT change the iteration path.
%
% Run extract_system.m first.
%
% See also: test_two_level_sketched, make_recording_pdef, augment_recycle_V,
%           src.precond.two_level_split_solve, src.precond.deflation_Psqrt_apply.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
repoRoot = fileparts(fileparts(thisFileDir));   % .../Recycle_RandRAND (for +src)
addpath(repoRoot);                              % +src on path for src.precond.*
addpath(fullfile(repoRoot, 'symindefinite', 'stokes_immersed_rotor'));  % local helpers
rng(1);

outDir = fullfile(thisFileDir, 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ---- load system ---------------------------------------------------------
matFile = fullfile(thisFileDir, 'stokes_kkt_system.mat');
assert(exist(matFile, 'file') == 2, ...
       'stokes_kkt_system.mat not found — run extract_system.m first.');
S = load(matFile);
A = S.A;  b = S.b;  meta = S.meta;
n = size(A, 1);
fprintf('[test] loaded A (n=%d, nnz=%d), b (||b||=%.3e)\n', n, nnz(A), norm(b));

tol   = 1e-8;
maxit = min(2000, n);
k     = 100;       % base (sketched) coarse-space size
q     = 2;         % sketch power-iteration count
tau   = 1;         % coarse-correction weight
nrec  = 50;        % # recycled Krylov vectors (matches DEFLAT_RECYCLE_K default)

% ---- ILDL smoother M = C C' (rebuild C explicitly, cf. test_two_level_sketched)
P    = src.precond.make_ildl_precond(A, struct('mode', 'nofill'));
Sinv = spdiags(1 ./ P.s, 0, n, n);
Pt   = sparse(P.p, (1:n)', 1, n, n);          % P^T
C    = Sinv * Pt * P.L * P.Dsqrt;             % M = C C'

% exact inverse of Ahat = C^-1 A C^-T   =>   Ahat^{-1} = C' A^{-1} C
dA      = decomposition(A);
AinvFun = @(Y) C' * (dA \ (C * Y));

Vbase = orthonormal(src.precond.subspace_iter_plain(AinvFun, randn(n, k), q));

%% ===== step 1: deflated solve + free capture of the Krylov residuals ======
rec1 = run_defl(A, b, tol, maxit, P, Vbase, tau, nrec);
ref1 = pack_ref(A, b, tol, maxit, P, Vbase, tau);

fprintf('\n===== step 1 (k=%d, tau=%.3g, recycle=%d) =====\n', k, tau, nrec);
fprintf('  %-34s %6s %8s %12s %12s\n', 'configuration', 'flag', 'iters', 'relres', 'true_res');
print_row('two_level_split_solve (reference)', ref1);
print_row('recording preconditioner',          rec1);
fprintf('  captured Krylov block: %d x %d\n', size(rec1.W, 1), size(rec1.W, 2));

% The tap must be non-intrusive: identical iteration path, identical solution.
assert(rec1.iters == ref1.iters, ...
       'recording tap changed the MINRES iteration count (%d vs %d)', ...
       rec1.iters, ref1.iters);
assert(abs(rec1.relres - ref1.relres) <= 1e-14 * max(ref1.relres, eps), ...
       'recording tap changed the MINRES relative residual');
assert(norm(rec1.x - ref1.x) <= 1e-12 * max(norm(ref1.x), eps), ...
       'recording tap changed the MINRES solution');

% Buffer semantics: at most nrec columns, and full whenever MINRES ran that long.
assert(size(rec1.W, 1) == n && size(rec1.W, 2) <= nrec, ...
       'captured block has the wrong shape');
assert(size(rec1.W, 2) >= min(nrec, rec1.iters), ...
       'captured fewer columns than MINRES iterations allow');
assert(all(isfinite(rec1.W(:))), 'captured block contains non-finite entries');

%% ===== step 2: the solid has moved — recycle into the coarse space ========
[A2, b2] = perturb_coupling(A, b, meta, 0.05);

Vall = augment_recycle_V(Vbase, rec1.W);
assert(norm(Vall' * Vall - eye(size(Vall, 2)), 'fro') < 1e-10, ...
       'augmented coarse space is not orthonormal');
assert(size(Vall, 2) > size(Vbase, 2) && size(Vall, 2) <= size(Vbase, 2) + nrec, ...
       'augmented coarse space has the wrong column count');

base2 = run_defl(A2, b2, tol, maxit, P, Vbase, tau, 0);   % no recycling
recy2 = run_defl(A2, b2, tol, maxit, P, Vall,  tau, 0);   % recycled

fprintf('\n===== step 2 (perturbed coupling block, delta=5%%) =====\n');
fprintf('  %-34s %6s %8s %12s %12s\n', 'configuration', 'flag', 'iters', 'relres', 'true_res');
print_row(sprintf('Vbase only          (dim %d)', size(Vbase, 2)), base2);
print_row(sprintf('Vbase + recycled    (dim %d)', size(Vall,  2)), recy2);
fprintf('  ------------------------------------------------------------------\n');
fprintf('  recycled - baseline iteration gap: %+d its\n', recy2.iters - base2.iters);
fprintf('==================================================================\n');

assert(all([ref1.flag rec1.flag base2.flag recy2.flag] == 0), ...
       'a MINRES run did not converge');
assert(max([ref1.tr rec1.tr base2.tr recy2.tr]) < 1e-6, ...
       'a two-level solution residual is too large');

%% ===== sweep the recycle count ===========================================
% Re-run step 1 for each cap (the buffer keeps the LAST nrec residuals, so the
% captured block genuinely differs with the cap), then solve step 2 with it.
recs  = [0, 10, 25, 50, 100];
sweep = struct('recycle_k', {}, 'coarse_dim', {}, 'iters', {}, 'flag', {}, 'true_res', {});
for rk = recs
    if rk == 0
        Vrk = Vbase;
    else
        r1  = run_defl(A, b, tol, maxit, P, Vbase, tau, rk);
        Vrk = augment_recycle_V(Vbase, r1.W);
    end
    r2 = run_defl(A2, b2, tol, maxit, P, Vrk, tau, 0);
    sweep(end+1) = struct('recycle_k', rk, 'coarse_dim', size(Vrk, 2), ...
        'iters', r2.iters, 'flag', r2.flag, 'true_res', r2.tr); %#ok<SAGROW>
end
Tsweep = struct2table(sweep);
disp('  step-2 MINRES iterations vs # recycled Krylov vectors:');
disp(Tsweep);
writetable(Tsweep, fullfile(outDir, 'two_level_recycle.csv'));

% ---- plot: iterations vs recycle count ----------------------------------
fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
plot(recs, Tsweep.iters, '-o', 'LineWidth', 1.8, 'Color', [0.20 0.45 0.70], ...
     'MarkerFaceColor', [0.20 0.45 0.70]); hold on;
yline(base2.iters, '--', sprintf('no recycling (%d)', base2.iters), ...
      'Color', [0.35 0.35 0.35], 'LineWidth', 1.2);
grid on; xlabel('# recycled Krylov vectors'); ylabel('MINRES iterations (step 2)');
legend({'V_{base} + recycled Krylov'}, 'Location', 'northeast');
title(sprintf('Krylov recycling into the two-level coarse space (k_{base}=%d)', k));
exportgraphics(fig, fullfile(outDir, 'two_level_recycle_convergence.png'), 'Resolution', 180);
close(fig);
fprintf('[test] saved %s and %s\n', ...
        fullfile(outDir, 'two_level_recycle.csv'), ...
        fullfile(outDir, 'two_level_recycle_convergence.png'));

%==========================================================================
%  Local helpers
%==========================================================================
function V = orthonormal(Y)
    Y = real(Y);
    [V, ~] = qr(Y, 0);
end

function res = run_defl(A, b, tol, maxit, P, V, tau, nrec)
%RUN_DEFL  Two-level P^{1/2} split solve with an optional recording tap.  Mirrors
% tl_solve_krylov in stokes_immersed_rotor/define_solver_list.m.
    Afun  = @(y) P.applyCinv(A * P.applyCtinv(y));
    Ahat2 = @(z) Afun(Afun(z));
    btil  = P.applyCinv(b);
    Pdef  = src.precond.deflation_Psqrt_apply(V, Ahat2, tau, 'handle');
    [Mfun, getU] = make_recording_pdef(Pdef, numel(btil), nrec);
    t = tic;
    [y, fl, rr, it] = minres(Afun, btil, tol, maxit, Mfun);
    res = pack(P.applyCtinv(y), fl, rr, it, toc(t), A, b);
    res.W = getU();
end

function res = pack_ref(A, b, tol, maxit, P, V, tau)
%PACK_REF  The untouched production path, for the non-intrusiveness assertion.
    t = tic;
    [x, fl, rr, it] = src.precond.two_level_split_solve(A, b, tol, maxit, P, V, tau);
    res = pack(x, fl, rr, it, toc(t), A, b);
end

function [A2, b2] = perturb_coupling(A, b, meta, delta)
%PERTURB_COUPLING  Stand-in for one time step of solid motion: scale the existing
% nonzeros of the coupling block C by 1 + delta*randn, keeping A2 symmetric.
    cidx = meta.nU + meta.nP + (1:meta.nC);
    Cblk = A(cidx, 1:meta.nU);
    [i, j, v] = find(Cblk);
    Cnew = sparse(i, j, v .* (1 + delta * randn(size(v))), meta.nC, meta.nU);
    A2 = A;
    A2(cidx, 1:meta.nU) = Cnew;
    A2(1:meta.nU, cidx) = Cnew';
    b2 = b;
end

function s = pack(x, fl, rr, it, tm, A, b)
    s = struct('x', x, 'flag', fl, 'relres', rr, 'iters', it, 'time', tm, ...
               'tr', norm(b - A*x)/norm(b), 'W', zeros(size(x, 1), 0));
end

function print_row(tag, s)
    fprintf('  %-34s %6d %8d %12.2e %12.2e\n', tag, s.flag, s.iters, s.relres, s.tr);
end
