% TEST_DEFLATION_MINRES  Test a P^{1/2}-of-A^2 deflation preconditioner on the
% symmetric-indefinite Stokes KKT system, with an exact vs a Gaussian-sketched
% deflation subspace.
%
% Loads the (A, b) pair saved by extract_system.m and preconditions MINRES with
% the SQUARE ROOT of the SPD deflation preconditioner built for A^2:
%
%   P   = (I - VV') + tau * V (V'A^2 V)^{-1}   V',   (SPD, since A^2 is SPD)
%   M   = P^{1/2} = (I - VV') + sqrt(tau)*V (V'A^2 V)^{-1/2} V'.
%
% Because A^2 is SPD the coarse matrix E2 = V'A^2V is SPD (no |.|-of-eigenvalues
% trick is needed, unlike the |E|^{-1} form for E = V'AV). On span(V), M
% approaches (A^2)^{-1/2} = |A|^{-1}, the ideal SPD preconditioner for MINRES on
% an indefinite operator. M is built via src.precond.deflation_Psqrt_apply by
% passing the SQUARED operator A2fun(Y) = A*(A*Y) in place of A, and is used as
% the SPD MINRES preconditioner (5th argument) for A.
%
% The deflation subspace V (the smallest-|lambda| modes of A that stall MINRES)
% is built two ways and compared head-to-head:
%   - exact    : k smallest-|lambda| eigenvectors of A via eigs.
%   - sketched : a Gaussian sketch of dimension 2k applied to the EXACT inverse
%                A^{-1} (which amplifies the small modes), q=2 subspace
%                iterations, then orthonormalized -> a 2k-dim coarse space.
%
% It compares iterations / residual against the unpreconditioned solve and the
% incomplete-LDL split solve, then sweeps k and tau (both subspace methods) and
% saves a CSV + convergence plot.
%
% Run extract_system.m first (to produce stokes_kkt_system.mat).
%
% See also: deflation_Psqrt_apply, subspace_iter_plain, make_ildl_precond,
%   test_ildl_minres, test_two_level_sketched.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
repoRoot = fileparts(fileparts(thisFileDir));   % .../Recycle_RandRAND (for +src)
addpath(repoRoot);
import src.precond.*                             % make_ildl_precond, deflation_Psqrt_apply, subspace_iter_plain
rng(1);

outDir = fullfile(thisFileDir, 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ---- load system ---------------------------------------------------------
matFile = fullfile(thisFileDir, 'stokes_kkt_system.mat');
assert(exist(matFile, 'file') == 2, ...
       'stokes_kkt_system.mat not found — run extract_system.m first.');
S = load(matFile);
A = S.A;  b = S.b;
n = size(A, 1);
fprintf('[test] loaded A (n=%d, nnz=%d), b (||b||=%.3e)\n', n, nnz(A), norm(b));

tol   = 1e-8;
maxit = min(2000, n);
k     = 500;       % deflation subspace size (swept below); sketched dim = 2k
tau   = 1;         % coarse-correction weight (swept below)
qpow  = 3;         % subspace-iteration steps for the inverse sketch

% ---- shared factorization (A is factored once, reused everywhere) --------
dA = decomposition(A);          % LDL for symmetric indefinite A
AinvFun = @(Y) dA \ Y;          % exact A^{-1} apply (for the inverse sketch)
A2fun   = @(Y) A * (A * Y);     % A^2 apply (never forms A^2 explicitly)

% ---- reference (direct) solve --------------------------------------------
x_ref = dA \ b;

% ---- 1. unpreconditioned MINRES ------------------------------------------
t0 = tic;
[~, fl0, rr0, it0, rv0] = minres(A, b, tol, maxit);
time0 = toc(t0);

% ---- 2. incomplete-LDL preconditioner (split form, baseline) -------------
tb = tic;
P = make_ildl_precond(A, struct('mode', 'nofill'));
time_ildl_build = toc(tb);

Afun = @(y) P.applyCinv(A * P.applyCtinv(y));
btil = P.applyCinv(b);
t1 = tic;
[ytil, fl1, rr1, it1, rv1] = minres(Afun, btil, tol, maxit);
time1 = toc(t1);
x_ildl = P.applyCtinv(ytil);

% ---- deflation subspaces -------------------------------------------------
% Exact: k smallest-|lambda| eigenvectors of A (the modes straddling zero).
tve = tic;
Vex = build_subspace_exact(A, k);
time_basis_ex = toc(tve);

% Sketched: Gaussian sketch of dim 2k on the EXACT inverse A^{-1}, q=2 iters.
tvs = tic;
Vsk = build_subspace_sketched(AinvFun, n, k, qpow);
time_basis_sk = toc(tvs);

% ---- 3. P^{1/2}-of-A^2 deflation, exact subspace (PRIMARY) ---------------
Msqrt_ex = deflation_Psqrt_apply(Vex, A2fun, tau, 'handle');
spd_ok_ex = spd_spotcheck(Msqrt_ex, n);
t3 = tic;
[x_ex, fl3, rr3, it3, rv3] = minres(A, b, tol, maxit, @(r) Msqrt_ex(r));
time3 = toc(t3);

% ---- 4. P^{1/2}-of-A^2 deflation, sketched subspace ----------------------
Msqrt_sk = deflation_Psqrt_apply(Vsk, A2fun, tau, 'handle');
spd_ok_sk = spd_spotcheck(Msqrt_sk, n);
t4 = tic;
[x_sk, fl4, rr4, it4, rv4] = minres(A, b, tol, maxit, @(r) Msqrt_sk(r));
time4 = toc(t4);

% ---- verification --------------------------------------------------------
true_res_ildl = norm(b - A * x_ildl) / norm(b);
true_res_ex   = norm(b - A * x_ex  ) / norm(b);
true_res_sk   = norm(b - A * x_sk  ) / norm(b);
err_ex        = norm(x_ex - x_ref) / max(norm(x_ref), eps);
err_sk        = norm(x_sk - x_ref) / max(norm(x_ref), eps);

fprintf('\n============= MINRES on symmetric-indefinite Stokes KKT =============\n');
fprintf('  preconditioner: M = P^{1/2} of the A^2 deflation operator (M ~ |A|^{-1})\n');
fprintf('  deflation subspace: exact k=%d  |  sketched 2k=%d (A^{-1} sketch, q=%d), tau=%.3g\n', ...
        k, 2*k, qpow, tau);
fprintf('  P^{1/2} SPD spot-check (5 random vectors): exact %s, sketched %s\n', ...
        tern(spd_ok_ex, 'PASS', 'FAIL'), tern(spd_ok_sk, 'PASS', 'FAIL'));
fprintf('  %-26s %6s %8s %12s %12s %10s\n', ...
        'solver', 'flag', 'iters', 'relres', 'true_res', 'time[s]');
fprintf('  %-26s %6d %8d %12.2e %12s %10.3f\n', ...
        'unpreconditioned', fl0, it0, rr0, '-', time0);
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', ...
        'incomplete-LDL (split)', fl1, it1, rr1, true_res_ildl, time1);
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', ...
        'P^{1/2}-A^2 exact', fl3, it3, rr3, true_res_ex, time3);
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', ...
        'P^{1/2}-A^2 sketched', fl4, it4, rr4, true_res_sk, time4);
fprintf('  basis build: exact(eigs) %.3f s | sketched(A^{-1}) %.3f s | ILDL build %.3f s\n', ...
        time_basis_ex, time_basis_sk, time_ildl_build);
fprintf('  ||x - x_ref||/||x_ref||:  exact %.2e   sketched %.2e\n', err_ex, err_sk);
if it0 > 0
    fprintf('  P^{1/2}-A^2 exact vs unpreconditioned:    %.2fx iters (%d vs %d)\n', ...
            it3 / max(it0, 1), it3, it0);
    fprintf('  P^{1/2}-A^2 sketched vs unpreconditioned: %.2fx iters (%d vs %d)\n', ...
            it4 / max(it0, 1), it4, it0);
end
fprintf('=====================================================================\n');

% P^{1/2}-of-A^2 is a valid (SPD), convergent MINRES preconditioner, with both
% the exact and the 2k inverse-sketched coarse space, and beats no-preconditioner.
assert(spd_ok_ex && spd_ok_sk, 'P^{1/2} operator failed SPD spot-check.');
assert(fl3 == 0, 'P^{1/2}-A^2 (exact) MINRES did not converge (flag=%d)', fl3);
assert(true_res_ex < 1e-6, 'P^{1/2}-A^2 (exact) residual too large: %.2e', true_res_ex);
assert(it3 < it0, 'P^{1/2}-A^2 (exact) did not beat unpreconditioned (%d vs %d)', it3, it0);
assert(fl4 == 0, 'P^{1/2}-A^2 (sketched) MINRES did not converge (flag=%d)', fl4);
assert(true_res_sk < 1e-6, 'P^{1/2}-A^2 (sketched) residual too large: %.2e', true_res_sk);
assert(it4 < it0, 'P^{1/2}-A^2 (sketched) did not beat unpreconditioned (%d vs %d)', it4, it0);

% ---- sweep over k and tau, both subspace methods -------------------------
ks   = [100, 250, 500];
taus = [0.5, 1, 2];
methods = {'exact', 'sketched'};
sweep = struct('method', {}, 'k', {}, 'tau', {}, 'iters', {}, 'flag', {}, ...
               'relres', {}, 'true_res', {});
for kk = ks
    Vk_ex = build_subspace_exact(A, kk);
    Vk_sk = build_subspace_sketched(AinvFun, n, kk, qpow);
    for mi = 1:numel(methods)
        method = methods{mi};
        if strcmp(method, 'exact'), Vk = Vk_ex; else, Vk = Vk_sk; end
        for tt = taus
            Msqrt_k = deflation_Psqrt_apply(Vk, A2fun, tt, 'handle');
            [xk, flk, rrk, itk] = minres(A, b, tol, maxit, @(r) Msqrt_k(r));
            sweep(end+1) = struct('method', method, 'k', kk, 'tau', tt, ...
                                  'iters', itk, 'flag', flk, 'relres', rrk, ...
                                  'true_res', norm(b - A*xk)/norm(b)); %#ok<SAGROW>
        end
    end
end
Tsweep = struct2table(sweep);
disp('  P^{1/2}-A^2 deflation sweep (method x k x tau):');
disp(Tsweep);
writetable(Tsweep, fullfile(outDir, 'deflation_minres.csv'));

% ---- convergence plot ----------------------------------------------------
% NOTE: rv1 is the residual of the split operator C^-1 A C^-T; rv3/rv4 are the
% (left-)preconditioned residuals. They are NOT the true ||b - A x|| residual,
% which is reported separately above.
fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
semilogy(0:numel(rv0)-1, rv0/rv0(1), '-', 'LineWidth', 1.6, 'Color', [0.20 0.45 0.70]); hold on;
semilogy(0:numel(rv1)-1, rv1/rv1(1), '-', 'LineWidth', 1.6, 'Color', [0.85 0.40 0.32]);
semilogy(0:numel(rv3)-1, rv3/rv3(1), '-', 'LineWidth', 1.6, 'Color', [0.30 0.65 0.35]);
semilogy(0:numel(rv4)-1, rv4/rv4(1), '--', 'LineWidth', 1.4, 'Color', [0.50 0.35 0.65]);
grid on; xlabel('MINRES iteration'); ylabel('relative residual');
legend({sprintf('unpreconditioned (%d its)', it0), ...
        sprintf('incomplete-LDL no-fill (%d its)', it1), ...
        sprintf('P^{1/2}-A^2 exact k=%d (%d its)', k, it3), ...
        sprintf('P^{1/2}-A^2 sketched 2k=%d (%d its)', 2*k, it4)}, ...
        'Location', 'northeast');
title('MINRES on Stokes KKT: P^{1/2}-of-A^2 deflation (exact vs sketched)');
exportgraphics(fig, fullfile(outDir, 'deflation_minres_convergence.png'), 'Resolution', 180);
close(fig);
fprintf('[test] saved %s and %s\n', ...
        fullfile(outDir, 'deflation_minres.csv'), ...
        fullfile(outDir, 'deflation_minres_convergence.png'));

%==========================================================================
%  Local helpers
%==========================================================================
function V = build_subspace_exact(A, k)
%BUILD_SUBSPACE_EXACT  k smallest-|lambda| eigenvectors of A, orthonormalized.
    [V, ~] = eigs(A, k, 'smallestabs', struct('tol', 1e-6, 'maxit', 1000));
    V = real(V);
    [V, ~] = qr(V, 0);                 % enforce V'V = I
end

function V = build_subspace_sketched(AinvFun, n, k, q)
%BUILD_SUBSPACE_SKETCHED  Gaussian sketch of dimension 2k applied to the exact
% inverse A^{-1} (q subspace iterations), then orthonormalized -> 2k-dim basis.
% Sketching on A^{-1} amplifies the smallest-|lambda| modes of A that we want.
    Omega  = randn(n, 2 * k);
    Y      = src.precond.subspace_iter_plain(AinvFun, Omega, q);
    [V, ~] = qr(real(Y), 0);           % n x 2k, orthonormal columns
end

function ok = spd_spotcheck(Mapply, n)
%SPD_SPOTCHECK  r' M r > 0 on 5 random vectors (necessary SPD sanity check).
    ok = true;
    for j = 1:5
        r  = randn(n, 1);
        ok = ok && (r' * Mapply(r) > 0);
    end
end

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
