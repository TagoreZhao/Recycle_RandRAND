% TEST_TWO_LEVEL_MINRES  Two-level (ILDL smoother + P^{1/2} deflation coarse op)
% preconditioning for MINRES on the symmetric-indefinite Stokes KKT system.
%
% Holds the smoother (incomplete-LDL, M = C C') and the coarse space fixed and
% adds a P^{1/2} deflation coarse correction, so the difference in MINRES
% iterations is purely the coarse-correction effect.  Both runs are MINRES on the
% SAME split operator  Ahat = C^-1 A C^-T  (solve Ahat y = C^-1 b, recover
% x = C^-T y); the two-level run applies an SPD inner preconditioner G as the
% MINRES 5th argument:
%
%   coarse space  Vhat = k smallest-|lambda| eigvecs of Ahat
%   coarse op     G = (I - Vhat Vhat') + sqrt(tau) * Vhat (Vhat' Ahat^2 Vhat)^{-1/2} Vhat'
%                   = P^{1/2} of the SPD deflation preconditioner for Ahat^2,
%                     ~ |Ahat|^{-1} on range(Vhat)   (via deflation_Psqrt_apply)
%
% Because Ahat^2 is SPD the coarse matrix Ehat2 = Vhat'Ahat^2 Vhat is SPD, so no
% |.|-of-eigenvalues trick is needed (unlike the retired |Ehat|^{-1} form).  This
% is the two-level scheme B = L^-T P L^-1 (L = C) with the modernized P^{1/2}
% coarse operator.
%
% Run extract_system.m first (to produce stokes_kkt_system.mat).
%
% See also: deflation_Psqrt_apply, make_ildl_precond, test_ildl_minres,
%           test_deflation_minres, plot_eigenspectrum.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
repoRoot = fileparts(fileparts(thisFileDir));   % .../Recycle_RandRAND (for +src)
addpath(repoRoot);
import src.precond.*                             % make_ildl_precond, deflation_Psqrt_apply
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
k     = 250;       % coarse-space size (swept below)
tau   = 1;         % coarse-correction weight (swept below)

x_ref = A \ b;

% ---- ILDL smoother M = C C' (rebuild C explicitly, cf. plot_eigenspectrum) -
P    = make_ildl_precond(A, struct('mode', 'nofill'));
Sinv = spdiags(1 ./ P.s, 0, n, n);
Pt   = sparse(P.p, (1:n)', 1, n, n);          % P^T
C    = Sinv * Pt * P.L * P.Dsqrt;             % M = C C'
M    = C * C';
M    = (M + M') / 2;

% split operator and rhs (MINRES runs on Ahat for ALL rows)
Afun  = @(y) P.applyCinv(A * P.applyCtinv(y));
Ahat2 = @(z) Afun(Afun(z));                   % Ahat^2 (SPD)
btil  = P.applyCinv(b);

% ---- coarse space: smallest-|lambda| eigvecs of Ahat = C^-1 A C^-T --------
% via generalized eig (A,M): A U = M U Lam, U M-orthonormal, Vhat = C' U.
tvb = tic;
[U, ~] = eigs(A, M, k, 'smallestabs', struct('tol', 1e-6, 'maxit', 1000));
Vhat = C' * U;
Vhat = real(Vhat);
[Vhat, ~] = qr(Vhat, 0);                       % enforce Vhat'Vhat = I
time_basis = toc(tvb);

% P^{1/2} deflation coarse operator on the SMOOTHED operator Ahat (squared).
Gmult = deflation_Psqrt_apply(Vhat, Ahat2, tau, 'handle');

% SPD spot-check of the inner operator
spd_ok = true;
for j = 1:5
    r = randn(n, 1);
    spd_ok = spd_ok && (r' * Gmult(r) > 0);
end

% ---- 1. ILDL only (no inner preconditioner) ------------------------------
t1 = tic;
[y1, fl1, rr1, it1, rv1] = minres(Afun, btil, tol, maxit);
time1 = toc(t1);
x1 = P.applyCtinv(y1);

% ---- 2. two-level: ILDL + P^{1/2} deflation ------------------------------
t2 = tic;
[y2, fl2, rr2, it2, rv2] = minres(Afun, btil, tol, maxit, Gmult);
time2 = toc(t2);
x2 = P.applyCtinv(y2);

% ---- verification (true residual, never the solver resvec) ---------------
tr1 = norm(b - A*x1)/norm(b);
tr2 = norm(b - A*x2)/norm(b);
er2 = norm(x2 - x_ref)/max(norm(x_ref), eps);

fprintf('\n========== MINRES two-level: ILDL + P^{1/2} deflation ==========\n');
fprintf('  coarse space: k=%d eigvecs of Ahat=C^-1 A C^-T, tau=%.3g\n', k, tau);
fprintf('  inner-op SPD spot-check (5 random vectors): %s\n', tern(spd_ok, 'PASS', 'FAIL'));
fprintf('  %-26s %6s %8s %12s %12s %10s\n', ...
        'composition', 'flag', 'iters', 'relres', 'true_res', 'time[s]');
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', 'ILDL only',            fl1, it1, rr1, tr1, time1);
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', 'ILDL + P^{1/2} deflation', fl2, it2, rr2, tr2, time2);
fprintf('  coarse-basis build time (gen. eigs): %.3f s\n', time_basis);
fprintf('  ||x_2 - x_ref|| / ||x_ref|| = %.2e\n', er2);
if it1 > 0
    fprintf('  two-level vs ILDL-only: %.2fx iters (%d vs %d)\n', it2/max(it1,1), it2, it1);
end
fprintf('==================================================================\n');

assert(fl1 == 0 && fl2 == 0, ...
       'a MINRES run did not converge (flags %d/%d)', fl1, fl2);
assert(tr2 < 1e-6, 'two-level solution residual too large: %.2e', tr2);
assert(it2 <= it1, ...
       'two-level did not accelerate over ILDL-only (%d vs %d)', it2, it1);

% ---- sweep over k and tau ------------------------------------------------
ks   = [50, 100, 250, 500];
taus = [0.5, 1, 2];
sweep = struct('k', {}, 'tau', {}, 'iters', {}, 'flag', {}, ...
               'relres', {}, 'true_res', {});
for kk = ks
    [Uk, ~] = eigs(A, M, kk, 'smallestabs', struct('tol', 1e-6, 'maxit', 1000));
    Vk = real(C' * Uk);
    [Vk, ~] = qr(Vk, 0);
    for tt = taus
        Gk = deflation_Psqrt_apply(Vk, Ahat2, tt, 'handle');
        [ym, flm, rrm, itm] = minres(Afun, btil, tol, maxit, Gk);
        sweep(end+1) = struct('k', kk, 'tau', tt, ...
            'iters', itm, 'flag', flm, 'relres', rrm, ...
            'true_res', norm(b - A*P.applyCtinv(ym))/norm(b)); %#ok<SAGROW>
    end
end
Tsweep = struct2table(sweep);
disp('  two-level sweep (k x tau):');
disp(Tsweep);
writetable(Tsweep, fullfile(outDir, 'two_level_minres.csv'));

% ---- convergence plot ----------------------------------------------------
% NOTE: resvec is the residual of the split operator C^-1 A C^-T (same base for
% both rows), NOT the true ||b - A x|| residual, which is reported above.
fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
semilogy(0:numel(rv1)-1, rv1/rv1(1), '-',  'LineWidth', 1.6, 'Color', [0.85 0.40 0.32]); hold on;
semilogy(0:numel(rv2)-1, rv2/rv2(1), '--', 'LineWidth', 1.6, 'Color', [0.50 0.35 0.65]);
grid on; xlabel('MINRES iteration'); ylabel('relative residual (split operator)');
legend({sprintf('ILDL only (%d its)', it1), ...
        sprintf('ILDL + P^{1/2} deflation (%d its)', it2)}, 'Location', 'northeast');
title(sprintf('MINRES on Stokes KKT: two-level P^{1/2} deflation (k=%d, \\tau=%.2g)', k, tau));
exportgraphics(fig, fullfile(outDir, 'two_level_minres_convergence.png'), 'Resolution', 180);
close(fig);
fprintf('[test] saved %s and %s\n', ...
        fullfile(outDir, 'two_level_minres.csv'), ...
        fullfile(outDir, 'two_level_minres_convergence.png'));

%==========================================================================
%  Local helpers
%==========================================================================
function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
