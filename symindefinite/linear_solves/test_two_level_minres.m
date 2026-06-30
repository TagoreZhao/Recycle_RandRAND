% TEST_TWO_LEVEL_MINRES  Additive vs multiplicative two-level preconditioning
% for MINRES on the symmetric-indefinite Stokes KKT system.
%
% Holds the smoother (incomplete-LDL, M = C C') and the coarse space fixed and
% varies ONLY the composition, so the difference in MINRES iterations is purely
% the additive-vs-multiplicative effect.  Both runs are MINRES on the SAME split
% operator  Ahat = C^-1 A C^-T  (solve Ahat y = C^-1 b, recover x = C^-T y) with
% an SPD inner preconditioner G applied as the MINRES 5th argument:
%
%   coarse space  Vhat = k smallest-|lambda| eigvecs of Ahat   (Ehat = Vhat'*Ahat*Vhat)
%   coarse op     Qhat = Vhat |Ehat|^{-1} Vhat'    (SPD, via deflation_P_apply_indef)
%   additive        G_add  = I + Qhat                 <=>  B_add  = M^-1 + Q
%   multiplicative  G_mult = (I - Vhat Vhat') + tau Qhat  <=>  B_mult = M^-1 - ZZ' + tau Q
%
% B_mult is exactly the reference solve_deflate_M_P.m scheme B = L^-T P L^-1 (L=C,
% chol -> |Ehat|).  On range(Vhat)^perp the two coincide; on range(Vhat) the
% multiplicative form clusters the deflated eigenvalues to +/- tau while the
% additive form merely shifts them to lambda + sign(lambda).
%
% Run extract_system.m first (to produce stokes_kkt_system.mat).
%
% See also: deflation_P_apply_indef, make_ildl_precond, test_ildl_minres,
%           test_deflation_minres, plot_eigenspectrum.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
repoRoot = fileparts(fileparts(thisFileDir));   % .../Recycle_RandRAND (for +src)
addpath(repoRoot);
import src.precond.*                             % make_ildl_precond, deflation_P_apply_indef
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
tau   = 1;         % multiplicative coarse-correction weight (swept below)

x_ref = A \ b;

% ---- ILDL smoother M = C C' (rebuild C explicitly, cf. plot_eigenspectrum) -
P    = make_ildl_precond(A, struct('mode', 'nofill'));
Sinv = spdiags(1 ./ P.s, 0, n, n);
Pt   = sparse(P.p, (1:n)', 1, n, n);          % P^T
C    = Sinv * Pt * P.L * P.Dsqrt;             % M = C C'
M    = C * C';
M    = (M + M') / 2;

% split operator and rhs (MINRES runs on Ahat for ALL rows)
Afun = @(y) P.applyCinv(A * P.applyCtinv(y));
btil = P.applyCinv(b);

% ---- coarse space: smallest-|lambda| eigvecs of Ahat = C^-1 A C^-T --------
% via generalized eig (A,M): A U = M U Lam, U M-orthonormal, Vhat = C' U.
tvb = tic;
[U, ~] = eigs(A, M, k, 'smallestabs', struct('tol', 1e-6, 'maxit', 1000));
Vhat = C' * U;
Vhat = real(Vhat);
[Vhat, ~] = qr(Vhat, 0);                       % enforce Vhat'Vhat = I
time_basis = toc(tvb);

% deflation operator on the SMOOTHED operator Ahat (returns Pdef = G_mult and
% decE.Qabs = Qhat for the additive form).
[Pdef, ~, decE] = deflation_P_apply_indef(Vhat, Afun, tau, 'handle', 0);
Gadd  = @(r) r + decE.Qabs(r);                 % additive inner operator I + Qhat
Gmult = @(r) Pdef(r);                          % multiplicative inner operator

% SPD spot-check of both inner operators
spd_ok = true;
for j = 1:5
    r = randn(n, 1);
    spd_ok = spd_ok && (r' * Gadd(r) > 0) && (r' * Gmult(r) > 0);
end

% ---- 1. ILDL only (no inner preconditioner) ------------------------------
t1 = tic;
[y1, fl1, rr1, it1, rv1] = minres(Afun, btil, tol, maxit);
time1 = toc(t1);
x1 = P.applyCtinv(y1);

% ---- 2. additive two-level ----------------------------------------------
t2 = tic;
[y2, fl2, rr2, it2, rv2] = minres(Afun, btil, tol, maxit, Gadd);
time2 = toc(t2);
x2 = P.applyCtinv(y2);

% ---- 3. multiplicative two-level (reference scheme) ----------------------
t3 = tic;
[y3, fl3, rr3, it3, rv3] = minres(Afun, btil, tol, maxit, Gmult);
time3 = toc(t3);
x3 = P.applyCtinv(y3);

% ---- verification (true residual, never the solver resvec) ---------------
tr1 = norm(b - A*x1)/norm(b);
tr2 = norm(b - A*x2)/norm(b);
tr3 = norm(b - A*x3)/norm(b);
er3 = norm(x3 - x_ref)/max(norm(x_ref), eps);

fprintf('\n========== MINRES two-level: additive vs multiplicative ==========\n');
fprintf('  coarse space: k=%d eigvecs of Ahat=C^-1 A C^-T, tau=%.3g\n', k, tau);
fprintf('  inner-op SPD spot-check (5 random vectors): %s\n', tern(spd_ok, 'PASS', 'FAIL'));
fprintf('  %-26s %6s %8s %12s %12s %10s\n', ...
        'composition', 'flag', 'iters', 'relres', 'true_res', 'time[s]');
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', 'ILDL only',          fl1, it1, rr1, tr1, time1);
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', 'additive  M^-1+Q',   fl2, it2, rr2, tr2, time2);
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', 'multiplic L^-T P L^-1', fl3, it3, rr3, tr3, time3);
fprintf('  coarse-basis build time (gen. eigs): %.3f s\n', time_basis);
fprintf('  ||x_mult - x_ref|| / ||x_ref|| = %.2e\n', er3);
if it1 > 0
    fprintf('  additive       vs ILDL-only: %.2fx iters (%d vs %d)\n', it2/max(it1,1), it2, it1);
    fprintf('  multiplicative vs ILDL-only: %.2fx iters (%d vs %d)\n', it3/max(it1,1), it3, it1);
    fprintf('  multiplicative vs additive : %.2fx iters (%d vs %d)\n', it3/max(it2,1), it3, it2);
end
fprintf('==================================================================\n');

assert(fl1 == 0 && fl2 == 0 && fl3 == 0, ...
       'a MINRES run did not converge (flags %d/%d/%d)', fl1, fl2, fl3);
assert(tr2 < 1e-6 && tr3 < 1e-6, ...
       'two-level solution residual too large (add=%.2e mult=%.2e)', tr2, tr3);
assert(it2 <= it1 && it3 <= it1, ...
       'two-level did not accelerate over ILDL-only (add=%d mult=%d vs %d)', it2, it3, it1);

% ---- sweep over k and tau ------------------------------------------------
ks   = [50, 100, 250, 500];
taus = [0.5, 1, 2];
sweep = struct('k', {}, 'tau', {}, 'comp', {}, 'iters', {}, 'flag', {}, ...
               'relres', {}, 'true_res', {});
for kk = ks
    [Uk, ~] = eigs(A, M, kk, 'smallestabs', struct('tol', 1e-6, 'maxit', 1000));
    Vk = real(C' * Uk);
    [Vk, ~] = qr(Vk, 0);
    % additive is tau-independent -> record once per k
    [~, ~, dk] = deflation_P_apply_indef(Vk, Afun, 1, 'handle', 0);
    Gak = @(r) r + dk.Qabs(r);
    [ya, fla, rra, ita] = minres(Afun, btil, tol, maxit, Gak);
    sweep(end+1) = struct('k', kk, 'tau', NaN, 'comp', "additive", ...
        'iters', ita, 'flag', fla, 'relres', rra, ...
        'true_res', norm(b - A*P.applyCtinv(ya))/norm(b)); %#ok<SAGROW>
    for tt = taus
        Pmk = deflation_P_apply_indef(Vk, Afun, tt, 'handle', 0);
        [ym, flm, rrm, itm] = minres(Afun, btil, tol, maxit, @(r) Pmk(r));
        sweep(end+1) = struct('k', kk, 'tau', tt, 'comp', "multiplicative", ...
            'iters', itm, 'flag', flm, 'relres', rrm, ...
            'true_res', norm(b - A*P.applyCtinv(ym))/norm(b)); %#ok<SAGROW>
    end
end
Tsweep = struct2table(sweep);
disp('  two-level sweep (k x tau x composition):');
disp(Tsweep);
writetable(Tsweep, fullfile(outDir, 'two_level_minres.csv'));

% ---- convergence plot ----------------------------------------------------
% NOTE: resvec is the residual of the split operator C^-1 A C^-T (same for all
% three rows), NOT the true ||b - A x|| residual, which is reported above.
fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
semilogy(0:numel(rv1)-1, rv1/rv1(1), '-',  'LineWidth', 1.6, 'Color', [0.85 0.40 0.32]); hold on;
semilogy(0:numel(rv2)-1, rv2/rv2(1), '-',  'LineWidth', 1.6, 'Color', [0.30 0.65 0.35]);
semilogy(0:numel(rv3)-1, rv3/rv3(1), '--', 'LineWidth', 1.6, 'Color', [0.50 0.35 0.65]);
grid on; xlabel('MINRES iteration'); ylabel('relative residual (split operator)');
legend({sprintf('ILDL only (%d its)', it1), ...
        sprintf('additive  M^{-1}+Q (%d its)', it2), ...
        sprintf('multiplicative L^{-T}PL^{-1} (%d its)', it3)}, 'Location', 'northeast');
title(sprintf('MINRES on Stokes KKT: additive vs multiplicative two-level (k=%d, \\tau=%.2g)', k, tau));
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
