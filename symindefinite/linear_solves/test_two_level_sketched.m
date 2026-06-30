% TEST_TWO_LEVEL_SKETCHED  Exact vs sketched coarse space, side by side, for
% additive vs multiplicative two-level MINRES on the symmetric-indefinite Stokes
% KKT system.
%
% Same setup as test_two_level_minres.m (ILDL smoother M = C C', split operator
% Ahat = C^-1 A C^-T, MINRES solves Ahat y = C^-1 b and recovers x = C^-T y),
% but it builds TWO coarse spaces of the same size k and compares them:
%
%   exact     Vhat = qr(C' * U),  [U,~] = eigs(A, M, k, 'smallestabs')
%             (exact smallest-|lambda| eigvecs of Ahat — the reference)
%   sketched  Vhat = qr( subspace_iter_plain(Ahat^{-1}, randn(n,k), q) )
%             (Gaussian sketch + plain power iteration on the EXACT inverse
%              Ahat^{-1} = C' A^{-1} C, via one factorization dA = decomposition(A))
%
% For each coarse space we run the additive (G = I + Qhat) and multiplicative
% (G = (I - Vhat Vhat') + tau Qhat) inner preconditioners.  With exact eigvecs
% the two compositions coincide; with a sketched (approximate) coarse space the
% multiplicative form deflates span(Vhat) exactly via the Galerkin projection
% while the additive form double-counts on span(Vhat) and degrades — this test
% shows that separation.
%
% Run extract_system.m first.
%
% See also: test_two_level_minres, deflation_P_apply_indef, make_ildl_precond,
%           src.precond.subspace_iter_plain.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
repoRoot = fileparts(fileparts(thisFileDir));   % .../Recycle_RandRAND (for +src)
addpath(repoRoot);
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
k     = 250;       % coarse-space size
q     = 2;         % sketch power-iteration count (swept below)
tau   = 1;         % multiplicative coarse-correction weight

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

% exact inverse of Ahat = C^-1 A C^-T   =>   Ahat^{-1} = C' A^{-1} C
dA      = decomposition(A);                   % one exact (LDL^T) factorization
AinvFun = @(Y) C' * (dA \ (C * Y));

% ---- coarse spaces -------------------------------------------------------
te = tic;
[U, ~]   = eigs(A, M, k, 'smallestabs', struct('tol', 1e-6, 'maxit', 1000));
Vexact   = orthonormal(C' * U);
t_exact  = toc(te);

ts = tic;
Vsketch  = build_sketched(AinvFun, n, k, q);
t_sketch = toc(ts);

% ---- runs ----------------------------------------------------------------
ref = run_ildl_only(Afun, btil, tol, maxit, P, A, b);
ex  = run_both(Afun, btil, tol, maxit, P, A, b, Vexact,  tau);
sk  = run_both(Afun, btil, tol, maxit, P, A, b, Vsketch, tau);

% ---- report --------------------------------------------------------------
fprintf('\n===== Exact vs sketched coarse space (k=%d, q=%d, tau=%.3g) =====\n', k, q, tau);
fprintf('  coarse build: exact eigs %.3f s   |   sketched (randn+plain-iter) %.3f s\n', ...
        t_exact, t_sketch);
fprintf('  %-30s %6s %8s %12s %12s\n', 'configuration', 'flag', 'iters', 'relres', 'true_res');
print_row('ILDL only',                 ref);
print_row('exact   | additive',        ex.add);
print_row('exact   | multiplicative',  ex.mul);
print_row('sketched| additive',        sk.add);
print_row('sketched| multiplicative',  sk.mul);
fprintf('  ------------------------------------------------------------------\n');
fprintf('  additive - multiplicative iteration gap:  exact = %+d    sketched = %+d\n', ...
        ex.add.iters - ex.mul.iters, sk.add.iters - sk.mul.iters);
fprintf('  ||x - x_ref||/||x_ref||:  sketched mult = %.2e\n', ...
        norm(sk.mul.x - x_ref)/max(norm(x_ref), eps));
fprintf('==================================================================\n');

assert(all([ref.flag ex.add.flag ex.mul.flag sk.add.flag sk.mul.flag] == 0), ...
       'a MINRES run did not converge');
assert(max([ex.add.tr ex.mul.tr sk.add.tr sk.mul.tr]) < 1e-6, ...
       'a two-level solution residual is too large');

% ---- sweep q for the sketched coarse space -------------------------------
qs = [0, 1, 2, 3, 5];
sweep = struct('q', {}, 'comp', {}, 'iters', {}, 'flag', {}, 'true_res', {});
for qq = qs
    Vq = build_sketched(AinvFun, n, k, qq);
    rq = run_both(Afun, btil, tol, maxit, P, A, b, Vq, tau);
    sweep(end+1) = struct('q', qq, 'comp', "additive", ...
        'iters', rq.add.iters, 'flag', rq.add.flag, 'true_res', rq.add.tr); %#ok<SAGROW>
    sweep(end+1) = struct('q', qq, 'comp', "multiplicative", ...
        'iters', rq.mul.iters, 'flag', rq.mul.flag, 'true_res', rq.mul.tr); %#ok<SAGROW>
end
Tsweep = struct2table(sweep);
Tsweep.exact_additive(:)       = ex.add.iters;   % constant reference columns
Tsweep.exact_multiplicative(:) = ex.mul.iters;
disp('  sketched coarse-space sweep over q (exact iters shown for reference):');
disp(Tsweep);
writetable(Tsweep, fullfile(outDir, 'two_level_sketched.csv'));

% ---- plot: iterations vs q ----------------------------------------------
add_it = Tsweep.iters(Tsweep.comp == "additive");
mul_it = Tsweep.iters(Tsweep.comp == "multiplicative");
fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
plot(qs, add_it, '-o', 'LineWidth', 1.8, 'Color', [0.30 0.65 0.35], ...
     'MarkerFaceColor', [0.30 0.65 0.35]); hold on;
plot(qs, mul_it, '-s', 'LineWidth', 1.8, 'Color', [0.50 0.35 0.65], ...
     'MarkerFaceColor', [0.50 0.35 0.65]);
if ex.add.iters == ex.mul.iters
    yline(ex.add.iters, '--', sprintf('exact (both = %d)', ex.add.iters), ...
          'Color', [0.35 0.35 0.35], 'LineWidth', 1.2);
else
    yline(ex.add.iters, '--', sprintf('exact additive (%d)', ex.add.iters), ...
          'Color', [0.30 0.65 0.35], 'LineWidth', 1.2);
    yline(ex.mul.iters, ':',  sprintf('exact multiplicative (%d)', ex.mul.iters), ...
          'Color', [0.50 0.35 0.65], 'LineWidth', 1.2);
end
grid on; xlabel('sketch power iterations  q'); ylabel('MINRES iterations');
legend({'sketched additive', 'sketched multiplicative'}, 'Location', 'northeast');
title(sprintf('Exact vs sketched coarse space: additive vs multiplicative (k=%d)', k));
exportgraphics(fig, fullfile(outDir, 'two_level_sketched_convergence.png'), 'Resolution', 180);
close(fig);
fprintf('[test] saved %s and %s\n', ...
        fullfile(outDir, 'two_level_sketched.csv'), ...
        fullfile(outDir, 'two_level_sketched_convergence.png'));

%==========================================================================
%  Local helpers
%==========================================================================
function V = orthonormal(Y)
    Y = real(Y);
    [V, ~] = qr(Y, 0);
end

function V = build_sketched(AinvFun, n, k, q)
%BUILD_SKETCHED  Gaussian sketch + plain subspace iteration on the exact inverse.
    Omega = randn(n, k);
    Y = src.precond.subspace_iter_plain(AinvFun, Omega, q);  % q applies of Ahat^{-1}
    V = orthonormal(Y);
end

function out = run_ildl_only(Afun, btil, tol, maxit, P, A, b)
    t = tic;
    [y, fl, rr, it] = minres(Afun, btil, tol, maxit);
    out = pack(P.applyCtinv(y), fl, rr, it, toc(t), A, b);
end

function res = run_both(Afun, btil, tol, maxit, P, A, b, V, tau)
%RUN_BOTH  Additive (I+Qhat) and multiplicative ((I-VV')+tau Qhat) for one V.
    [Pdef, ~, decE] = deflation_P_apply_indef(V, Afun, tau, 'handle', 0);
    Gadd  = @(r) r + decE.Qabs(r);
    Gmult = @(r) Pdef(r);

    t = tic;
    [ya, fla, rra, ita] = minres(Afun, btil, tol, maxit, Gadd);
    res.add = pack(P.applyCtinv(ya), fla, rra, ita, toc(t), A, b);

    t = tic;
    [ym, flm, rrm, itm] = minres(Afun, btil, tol, maxit, Gmult);
    res.mul = pack(P.applyCtinv(ym), flm, rrm, itm, toc(t), A, b);
end

function s = pack(x, fl, rr, it, tm, A, b)
    s = struct('x', x, 'flag', fl, 'relres', rr, 'iters', it, 'time', tm, ...
               'tr', norm(b - A*x)/norm(b));
end

function print_row(tag, s)
    fprintf('  %-30s %6d %8d %12.2e %12.2e\n', tag, s.flag, s.iters, s.relres, s.tr);
end
