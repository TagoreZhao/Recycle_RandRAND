% TEST_DEFLATION_MINRES  Test that a two-level deflation preconditioner
% (with the |E|^{-1} SPD-ification) accelerates MINRES on the symmetric-
% indefinite Stokes KKT system.
%
% Loads the (A, b) pair saved by extract_system.m, builds an exact deflation
% basis V from the smallest-magnitude eigenvectors of A (the near-zero modes
% straddling the origin that stall MINRES), builds the SPD deflation operator
%   P = (I - VV') + tau * V |E|^{-1} V'     (E = V'AV, |E|^{-1} = W|L|^{-1}W')
% via deflation_P_apply_indef, and runs MINRES with P as the SPD preconditioner.
% It compares iterations / residual against the unpreconditioned solve and the
% incomplete-LDL split solve, then sweeps k and tau and saves a CSV + plot.
%
% Run extract_system.m first (to produce stokes_kkt_system.mat).
%
% See also: deflation_P_apply_indef, make_ildl_precond, test_ildl_minres.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
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
k     = 500;       % deflation subspace size (swept below)
tau   = 1;         % coarse-correction weight (swept below)

% ---- reference (direct) solve --------------------------------------------
x_ref = A \ b;

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

% ---- 3. deflation as an SPD MINRES preconditioner (PRIMARY) --------------
% Exact small-magnitude eigenvectors of A: the modes straddling zero.
tvb = tic;
[V, ~] = eigs(A, k, 'smallestabs', struct('tol', 1e-6, 'maxit', 1000));
V = real(V);
[V, ~] = qr(V, 0);                 % enforce V'V = I
time_basis = toc(tvb);

[Pdef, E, decE] = deflation_P_apply_indef(V, A, tau, 'handle', 0);

% SPD spot-check on the deflation operator: r' P r > 0 on random vectors.
spd_ok = true;
for j = 1:5
    r = randn(n, 1);
    spd_ok = spd_ok && (r' * Pdef(r) > 0);
end

t3 = tic;
[x_def, fl3, rr3, it3, rv3] = minres(A, b, tol, maxit, @(r) Pdef(r));
time3 = toc(t3);

% ---- 4. (optional) ILDL + deflation, additive two-level ------------------
% B = M_ildl^{-1} + V|E|^{-1}V'  (sum of two SPD operators -> SPD).
Badd = @(r) P.applyMinv(r) + decE.Qabs(r);
t4 = tic;
[x_add, fl4, rr4, it4, rv4] = minres(A, b, tol, maxit, @(r) Badd(r));
time4 = toc(t4);

% ---- verification --------------------------------------------------------
true_res_ildl = norm(b - A * x_ildl) / norm(b);
true_res_def  = norm(b - A * x_def ) / norm(b);
true_res_add  = norm(b - A * x_add ) / norm(b);
err_def       = norm(x_def - x_ref) / max(norm(x_ref), eps);

fprintf('\n============= MINRES on symmetric-indefinite Stokes KKT =============\n');
fprintf('  deflation subspace: k=%d (smallest-|lambda| eigvecs), tau=%.3g\n', k, tau);
fprintf('  P SPD spot-check (5 random vectors): %s\n', tern(spd_ok, 'PASS', 'FAIL'));
fprintf('  %-26s %6s %8s %12s %12s %10s\n', ...
        'solver', 'flag', 'iters', 'relres', 'true_res', 'time[s]');
fprintf('  %-26s %6d %8d %12.2e %12s %10.3f\n', ...
        'unpreconditioned', fl0, it0, rr0, '-', time0);
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', ...
        'incomplete-LDL (split)', fl1, it1, rr1, true_res_ildl, time1);
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', ...
        'deflation |E|^-1 (SPD)', fl3, it3, rr3, true_res_def, time3);
fprintf('  %-26s %6d %8d %12.2e %12.2e %10.3f\n', ...
        'ILDL + deflation (add)', fl4, it4, rr4, true_res_add, time4);
fprintf('  basis build time (eigs): %.3f s   |   ILDL build: %.3f s\n', ...
        time_basis, time_ildl_build);
fprintf('  ||x_def - x_ref|| / ||x_ref|| = %.2e\n', err_def);
if it1 > 0
    % Deflation is a coarse-space CORRECTION that augments a smoother, so the
    % meaningful comparison is (ILDL + deflation) vs ILDL alone. Deflation
    % alone (it3) treats only the k near-zero modes and is expectedly weaker
    % than the full ILDL smoother (it1).
    fprintf('  deflation alone vs ILDL-only:      %.2fx iters (%d vs %d)\n', ...
            it3 / max(it1, 1), it3, it1);
    fprintf('  ILDL+deflation vs ILDL-only:       %.2fx iters (%d vs %d)\n', ...
            it4 / max(it1, 1), it4, it1);
end
fprintf('=====================================================================\n');

% Deflation operator is a valid (SPD), convergent MINRES preconditioner.
assert(fl3 == 0, 'deflation-only MINRES did not converge (flag=%d)', fl3);
assert(true_res_def < 1e-6, 'deflation-only solution residual too large: %.2e', true_res_def);
% The two-level smoother+coarse combination accelerates over the smoother alone.
assert(fl4 == 0, 'ILDL+deflation MINRES did not converge (flag=%d)', fl4);
assert(true_res_add < 1e-6, 'ILDL+deflation solution residual too large: %.2e', true_res_add);
assert(it4 <= it1, ...
       'ILDL+deflation did not reduce iters vs ILDL-only (%d vs %d)', it4, it1);

% ---- sweep over k and tau ------------------------------------------------
ks   = [100, 250, 500];
taus = [0.5, 1, 2];
sweep = struct('k', {}, 'tau', {}, 'iters', {}, 'flag', {}, ...
               'relres', {}, 'true_res', {});
for kk = ks
    [Vk, ~] = eigs(A, kk, 'smallestabs', struct('tol', 1e-6, 'maxit', 1000));
    Vk = real(Vk);
    [Vk, ~] = qr(Vk, 0);
    for tt = taus
        Pk = deflation_P_apply_indef(Vk, A, tt, 'handle', 0);
        [xk, flk, rrk, itk] = minres(A, b, tol, maxit, @(r) Pk(r));
        sweep(end+1) = struct('k', kk, 'tau', tt, 'iters', itk, 'flag', flk, ...
                              'relres', rrk, ...
                              'true_res', norm(b - A*xk)/norm(b)); %#ok<SAGROW>
    end
end
Tsweep = struct2table(sweep);
disp('  deflation sweep (k x tau):');
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
        sprintf('deflation |E|^{-1} k=%d (%d its)', k, it3), ...
        sprintf('ILDL + deflation (%d its)', it4)}, 'Location', 'northeast');
title('MINRES on Stokes KKT: ILDL vs deflation (|E|^{-1})');
exportgraphics(fig, fullfile(outDir, 'deflation_minres_convergence.png'), 'Resolution', 180);
close(fig);
fprintf('[test] saved %s and %s\n', ...
        fullfile(outDir, 'deflation_minres.csv'), ...
        fullfile(outDir, 'deflation_minres_convergence.png'));

%==========================================================================
%  Local helpers
%==========================================================================
function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
