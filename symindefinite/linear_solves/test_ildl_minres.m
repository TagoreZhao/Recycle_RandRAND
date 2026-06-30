% TEST_ILDL_MINRES  Test that an incomplete-LDL preconditioner accelerates MINRES
% on the symmetric-indefinite Stokes KKT system.
%
% Loads the (A, b) pair saved by extract_system.m, builds the SPD incomplete-LDL
% preconditioner with make_ildl_precond, FORMS THE PRECONDITIONED (split) LINEAR
% SYSTEM  C^-1 A C^-T  and passes it into MINRES, then recovers x = C^-T y.  It
% compares iterations / residual against the unpreconditioned solve and against
% the direct backslash solution, and saves a convergence plot + CSV.
%
% Run extract_system.m first.
%
% See also: make_ildl_precond, extract_system.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
repoRoot = fileparts(fileparts(thisFileDir));   % .../Recycle_RandRAND (for +src)
addpath(repoRoot);
import src.precond.*                             % make_ildl_precond
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

% ---- reference (direct) solve --------------------------------------------
x_ref = A \ b;

% ---- 1. unpreconditioned MINRES ------------------------------------------
t0 = tic;
[~, fl0, rr0, it0, rv0] = minres(A, b, tol, maxit);
time0 = toc(t0);

% ---- 2. incomplete-LDL preconditioner (no-fill) --------------------------
tb = tic;
P = make_ildl_precond(A, struct('mode', 'nofill'));
time_build = toc(tb);

% spot-check M^-1 is SPD: r' M^-1 r > 0 on random vectors
qpos = true;
for j = 1:5
    r = randn(n, 1);
    qpos = qpos && (r' * P.applyMinv(r) > 0);
end

% ---- 3. form the preconditioned (split) system and run MINRES ------------
% Solve  (C^-1 A C^-T) y = C^-1 b   with unpreconditioned MINRES, then x = C^-T y.
Afun = @(y) P.applyCinv(A * P.applyCtinv(y));
btil = P.applyCinv(b);

t1 = tic;
[ytil, fl1, rr1, it1, rv1] = minres(Afun, btil, tol, maxit);
time1 = toc(t1);
x_ildl = P.applyCtinv(ytil);

% ---- verification --------------------------------------------------------
true_res_ildl = norm(b - A * x_ildl) / norm(b);
err_vs_ref    = norm(x_ildl - x_ref) / max(norm(x_ref), eps);

fprintf('\n================ MINRES on symmetric-indefinite Stokes KKT ================\n');
fprintf('  preconditioner factor: mode=%s  nnz(L)=%d  fill ratio vs tril(A)=%.2fx\n', ...
        P.mode, P.nnzL, P.fill_ratio);
fprintf('  M^-1 SPD spot-check (5 random vectors): %s\n', tern(qpos, 'PASS', 'FAIL'));
fprintf('  %-22s %8s %8s %12s %10s\n', 'solver', 'flag', 'iters', 'relres', 'time[s]');
fprintf('  %-22s %8d %8d %12.2e %10.3f\n', 'unpreconditioned', fl0, it0, rr0, time0);
fprintf('  %-22s %8d %8d %12.2e %10.3f\n', 'incomplete-LDL (split)', fl1, it1, rr1, time1);
fprintf('  preconditioner build time: %.3f s\n', time_build);
fprintf('  true relative residual ||b-A x_ildl||/||b|| = %.2e\n', true_res_ildl);
fprintf('  ||x_ildl - x_ref|| / ||x_ref||             = %.2e\n', err_vs_ref);
if it0 > 0
    fprintf('  iteration reduction: %.1fx fewer MINRES iterations\n', it0 / max(it1,1));
end
fprintf('===========================================================================\n');

assert(fl1 == 0, 'incomplete-LDL MINRES did not converge (flag=%d)', fl1);
assert(true_res_ildl < 1e-6, 'incomplete-LDL solution residual too large');

% ---- droptol sweep (fill vs iterations) ----------------------------------
droptols = [0, 1e-4, 1e-3, 1e-2, 1e-1];
sweep = struct('droptol', {}, 'fill_ratio', {}, 'iters', {}, 'flag', {}, 'relres', {});
for d = droptols
    if d == 0
        Pd = make_ildl_precond(A, struct('mode', 'nofill'));
    else
        Pd = make_ildl_precond(A, struct('mode', 'droptol', 'droptol', d));
    end
    Ad = @(y) Pd.applyCinv(A * Pd.applyCtinv(y));
    bd = Pd.applyCinv(b);
    [~, fld, rrd, itd] = minres(Ad, bd, tol, maxit);
    sweep(end+1) = struct('droptol', d, 'fill_ratio', Pd.fill_ratio, ...
                          'iters', itd, 'flag', fld, 'relres', rrd); %#ok<SAGROW>
end
Tsweep = struct2table(sweep);
disp('  droptol sweep (0 = no-fill level-0):');
disp(Tsweep);

% ---- save CSV + convergence plot -----------------------------------------
writetable(Tsweep, fullfile(outDir, 'ildl_droptol_sweep.csv'));

fig = figure('Visible', 'off', 'Position', [100 100 720 480]);
semilogy(0:numel(rv0)-1, rv0/rv0(1), '-',  'LineWidth', 1.6, 'Color', [0.20 0.45 0.70]); hold on;
semilogy(0:numel(rv1)-1, rv1/rv1(1), '-',  'LineWidth', 1.6, 'Color', [0.85 0.40 0.32]);
grid on; xlabel('MINRES iteration'); ylabel('relative residual');
legend({sprintf('unpreconditioned (%d its)', it0), ...
        sprintf('incomplete-LDL no-fill (%d its)', it1)}, 'Location', 'northeast');
title('MINRES on symmetric-indefinite Stokes KKT: incomplete-LDL preconditioning');
exportgraphics(fig, fullfile(outDir, 'ildl_minres_convergence.png'), 'Resolution', 180);
close(fig);
fprintf('[test] saved %s and %s\n', ...
        fullfile(outDir, 'ildl_droptol_sweep.csv'), ...
        fullfile(outDir, 'ildl_minres_convergence.png'));

%==========================================================================
%  Local helpers
%==========================================================================
function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
