% TEST_TWO_SIDED_DEFLATION_PCG  Two sequential deflations of an SPD Schur system.
%
% A Gaussian power sketch of S captures the largest spectral tail, while a
% Gaussian inverse-power sketch captures the smallest tail.  The two blocks
% are concatenated and orthogonalized once,
%
%     V = orth([Vlarge, Vsmall]),
%
% and this same V is used in both deflation stages.  The second stage is built
% against the symmetrically deflated operator
%
%     S1 = P1^(1/2) * S * P1^(1/2).
%
% To retain the symmetry and positive definiteness required by PCG, the final
% inverse-preconditioner apply is the symmetric composition
%
%     M2 = P1^(1/2) * P2 * P1^(1/2).
%
% The script compares unpreconditioned PCG, the first deflation alone, and the
% two-stage construction.  It uses eigenvalues only to choose scale-aware tau
% values; no saved eigenvectors are used to construct V.

clear; clc;

thisDir = fileparts(mfilename('fullpath'));
linearSolvesDir = fileparts(thisDir);
repoRoot = fileparts(fileparts(linearSolvesDir));
addpath(repoRoot);                    % make +src available
import src.precond.*

% ---- editable experiment parameters ------------------------------------
rng(1);
kLarge = 20;
kSmall = 20;
q = 2;
tol = 1e-10;
maxitCap = 1000;

% ---- load and validate the saved Schur system ---------------------------
matFile = fullfile(thisDir, ...
    'varvisc_schur_example_bar_rotating_nu_orbiting_h0p05_step01.mat');
assert(exist(matFile, 'file') == 2, ...
    'Schur example not found: %s', matFile);

d = load(matFile, 'S', 'rhs_S', 'eigenvalues', 'y_ref');
S = d.S;
b = d.rhs_S;
lambda = sort(real(d.eigenvalues(:)), 'ascend');
yRef = d.y_ref;
n = size(S, 1);

assert(isequal(size(S), [n n]), 'S must be square.');
assert(isequal(size(b), [n 1]), 'rhs_S has the wrong size.');
assert(isequal(size(yRef), [n 1]), 'y_ref has the wrong size.');
assert(numel(lambda) == n && all(isfinite(lambda)), ...
    'The saved eigenvalue vector is invalid.');
symmetryResidual = norm(S-S', 'fro') / max(norm(S, 'fro'), eps);
assert(symmetryResidual < 1e-13, ...
    'S is not symmetric (relative symmetry residual %.3e).', symmetryResidual);

[R, cholFlag] = chol(S, 'lower');
assert(cholFlag == 0 && lambda(1) > 0, 'S must be positive definite.');

kLarge = min(kLarge, n-1);
kSmall = min(kSmall, n-1);
maxit = min(maxitCap, n);

fprintf('[two-sided deflation] loaded n=%d Schur system, kappa=%.3e\n', ...
    n, lambda(end)/lambda(1));

% ---- Gaussian sketches of both spectral tails --------------------------
Sinv = @(X) R' \ (R \ X);
OmegaLarge = randn(n, kLarge);
OmegaSmall = randn(n, kSmall);

Ylarge = subspace_iter_plain(@(X) S*X, OmegaLarge, q);
Ysmall = subspace_iter_plain(Sinv, OmegaSmall, q);
Vlarge = orth(real(Ylarge));
Vsmall = orth(real(Ysmall));
V = orth([Vlarge, Vsmall]);

assert(~isempty(V), 'The combined Gaussian sketch collapsed to rank zero.');
orthogonalityResidual = norm(V'*V-eye(size(V,2)), 'fro');
assert(orthogonalityResidual < 1e-10, ...
    'The combined basis is not orthonormal (residual %.3e).', ...
    orthogonalityResidual);

% The first stage moves the captured modes to the unresolved upper boundary.
% Because the second stage reuses the same V, its tau largely determines the
% final captured cluster.  Place that cluster at the geometric center of the
% remaining spectral interval instead of at its lower edge; this balances the
% logarithmic distance to both unresolved tails.
lambdaLower = lambda(kSmall+1);
lambdaUpper = lambda(n-kLarge);
tauLarge = lambdaUpper;
tauSmall = sqrt(lambdaLower*lambdaUpper);

% ---- stage 1: deflate S with tauLarge -----------------------------------
[P1half, E1] = deflation_Psqrt_apply(V, S, tauLarge, 'handle');
[~, coarseFlag1] = chol((E1+E1')/2);
assert(coarseFlag1 == 0, 'The stage-1 coarse matrix is not SPD.');

P1apply = @(X) P1half(P1half(X));
S1apply = @(X) P1half(S*P1half(X));

% ---- stage 2: deflate S1 with the identical V and tauSmall --------------
[P2apply, E2] = deflation_P_apply(V, S1apply, tauSmall, 'handle', 0);
[~, coarseFlag2] = chol((E2+E2')/2);
assert(coarseFlag2 == 0, 'The stage-2 coarse matrix is not SPD.');

% Symmetric composition of both inverse preconditioners for PCG on S.
PtwoApply = @(X) P1half(P2apply(P1half(X)));

% A small matrix-free sanity check of the final symmetric SPD apply.
u = randn(n,1);
v = randn(n,1);
Mu = PtwoApply(u);
Mv = PtwoApply(v);
applySymmetryResidual = abs(u'*Mv-v'*Mu) / ...
    max([abs(u'*Mv), abs(v'*Mu), eps]);
assert(applySymmetryResidual < 1e-10, ...
    'The two-stage apply is not symmetric (residual %.3e).', ...
    applySymmetryResidual);
assert(u'*Mu > 0 && v'*Mv > 0, ...
    'The two-stage apply failed a positive-definiteness check.');

% ---- PCG comparison -----------------------------------------------------
unpreconditioned = run_pcg(S, b, tol, maxit, [], yRef);
stage1 = run_pcg(S, b, tol, maxit, P1apply, yRef);
stage2 = run_pcg(S, b, tol, maxit, PtwoApply, yRef);

fprintf('\n===== Gaussian two-tail deflation with one shared basis =====\n');
fprintf('  q=%d, dims: Vlarge=%d, Vsmall=%d, orth([Vlarge,Vsmall])=%d\n', ...
    q, size(Vlarge,2), size(Vsmall,2), size(V,2));
fprintf('  tau_large=%.6e, tau_small=%.6e\n', tauLarge, tauSmall);
fprintf('  %-24s %5s %7s %12s %12s %12s\n', ...
    'configuration', 'flag', 'iters', 'relres', 'true_res', 'ref_error');
print_row('unpreconditioned', unpreconditioned);
print_row('stage 1: tau_large', stage1);
print_row('stage 2: tau_small', stage2);
fprintf(['\n  Note: because both stages use the same combined V, the second ', ...
    'tau can replace\n  much of the first-stage relocation on accurately captured modes.\n']);
fprintf('=============================================================\n');

runs = [unpreconditioned, stage1, stage2];
assert(all([runs.flag] == 0), 'At least one PCG run did not converge.');
assert(all(isfinite([runs.trueResidual])) && ...
       max([runs.trueResidual]) <= 50*tol, ...
    'A PCG true residual is inconsistent with the requested tolerance.');
assert(all(isfinite([runs.referenceError])) && ...
       max([runs.referenceError]) <= 1e-6, ...
    'A PCG solution differs excessively from y_ref.');

% ---- convergence plot ---------------------------------------------------
outDir = fullfile(linearSolvesDir, 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end
outFile = fullfile(outDir, 'two_sided_deflation_pcg_convergence.png');

fig = figure('Visible', 'off', 'Position', [100 100 780 500]);
semilogy(0:numel(unpreconditioned.resvec)-1, ...
    unpreconditioned.resvec/unpreconditioned.resvec(1), ...
    'LineWidth', 1.5, 'DisplayName', 'unpreconditioned');
hold on;
semilogy(0:numel(stage1.resvec)-1, stage1.resvec/stage1.resvec(1), ...
    'LineWidth', 1.7, 'DisplayName', 'stage 1: \tau_{large}');
semilogy(0:numel(stage2.resvec)-1, stage2.resvec/stage2.resvec(1), ...
    'LineWidth', 1.7, 'DisplayName', 'stage 2: \tau_{small}');
grid on;
xlabel('PCG iteration');
ylabel('relative residual norm');
title(sprintf('Two-stage Schur deflation with shared Gaussian basis (q=%d, dim V=%d)', ...
    q, size(V,2)));
legend('Location', 'northeast');
exportgraphics(fig, outFile, 'Resolution', 180);
close(fig);
fprintf('[two-sided deflation] saved %s\n', outFile);

%==========================================================================
% Local helpers
%==========================================================================
function out = run_pcg(A, b, tol, maxit, inversePreconditioner, xRef)
    if isempty(inversePreconditioner)
        [x, flag, relres, iters, resvec] = pcg(A, b, tol, maxit);
    else
        [x, flag, relres, iters, resvec] = ...
            pcg(A, b, tol, maxit, inversePreconditioner);
    end
    out = struct( ...
        'x', x, ...
        'flag', flag, ...
        'relres', relres, ...
        'iters', iters, ...
        'resvec', resvec, ...
        'trueResidual', norm(A*x-b)/max(norm(b), eps), ...
        'referenceError', norm(x-xRef)/max(norm(xRef), eps));
end

function print_row(label, run)
    fprintf('  %-24s %5d %7d %12.3e %12.3e %12.3e\n', ...
        label, run.flag, run.iters, run.relres, ...
        run.trueResidual, run.referenceError);
end
