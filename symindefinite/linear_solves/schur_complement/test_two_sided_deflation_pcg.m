% TEST_TWO_SIDED_DEFLATION_PCG  Compare several two-tail Schur deflations.
%
% The experiment retains the original two-stage Gaussian construction and
% adds two alternatives:
%
%   1. One standard deflation preconditioner whose basis is obtained from
%      Gaussian sketches of both spectral tails.
%
%   2. A specialized small-eigenvalue lift
%
%          P1 = I + tauLift^(-1) Vsmall Vsmall',
%
%      followed by a standard large-eigenvalue deflation of
%      P1^(1/2) S P1^(1/2).  Vsmall is computed by direct, fully
%      reorthogonalized Lanczos on S.  The randomized sketches never
%      orthogonalize between power iterations; orth is applied only once,
%      immediately before a deflation preconditioner is constructed.

clear; clc;

thisDir = fileparts(mfilename('fullpath'));
linearSolvesDir = fileparts(thisDir);
repoRoot = fileparts(fileparts(linearSolvesDir));
addpath(repoRoot);
addpath(fullfile(thisDir, 'local'));
import src.precond.*

% ---- editable experiment parameters ------------------------------------
rng(1);
kLarge = 50;
kSmall = 20;
sketchOversampling = 2;
qBoth = 1;
qPostLift = 1;
tauLift = 1e-10;
lanczosTol = 1e-12;
lanczosCheckEvery = 10;
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
assert(sketchOversampling >= 1 && isfinite(sketchOversampling), ...
    'The sketch oversampling factor must be finite and at least one.');
largeGaussianWidth = min(n, ceil(sketchOversampling*kLarge));
smallGaussianWidth = min(n, ceil(sketchOversampling*kSmall));
combinedGaussianWidth = min(n, largeGaussianWidth+smallGaussianWidth);
postLiftTargetRank = min(n, kSmall+kLarge);
postLiftSketchWidth = min(n, ceil(sketchOversampling*postLiftTargetRank));
maxit = min(maxitCap, n);
normS = lambda(end);

fprintf('[two-sided deflation] loaded n=%d Schur system, kappa=%.3e\n', ...
    n, lambda(end)/lambda(1));

check_small_eigenvalue_lift();

% ---- Gaussian sketches of both spectral tails --------------------------
Sinv = @(X) R' \ (R \ X);
OmegaLarge = randn(n, largeGaussianWidth);
OmegaSmall = randn(n, smallGaussianWidth);

Ylarge = plain_subspace_products(@(X) S*X, OmegaLarge, qBoth);
Ysmall = plain_subspace_products(Sinv, OmegaSmall, qBoth);
Vlarge = orth(real(Ylarge));
Vsmall = orth(real(Ysmall));
Vboth = orth([Vlarge, Vsmall]);

assert(~isempty(Vboth), 'The combined Gaussian sketch collapsed to rank zero.');
assert_orthonormal('combined Gaussian basis', Vboth, 1e-10);

lambdaLowerTarget = lambda(kSmall+1);
lambdaUpperTarget = lambda(n-kLarge);
tauLargeTarget = lambdaUpperTarget;
tauTarget = sqrt(lambdaLowerTarget*lambdaUpperTarget);
assert(lambdaLowerTarget <= tauTarget && tauTarget <= lambdaUpperTarget, ...
    'The target-rank tau is outside the unresolved spectral interval.');

assert(size(Vlarge,2) >= kLarge+1, ...
    'The oversampled large basis needs at least kLarge+1 columns.');
largeProjected = Vlarge'*(S*Vlarge);
largeProjected = (largeProjected+largeProjected')/2;
[largeRitzVectors, largeRitzValues] = eig(largeProjected);
[largeRitzValues, largeRitzOrder] = ...
    sort(real(diag(largeRitzValues)), 'descend');
largeRitzVectors = largeRitzVectors(:,largeRitzOrder);
upperCutoffEstimate = largeRitzValues(kLarge+1);
upperCutoffVector = Vlarge*largeRitzVectors(:,kLarge+1);
upperCutoffResidual = norm(S*upperCutoffVector- ...
    upperCutoffEstimate*upperCutoffVector);
upperCutoffRelativeError = abs(upperCutoffEstimate-lambdaUpperTarget) / ...
    max(lambdaUpperTarget,eps);

% ---- original shared-basis two-stage construction ----------------------
[Pshared1half, Eshared1] = ...
    deflation_Psqrt_apply(Vboth, S, tauLargeTarget, 'handle');
assert_spd_coarse('shared stage-1', Eshared1);

Pshared1 = @(X) Pshared1half(Pshared1half(X));
Sshared1 = @(X) Pshared1half(S*Pshared1half(X));

[Pshared2, Eshared2] = ...
    deflation_P_apply(Vboth, Sshared1, tauTarget, 'handle', 0);
assert_spd_coarse('shared stage-2', Eshared2);
Pshared = @(X) Pshared1half(Pshared2(Pshared1half(X)));
assert_spd_apply('shared two-stage apply', Pshared, n);

% ---- one standard deflator with one basis for both tails ----------------
[PsingleHalf, Esingle] = ...
    deflation_Psqrt_apply(Vboth, S, tauTarget, 'handle');
assert_spd_coarse('single both-tail', Esingle);
Psingle = @(X) PsingleHalf(PsingleHalf(X));
Ssingle = @(X) PsingleHalf(S*PsingleHalf(X));
assert_spd_apply('single both-tail apply', Psingle, n);

% ---- exact small basis by direct fully reorthogonalized Lanczos ---------
lanczosOptions = struct( ...
    'tolerance', lanczosTol, ...
    'checkEvery', lanczosCheckEvery, ...
    'maxSteps', n, ...
    'operatorNorm', normS);
[VsmallRitz, thetaSmallRitz, lanczosInfo] = ...
    fully_reorthogonalized_lanczos_smallest(S, kSmall+1, lanczosOptions);
VsmallExact = VsmallRitz(:,1:kSmall);
thetaSmall = thetaSmallRitz(1:kSmall);
lowerCutoffEstimate = thetaSmallRitz(kSmall+1);
lowerCutoffRelativeError = abs(lowerCutoffEstimate-lambdaLowerTarget) / ...
    max(lambdaLowerTarget,eps);
tauEstimate = sqrt(lowerCutoffEstimate*upperCutoffEstimate);
tauEstimateRelativeError = abs(tauEstimate-tauTarget) / max(tauTarget,eps);

assert_orthonormal('exact Lanczos small basis', VsmallExact, 1e-10);
assert(max(lanczosInfo.relativeResiduals) <= lanczosTol, ...
    'The direct Lanczos Ritz residual tolerance was not reached.');
smallEigenvalueError = norm(thetaSmall-lambda(1:kSmall), inf) / ...
    max(lambda(kSmall), eps);
assert(smallEigenvalueError <= 1e-8, ...
    'Lanczos smallest eigenvalues disagree with the saved spectrum (%.3e).', ...
    smallEigenvalueError);

% ---- specialized small-mode lift ---------------------------------------
[Plift, PliftHalf] = small_eigenvalue_lift(VsmallExact, tauLift);
Slift = @(X) PliftHalf(S*PliftHalf(X));
assert_spd_apply('small-mode lift', Plift, n);

liftedRayleigh = diag(VsmallExact' * Slift(VsmallExact));
expectedLifted = (1 + 1/tauLift) * thetaSmall;
liftedRayleighError = norm(liftedRayleigh-expectedLifted, inf) / ...
    max(expectedLifted(end), eps);
assert(liftedRayleighError <= 1e-10, ...
    'The small-mode lifting eigenvalue check failed (%.3e).', ...
    liftedRayleighError);

% ---- large-tail sketch after the lift ----------------------------------
largeSketchWidth = postLiftSketchWidth;
assert(largeSketchWidth == min(n, ...
        ceil(sketchOversampling*(kSmall+kLarge))), ...
    'Unexpected large-tail sketch width.');

OmegaLiftLarge = randn(n, largeSketchWidth);
YliftLarge = plain_subspace_products(Slift, OmegaLiftLarge, qPostLift);
VliftLarge = orth(real(YliftLarge));
assert(~isempty(VliftLarge), ...
    'The post-lift large-tail sketch collapsed to rank zero.');
assert(size(VliftLarge,2) == largeSketchWidth, ...
    ['The post-lift sketch lost numerical rank: requested %d columns, ', ...
     'retained %d.'], largeSketchWidth, size(VliftLarge,2));
assert_orthonormal('post-lift large-tail basis', VliftLarge, 1e-10);

[PlargeHalf, Elarge] = ...
    deflation_Psqrt_apply(VliftLarge, Slift, tauTarget, 'handle');
assert_spd_coarse('post-lift large-tail', Elarge);
Plarge = @(X) PlargeHalf(PlargeHalf(X));
PexactTwoTail = @(X) PliftHalf(Plarge(PliftHalf(X)));
SexactTwoTail = @(X) PlargeHalf(Slift(PlargeHalf(X)));
assert_spd_apply('small-lift plus large-tail apply', PexactTwoTail, n);

% ---- same-basis equivalence control ------------------------------------
Vequivalence = orth([VsmallExact, Vlarge]);
[PequivalentDirectHalf, ~] = ...
    deflation_Psqrt_apply(Vequivalence, S, tauTarget, 'handle');
PequivalentDirect = @(X) PequivalentDirectHalf(PequivalentDirectHalf(X));
[PequivalentLiftHalf, ~] = ...
    deflation_Psqrt_apply(Vequivalence, Slift, tauTarget, 'handle');
PequivalentLift = @(X) PliftHalf(PequivalentLiftHalf( ...
    PequivalentLiftHalf(PliftHalf(X))));
equivalenceProbe = randn(n,8);
equivalenceApplyError = norm(PequivalentDirect(equivalenceProbe)- ...
    PequivalentLift(equivalenceProbe), 'fro') / ...
    max(norm(PequivalentDirect(equivalenceProbe),'fro'),eps);
assert(equivalenceApplyError <= 1e-8, ...
    'Same-basis one-stage and lift-then-deflate applies disagree (%.3e).', ...
    equivalenceApplyError);

% ---- PCG comparison -----------------------------------------------------
unpreconditioned = run_pcg(S, b, tol, maxit, [], yRef);
sharedStage1 = run_pcg(S, b, tol, maxit, Pshared1, yRef);
sharedStage2 = run_pcg(S, b, tol, maxit, Pshared, yRef);
singleBoth = run_pcg(S, b, tol, maxit, Psingle, yRef);
liftOnly = run_pcg(S, b, tol, maxit, Plift, yRef);
liftAndLarge = run_pcg(S, b, tol, maxit, PexactTwoTail, yRef);
equivalentDirect = run_pcg(S, b, tol, maxit, PequivalentDirect, yRef);
equivalentLift = run_pcg(S, b, tol, maxit, PequivalentLift, yRef);
assert(equivalentDirect.flag == 0 && equivalentLift.flag == 0, ...
    'A same-basis equivalence PCG control did not converge.');
assert(abs(equivalentDirect.iters-equivalentLift.iters) <= 2, ...
    'Same-basis equivalent preconditioners have inconsistent iteration counts.');

singleSpectrum = estimate_extreme_spectrum(Ssingle, n);
liftLargeSpectrum = estimate_extreme_spectrum(SexactTwoTail, n);

fprintf('\n===== Two-tail Schur deflation comparison =====\n');
fprintf(['  Gaussian powers: both-tail q=%d, post-lift q=%d; ', ...
    'no intermediate orthogonalization\n'], qBoth, qPostLift);
fprintf(['  target ranks: large=%d, small=%d; oversampling=%.3g; ', ...
    'requested widths: large=%d, small=%d, combined=%d, post-lift=%d\n'], ...
    kLarge, kSmall, sketchOversampling, largeGaussianWidth, ...
    smallGaussianWidth, combinedGaussianWidth, largeSketchWidth);
fprintf(['  final basis dimensions: large=%d, small=%d, combined=%d, ', ...
    'post-lift=%d\n'], size(Vlarge,2), size(Vsmall,2), size(Vboth,2), ...
    size(VliftLarge,2));
fprintf(['  target interval: [%.6e, %.6e], geometric tau=%.6e; ', ...
    'stage-1 upper tau=%.6e, lift tau=%.1e\n'], ...
    lambdaLowerTarget, lambdaUpperTarget, tauTarget, ...
    tauLargeTarget, tauLift);
fprintf(['  cutoff estimates: lower=%.6e (relerr %.3e), ', ...
    'upper=%.6e (relerr %.3e, residual %.3e), ', ...
    'tau=%.6e (relerr %.3e)\n'], ...
    lowerCutoffEstimate, lowerCutoffRelativeError, ...
    upperCutoffEstimate, upperCutoffRelativeError, upperCutoffResidual, ...
    tauEstimate, tauEstimateRelativeError);
fprintf(['  direct Lanczos: steps=%d, restarts=0, max Ritz residual=%.3e, ', ...
    'eigenvalue error=%.3e\n'], ...
    lanczosInfo.steps, max(lanczosInfo.relativeResiduals), smallEigenvalueError);
fprintf('  lifted-mode Rayleigh error=%.3e\n', liftedRayleighError);
fprintf(['  same-basis equivalence: dim=%d, apply error=%.3e, ', ...
    'iterations=%d/%d\n'], size(Vequivalence,2), equivalenceApplyError, ...
    equivalentDirect.iters, equivalentLift.iters);
print_spectrum('single combined-tail', singleSpectrum);
print_spectrum('small lift + large', liftLargeSpectrum);
fprintf('\n  %-28s %5s %7s %12s %12s %12s\n', ...
    'configuration', 'flag', 'iters', 'relres', 'true_res', 'ref_error');
print_row('unpreconditioned', unpreconditioned);
print_row('shared stage 1', sharedStage1);
print_row('shared two-stage', sharedStage2);
print_row('single combined-tail', singleBoth);
print_row('small-mode lift only', liftOnly);
print_row('small lift + large-tail', liftAndLarge);
fprintf('================================================\n');

runs = [unpreconditioned, sharedStage1, sharedStage2, ...
    singleBoth, liftOnly, liftAndLarge];
requiredRuns = [unpreconditioned, sharedStage1, sharedStage2, ...
    singleBoth, liftAndLarge];
assert(all([requiredRuns.flag] == 0), ...
    'At least one complete preconditioning scheme did not converge.');
assert(all(isfinite([runs.trueResidual])), ...
    'A PCG run produced a non-finite true residual.');
assert(max([requiredRuns.trueResidual]) <= 50*tol, ...
    'A PCG true residual is inconsistent with the requested tolerance.');
assert(all(isfinite([runs.referenceError])), ...
    'A PCG run produced a non-finite reference error.');
assert(max([requiredRuns.referenceError]) <= 1e-6, ...
    'A PCG solution differs excessively from y_ref.');
if liftOnly.flag ~= 0
    fprintf(['[two-sided deflation] lift-only PCG is diagnostic and did not ', ...
        'converge (flag=%d, true residual %.3e).\n'], ...
        liftOnly.flag, liftOnly.trueResidual);
end

% ---- convergence plot ---------------------------------------------------
outDir = thisDir;
outFile = fullfile(outDir, 'two_sided_deflation_pcg_convergence.png');

fig = figure('Visible', 'off', 'Position', [100 100 900 560]);
plot_run(unpreconditioned, 'unpreconditioned', 1.4, '-');
hold on;
plot_run(sharedStage1, 'shared stage 1', 1.5, '--');
plot_run(sharedStage2, 'shared two-stage', 1.7, '-');
plot_run(singleBoth, 'single combined-tail', 1.7, '-');
plot_run(liftOnly, 'small-mode lift only', 1.5, '--');
plot_run(liftAndLarge, 'small lift + large-tail', 1.9, '-');
grid on;
xlabel('PCG iteration');
ylabel('relative residual norm');
title(sprintf(['Schur two-tail deflation comparison ', ...
    '(q_{both}=%d, q_{post-lift}=%d, oversampling=%.3g)'], ...
    qBoth, qPostLift, sketchOversampling));
legend('Location', 'northeast');
exportgraphics(fig, outFile, 'Resolution', 180);
close(fig);
fprintf('[two-sided deflation] saved %s\n', outFile);

%==========================================================================
% Local helpers
%==========================================================================
function Y = plain_subspace_products(apply, Omega, q)
    Y = Omega;
    for iteration = 1:q
        Y = apply(Y);
    end
end

function spectrum = estimate_extreme_spectrum(apply, n)
    options = struct('issym', true, 'isreal', true, ...
        'tol', 1e-8, 'maxit', 5000, 'disp', 0, 'p', min(100,n));
    try
        [~, minimumValue, minimumFlag] = ...
            eigs(apply, n, 1, 'smallestreal', options);
        [~, maximumValue, maximumFlag] = ...
            eigs(apply, n, 1, 'largestreal', options);
        lambdaMin = real(minimumValue(1,1));
        lambdaMax = real(maximumValue(1,1));
        if minimumFlag ~= 0 || ~isfinite(lambdaMin)
            lambdaMin = NaN;
        end
        if maximumFlag ~= 0 || ~isfinite(lambdaMax)
            lambdaMax = NaN;
        end
        spectrum = struct('lambdaMin', lambdaMin, 'lambdaMax', lambdaMax, ...
            'conditionEstimate', lambdaMax/lambdaMin);
    catch
        spectrum = struct('lambdaMin', NaN, 'lambdaMax', NaN, ...
            'conditionEstimate', NaN);
    end
end

function check_small_eigenvalue_lift()
    V = eye(5,2);
    tau = 1e-3;
    [Papply, PsqrtApply] = small_eigenvalue_lift(V, tau);
    X = reshape(1:15,5,3);
    expected = X + (1/tau)*V*(V'*X);
    compositionError = norm(PsqrtApply(PsqrtApply(X))-expected, 'fro') / ...
        max(norm(expected,'fro'),eps);
    directError = norm(Papply(X)-expected, 'fro') / ...
        max(norm(expected,'fro'),eps);
    assert(directError <= 1e-14 && compositionError <= 1e-12, ...
        'The local small-eigenvalue lifting algebra check failed.');
end

function assert_orthonormal(label, V, tolerance)
    residual = norm(V'*V-eye(size(V,2)), 'fro');
    assert(residual <= tolerance, ...
        '%s is not orthonormal (residual %.3e).', label, residual);
end

function assert_spd_coarse(label, E)
    [~, flag] = chol((E+E')/2);
    assert(flag == 0, '%s coarse matrix is not SPD.', label);
end

function assert_spd_apply(label, apply, n)
    u = randn(n,1);
    v = randn(n,1);
    Mu = apply(u);
    Mv = apply(v);
    symmetryResidual = abs(u'*Mv-v'*Mu) / ...
        max([abs(u'*Mv), abs(v'*Mu), eps]);
    assert(symmetryResidual <= 1e-9, ...
        '%s is not symmetric (residual %.3e).', label, symmetryResidual);
    assert(u'*Mu > 0 && v'*Mv > 0, '%s is not positive definite.', label);
end

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

function print_spectrum(label, spectrum)
    fprintf('  spectrum %-20s min=%11.3e max=%11.3e kappa~=%11.3e\n', ...
        label, spectrum.lambdaMin, spectrum.lambdaMax, ...
        spectrum.conditionEstimate);
end

function print_row(label, run)
    fprintf('  %-28s %5d %7d %12.3e %12.3e %12.3e\n', ...
        label, run.flag, run.iters, run.relres, ...
        run.trueResidual, run.referenceError);
end

function plot_run(run, label, lineWidth, lineStyle)
    semilogy(0:numel(run.resvec)-1, run.resvec/run.resvec(1), ...
        'LineWidth', lineWidth, 'LineStyle', lineStyle, ...
        'DisplayName', label);
end
