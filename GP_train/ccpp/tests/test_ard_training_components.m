function test_ard_training_components
%TEST_ARD_TRAINING_COMPONENTS  Numerical checks for the ARD GP benchmark.

    thisDir = fileparts(mfilename('fullpath'));
    ccppDir = fileparts(thisDir);
    repoRoot = fileparts(fileparts(ccppDir));
    addpath(ccppDir); addpath(repoRoot);
    rng(42, 'twister');

    n = 12; d = 4;
    X = randn(n, d); y = randn(n, 1);
    theta = [log([0.7; 1.1; 1.6; 0.9]); log(1.3); log(0.08)];
    [K, dK, D2] = ard_rbf_kernel(X, exp(theta(1:d)));
    assert(norm(K - K', 'fro') < 1e-12, 'ARD kernel is not symmetric.');
    A = exp(theta(5)) * K + exp(theta(6)) * eye(n);
    [~, cholFlag] = chol(A);
    assert(cholFlag == 0, 'Kernel system did not admit Cholesky.');

    % The dense exact factor is an exact inverse at the initial state, and
    % its symmetrically preconditioned initial operator is the identity.
    L0 = chol(A, 'lower'); testRhs = randn(n, 3);
    recycledApply = L0' \ (L0 \ testRhs);
    directApply = A \ testRhs;
    assert(norm(recycledApply - directApply, 'fro') / norm(directApply, 'fro') < 1e-12);
    split0 = L0 \ A / L0'; split0 = (split0 + split0') / 2;
    assert(max(abs(eig(split0, 'vector') - 1)) < 1e-11, ...
           'Initial recycled-exact split spectrum is not identity.');

    % A fully recycled deflation handle must retain the state-1 coarse inverse
    % after the system changes.  Rebuilding with A2 should produce a different
    % action for a generic coarse space.
    V = orth(randn(n, 4)); tauDefl = 0.5;
    [P1, E1] = src.precond.deflation_P_apply(V, A, tauDefl);
    frozenAtBuild = P1(testRhs);
    A2 = A + diag(linspace(0.02, 0.2, n));
    frozenAfterChange = P1(testRhs);
    [P2, E2] = src.precond.deflation_P_apply(V, A2, tauDefl);
    assert(norm(frozenAfterChange - frozenAtBuild, 'fro') < 10 * eps, ...
           'Fully recycled deflation changed without being rebuilt.');
    assert(norm(E2 - E1, 'fro') / norm(E1, 'fro') > 1e-4, ...
           'Test matrix change did not alter the coarse block.');
    assert(norm(P2(testRhs) - frozenAtBuild, 'fro') / norm(frozenAtBuild, 'fro') > 1e-6, ...
           'Refreshed and fully recycled deflation actions unexpectedly agree.');

    % Scaled coordinate probes make the Hutchinson average exactly equal to
    % the trace: (1/n) sum_i (sqrt(n)e_i)' B (sqrt(n)e_i) = trace(B).
    Z = sqrt(n) * eye(n);
    alpha = A \ y; U = A \ Z;
    analytic = ard_gp_gradient(alpha, U, Z, K, dK, exp(theta(5)), exp(theta(6)));
    finiteDiff = zeros(size(theta)); h = 1e-5;
    for j = 1:numel(theta)
        tp = theta; tm = theta; tp(j) = tp(j) + h; tm(j) = tm(j) - h;
        finiteDiff(j) = (nlml_per_point(X, y, tp, D2) - ...
                         nlml_per_point(X, y, tm, D2)) / (2*h);
    end
    rel = norm(analytic - finiteDiff) / max(norm(finiteDiff), eps);
    assert(rel < 2e-5, 'ARD GP gradient finite-difference error %.3e.', rel);

    lambda = [-9; -0.1; 0.5; 2; 7; 0.02];
    tails = extract_spectrum_tails(lambda, 2);
    assert(isequal(tails.small_abs, [0.02; 0.1]));
    assert(isequal(tails.large_abs, [9; 7]));
    assert(isequal(tails.large, [-9; 7]));

    lambdaSPD = [0.01; 0.1; 1; 4; 10; 20]; tau = 0.5;
    modes = {'none','small','large','both'};
    for i = 1:numel(modes)
        predicted = ideal_deflated_spectrum(lambdaSPD, modes{i}, 2, tau);
        explicit = explicit_diagonal_deflation(lambdaSPD, modes{i}, 2, tau);
        assert(norm(sort(predicted) - sort(explicit)) < 1e-11, ...
               'Ideal spectrum mismatch for mode %s.', modes{i});
    end

    Kcross = ard_rbf_cross_kernel(X(1:5,:), X(6:end,:), exp(theta(1:d)));
    assert(isequal(size(Kcross), [5, n-5]) && all(isfinite(Kcross(:))));
    fprintf('[test_ard_training_components] PASS (gradient relerr %.3e)\n', rel);
end

function value = nlml_per_point(X, y, theta, D2)
    K = ard_rbf_kernel(X, exp(theta(1:4)), D2);
    A = exp(theta(5)) * K + exp(theta(6)) * eye(size(X, 1));
    L = chol(A, 'lower'); alpha = L' \ (L \ y);
    value = (0.5*y'*alpha + sum(log(diag(L))) + 0.5*numel(y)*log(2*pi)) / numel(y);
end

function lambda = explicit_diagonal_deflation(d, mode, rankTotal, tau)
    A = diag(d); n = numel(d); [~, asc] = sort(abs(d), 'ascend');
    switch mode
        case 'none', idx = [];
        case 'small', idx = asc(1:rankTotal);
        case 'large', idx = asc(end-rankTotal+1:end);
        case 'both'
            ns = floor(rankTotal/2); nl = rankTotal - ns;
            idx = [asc(1:ns); asc(end-nl+1:end)];
    end
    if isempty(idx)
        B = A;
    else
        V = eye(n); V = V(:, idx);
        Psqrt = src.precond.deflation_Psqrt_apply(V, A, tau, 'matrix');
        B = Psqrt * A * Psqrt;
    end
    lambda = eig((B+B')/2, 'vector');
end
