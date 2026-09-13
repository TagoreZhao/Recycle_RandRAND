function test_dynamic_tau_deflation()
%TEST_DYNAMIC_TAU_DEFLATION Unit checks for the spectral cutoff and relocation.
    here = fileparts(mfilename('fullpath'));
    repo = fileparts(fileparts(here));
    addpath(repo);

    lambda = [4; -0.1; 0.2; -1; 3; -6];
    A = diag(lambda);
    rankV = 2;
    [tau, info] = src.precond.select_deflation_tau_spectral( ...
        A, struct(), rankV, eye(size(A)));
    assert(abs(info.cutoff_abs - 1) < 1e-12);
    assert(abs(tau - 1) < 1e-12);

    V = eye(6);
    V = V(:, [2, 3]);
    A2 = @(x) A * (A * x);
    Pdef = src.precond.deflation_Psqrt_apply(V, A2, tau, 'handle');
    relocated = V' * Pdef(A * V);
    expected = diag([-info.cutoff_abs, info.cutoff_abs]);
    assert(norm(relocated - expected, 'fro') < 1e-12, ...
        'Captured modes were not relocated to +/-sqrt(tau).');

    caught = false;
    try
        src.precond.select_deflation_tau_spectral( ...
            A, struct(), size(A, 1) - 1, eye(size(A)));
    catch err
        caught = strcmp(err.identifier, ...
            'select_deflation_tau_spectral:rankTooLarge');
    end
    assert(caught, 'Invalid cutoff rank was not rejected.');
    fprintf('test_dynamic_tau_deflation: PASS\n');
end
