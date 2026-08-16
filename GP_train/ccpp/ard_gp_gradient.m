function grad = ard_gp_gradient(alpha, probeSolutions, probes, K, dKlogell, sf2, sn2)
%ARD_GP_GRADIENT  Hutchinson gradient of GP negative log marginal likelihood.
%   Hyperparameters are log lengthscales, log signal variance, and log noise
%   variance.  The result is normalized per training observation.

    n = size(K, 1);
    if numel(alpha) ~= n || size(probeSolutions, 1) ~= n || ...
            ~isequal(size(probeSolutions), size(probes))
        error('ard_gp_gradient:sizeMismatch', 'Solution/probe dimensions do not match K.');
    end
    m = size(probes, 2);
    if m < 1
        error('ard_gp_gradient:noProbes', 'At least one Hutchinson probe is required.');
    end

    d = numel(dKlogell);
    grad = zeros(d + 2, 1);
    for r = 1:d
        dA = sf2 * dKlogell{r};
        grad(r) = component_gradient(dA, alpha, probeSolutions, probes, m, n);
    end

    dA = sf2 * K;
    grad(d + 1) = component_gradient(dA, alpha, probeSolutions, probes, m, n);

    traceNoise = sn2 * mean(sum(probeSolutions .* probes, 1));
    quadNoise  = sn2 * (alpha' * alpha);
    grad(d + 2) = 0.5 * (traceNoise - quadNoise) / n;
end

function g = component_gradient(dA, alpha, U, Z, m, n)
    dAZ = dA * Z;
    traceTerm = sum(sum(U .* dAZ, 1)) / m;
    quadTerm  = alpha' * (dA * alpha);
    g = 0.5 * (traceTerm - quadTerm) / n;
end
