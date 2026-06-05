function filt = make_balanced_cheb_filter(lambda_min, lambda_max, target_interval, opts)
%MAKE_BALANCED_CHEB_FILTER Construct a Chebyshev polynomial filter.
%
%   filt = make_balanced_cheb_filter(lambda_min, lambda_max, target_interval, opts)
%
% Inputs
%   lambda_min      estimated smallest eigenvalue of A
%   lambda_max      estimated largest eigenvalue of A
%   target_interval [a,b] in the ORIGINAL eigenvalue scale
%                    For smallest eigenvectors, use [lambda_min, cutoff].
%   opts            optional struct:
%       opts.mode       = 'left', 'right', 'interior', or 'auto'
%                       'left'     : target interval touches lambda_min
%                       'right'    : target interval touches lambda_max
%                       'interior' : two-sided balanced interval
%                       'auto'     : decide from target_interval
%       opts.degree     = [] or positive integer
%                       If empty, degree is selected automatically.
%       opts.minDegree  = 3
%       opts.maxDegree  = 80
%       opts.phi        = [].
%                       If empty: 0.3 for end interval, 0.6 for interior.
%       opts.damping    = 'sigma', 'jackson', or 'none'
%       opts.maxNewton  = 5
%       opts.newtonTol  = 1e-12
%
% Output
%   filt struct with fields:
%       filt.k          polynomial degree
%       filt.coeff      Chebyshev coefficients a_0,...,a_k
%                       rho_k(t) = sum_{ell=0}^k coeff(ell+1) T_ell(t)
%       filt.c, filt.d  scaling constants:
%                       Ahat = (A - c I)/d
%       filt.xi, filt.eta       scaled target interval in [-1,1]
%       filt.gamma              scaled filter center
%       filt.theta_gamma        acos(gamma)
%       filt.mode
%       filt.damping
%       filt.boundary_value     filter value at separating boundary
%
% Notes
%   For smallest eigenvector subspace, use:
%       opts.mode = 'left';
%       target_interval = [lambda_min, cutoff];
%
%   For the left-end case, the filter is centered at -1 and the degree is
%   selected so that rho(eta) <= phi. This is the natural one-sided version.
%
%   For an interior interval, the center gamma is adjusted so that
%       rho(xi) = rho(eta),
%   following the balancing idea in the paper.

    arguments
        lambda_min (1,1) double
        lambda_max (1,1) double
        target_interval (1,2) double
        opts.mode char = 'auto'
        opts.degree double = []
        opts.minDegree (1,1) double = 3
        opts.maxDegree (1,1) double = 80
        opts.phi double = []
        opts.damping char = 'sigma'
        opts.maxNewton (1,1) double = 5
        opts.newtonTol (1,1) double = 1e-12
    end

    if lambda_max <= lambda_min
        error('lambda_max must be larger than lambda_min.');
    end

    a = target_interval(1);
    b = target_interval(2);
    if b <= a
        error('target_interval must satisfy target_interval(1) < target_interval(2).');
    end

    % Affine map from original eigenvalue scale to [-1,1].
    c = 0.5 * (lambda_max + lambda_min);
    d = 0.5 * (lambda_max - lambda_min);

    xi  = (a - c) / d;
    eta = (b - c) / d;

    % Clamp mildly to protect against small spectral-bound errors.
    xi  = max(-1, min(1, xi));
    eta = max(-1, min(1, eta));

    if xi >= eta
        error('Scaled target interval is invalid after mapping to [-1,1].');
    end

    % Decide mode.
    mode = lower(opts.mode);
    if strcmp(mode, 'auto')
        tolEnd = 1e-10;
        if abs(xi + 1) <= tolEnd
            mode = 'left';
        elseif abs(eta - 1) <= tolEnd
            mode = 'right';
        else
            mode = 'interior';
        end
    end

    if isempty(opts.phi)
        if strcmp(mode, 'interior')
            phi = 0.6;
        else
            phi = 0.3;
        end
    else
        phi = opts.phi;
    end

    if isempty(opts.degree)
        degreeList = opts.minDegree:opts.maxDegree;
    else
        degreeList = opts.degree;
    end

    best = [];

    for k = degreeList
        g = damping_coefficients(k, opts.damping);

        switch mode
            case 'left'
                theta_gamma = pi;       % gamma = -1
                gamma = -1;

            case 'right'
                theta_gamma = 0;        % gamma = 1
                gamma = 1;

            case 'interior'
                [theta_gamma, gamma] = balance_center_newton( ...
                    xi, eta, k, g, opts.maxNewton, opts.newtonTol);

            otherwise
                error('Unknown mode: %s', mode);
        end

        coeff = cheb_filter_coefficients(k, g, theta_gamma);

        % Normalize so that rho(gamma) = 1.
        val_gamma = eval_cheb_series_scalar(coeff, gamma);
        if abs(val_gamma) < eps
            error('Normalization failed: rho(gamma) is numerically zero.');
        end
        coeff = coeff / val_gamma;

        % Boundary value used for automatic degree choice.
        switch mode
            case 'left'
                boundary_value = abs(eval_cheb_series_scalar(coeff, eta));
            case 'right'
                boundary_value = abs(eval_cheb_series_scalar(coeff, xi));
            case 'interior'
                vxi  = eval_cheb_series_scalar(coeff, xi);
                veta = eval_cheb_series_scalar(coeff, eta);
                boundary_value = max(abs(vxi), abs(veta));
        end

        candidate.k = k;
        candidate.coeff = coeff(:);
        candidate.gamma = gamma;
        candidate.theta_gamma = theta_gamma;
        candidate.boundary_value = boundary_value;

        best = candidate;

        % If degree was fixed, accept directly.
        if ~isempty(opts.degree)
            break;
        end

        % If degree was automatic, accept first degree passing the threshold.
        if boundary_value <= phi
            break;
        end
    end

    filt = struct();
    filt.k = best.k;
    filt.coeff = best.coeff;
    filt.c = c;
    filt.d = d;
    filt.lambda_min = lambda_min;
    filt.lambda_max = lambda_max;
    filt.target_interval = target_interval;
    filt.xi = xi;
    filt.eta = eta;
    filt.gamma = best.gamma;
    filt.theta_gamma = best.theta_gamma;
    filt.mode = mode;
    filt.damping = lower(opts.damping);
    filt.phi = phi;
    filt.boundary_value = best.boundary_value;

    if isempty(opts.degree) && best.boundary_value > phi
        warning(['Automatic degree selection reached maxDegree=%d, ', ...
                 'but boundary_value=%g is still larger than phi=%g.'], ...
                 opts.maxDegree, best.boundary_value, phi);
    end
end


function g = damping_coefficients(k, damping)
%DAMPING_COEFFICIENTS Return damping coefficients g_0,...,g_k.

    damping = lower(damping);
    ell = (0:k).';

    switch damping
        case 'none'
            g = ones(k+1, 1);

        case 'sigma'
            % Lanczos sigma damping:
            % g_0 = 1, g_l = sin(l theta_k)/(l theta_k),
            % theta_k = pi/(k+1).
            theta_k = pi / (k + 1);
            g = ones(k+1, 1);
            for j = 1:k
                x = j * theta_k;
                g(j+1) = sin(x) / x;
            end

        case 'jackson'
            % Jackson damping coefficients.
            % Formula follows the standard Chebyshev Jackson kernel.
            alpha = pi / (k + 2);
            g = zeros(k+1, 1);
            for j = 0:k
                g(j+1) = sin((j+1)*alpha) / ((k+2)*sin(alpha)) + ...
                         (1 - (j+1)/(k+2)) * cos(j*alpha);
            end

        otherwise
            error('Unknown damping type: %s', damping);
    end
end


function coeff = cheb_filter_coefficients(k, g, theta_gamma)
%CHEB_FILTER_COEFFICIENTS Build coefficients a_0,...,a_k.
%
% rho_k(t) = sum_{ell=0}^k a_ell T_ell(t)
% a_0 = 0.5 g_0
% a_l = g_l cos(l theta_gamma), l >= 1.

    coeff = zeros(k+1, 1);
    coeff(1) = 0.5 * g(1);

    for ell = 1:k
        coeff(ell+1) = g(ell+1) * cos(ell * theta_gamma);
    end
end


function [theta_gamma, gamma] = balance_center_newton(xi, eta, k, g, maxNewton, tol)
%BALANCE_CENTER_NEWTON Solve rho_k(xi) = rho_k(eta) in theta variable.

    theta_xi  = acos(xi);
    theta_eta = acos(eta);

    % Since cos(theta) is decreasing, theta_xi >= theta_eta.
    theta_lo = min(theta_xi, theta_eta);
    theta_hi = max(theta_xi, theta_eta);

    theta_gamma = 0.5 * (theta_xi + theta_eta);

    for it = 1:maxNewton
        f  = 0.0;
        fp = 0.0;

        for ell = 1:k
            delta = cos(ell * theta_xi) - cos(ell * theta_eta);
            f  = f  + g(ell+1) * cos(ell * theta_gamma) * delta;
            fp = fp - g(ell+1) * ell * sin(ell * theta_gamma) * delta;
        end

        if abs(f) < tol
            break;
        end

        if abs(fp) < eps
            break;
        end

        theta_new = theta_gamma - f / fp;

        % Keep the center inside the interval in theta-space.
        if theta_new < theta_lo || theta_new > theta_hi || ~isfinite(theta_new)
            theta_new = 0.5 * (theta_gamma + 0.5 * (theta_lo + theta_hi));
        end

        if abs(theta_new - theta_gamma) < tol
            theta_gamma = theta_new;
            break;
        end

        theta_gamma = theta_new;
    end

    gamma = cos(theta_gamma);
end


function val = eval_cheb_series_scalar(coeff, t)
%EVAL_CHEB_SERIES_SCALAR Evaluate sum coeff(l+1) T_l(t) at scalar t.

    k = length(coeff) - 1;

    if k == 0
        val = coeff(1);
        return;
    end

    T0 = 1.0;
    T1 = t;

    val = coeff(1) * T0 + coeff(2) * T1;

    for ell = 2:k
        T2 = 2 * t * T1 - T0;
        val = val + coeff(ell+1) * T2;
        T0 = T1;
        T1 = T2;
    end
end