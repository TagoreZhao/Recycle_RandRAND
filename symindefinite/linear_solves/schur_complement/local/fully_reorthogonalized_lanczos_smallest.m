function [V, theta, info] = fully_reorthogonalized_lanczos_smallest(A, k, options)
%FULLY_REORTHOGONALIZED_LANCZOS_SMALLEST  Stable smallest Ritz pairs.
%   This routine uses only products with the symmetric matrix A. It does not
%   factor A. Two full modified-Gram-Schmidt passes are applied at every
%   Lanczos step before the next vector is accepted.

    validateattributes(A, {'numeric'}, {'2d','square','real','finite'});
    validateattributes(k, {'numeric'}, {'scalar','integer','positive'});
    n = size(A,1);
    if k > n
        error('fully_reorthogonalized_lanczos_smallest:rankTooLarge', ...
              'k=%d exceeds the matrix dimension n=%d.', k, n);
    end

    maxSteps = min(local_option(options,'maxSteps',n), n);
    checkEvery = max(1, local_option(options,'checkEvery',10));
    tolerance = local_option(options,'tolerance',1e-12);
    operatorNorm = local_option(options,'operatorNorm',norm(A,2));
    if maxSteps < k
        error('fully_reorthogonalized_lanczos_smallest:tooFewSteps', ...
              'maxSteps must be at least k.');
    end

    Q = zeros(n, maxSteps);
    alpha = zeros(maxSteps,1);
    beta = zeros(max(maxSteps-1,1),1);
    q = randn(n,1);
    q = q/norm(q);
    qPrevious = zeros(n,1);
    betaPrevious = 0;
    converged = false;
    theta = [];
    relativeResiduals = [];

    for step = 1:maxSteps
        Q(:,step) = q;
        residual = A*q-betaPrevious*qPrevious;
        alpha(step) = q'*residual;
        residual = residual-alpha(step)*q;

        activeQ = Q(:,1:step);
        for pass = 1:2
            residual = residual-activeQ*(activeQ'*residual);
        end
        betaCurrent = norm(residual);

        shouldCheck = step >= k && ...
            (mod(step,checkEvery) == 0 || step == maxSteps || betaCurrent == 0);
        if shouldCheck
            tridiagonal = diag(alpha(1:step));
            if step > 1
                offDiagonal = beta(1:step-1);
                tridiagonal = tridiagonal+ ...
                    diag(offDiagonal,1)+diag(offDiagonal,-1);
            end
            [ritzVectors,ritzValues] = eig((tridiagonal+tridiagonal')/2);
            [~,order] = sort(real(diag(ritzValues)),'ascend');
            selected = order(1:k);
            estimatedResiduals = betaCurrent*abs(ritzVectors(end,selected)) / ...
                max(operatorNorm,eps);

            if max(estimatedResiduals) <= tolerance || ...
                    step == maxSteps || betaCurrent == 0
                candidate = activeQ*ritzVectors(:,selected);
                [V,theta,relativeResiduals] = ...
                    local_refine_ritz_pairs(A,candidate,operatorNorm);
                if max(relativeResiduals) <= tolerance
                    converged = true;
                    break
                end
            end
        end

        if step == maxSteps || betaCurrent == 0
            break
        end
        beta(step) = betaCurrent;
        qPrevious = q;
        betaPrevious = betaCurrent;
        q = residual/betaCurrent;
    end

    if ~converged
        error('fully_reorthogonalized_lanczos_smallest:noConvergence', ...
              ['Fully reorthogonalized Lanczos did not converge %d ', ...
               'smallest eigenpairs in %d steps.'], k, step);
    end
    info = struct('steps',step, ...
                  'relativeResiduals',relativeResiduals, ...
                  'orthogonalityResidual',norm(V'*V-eye(k),'fro'));
end

function value = local_option(options, name, defaultValue)
    value = defaultValue;
    if nargin >= 1 && isstruct(options) && isfield(options,name) && ...
            ~isempty(options.(name))
        value = options.(name);
    end
end

function [V,theta,relativeResiduals] = local_refine_ritz_pairs( ...
        A,candidate,operatorNorm)
    basis = orth(candidate);
    projected = basis'*(A*basis);
    projected = (projected+projected')/2;
    [rotation,values] = eig(projected);
    [theta,order] = sort(real(diag(values)),'ascend');
    V = basis*rotation(:,order);
    AV = A*V;
    relativeResiduals = vecnorm(AV-V.*theta',2,1)'/max(operatorNorm,eps);
end
