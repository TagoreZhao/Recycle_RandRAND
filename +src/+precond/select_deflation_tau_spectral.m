function [tau, info] = select_deflation_tau_spectral(K, P, rankV, C)
%SELECT_DEFLATION_TAU_SPECTRAL Per-system cutoff for two-level deflation.
%
%   [TAU,INFO] = SELECT_DEFLATION_TAU_SPECTRAL(K,P,RANKV)
%   [TAU,INFO] = SELECT_DEFLATION_TAU_SPECTRAL(K,P,RANKV,C)
%
% For the ILDL split operator Ahat = C^-1 K C^-T, choose the first
% non-deflated spectral magnitude as the target of the coarse correction:
%
%       cutoff = |lambda_(rankV+1)(Ahat)|,   tau = cutoff^2.
%
% The square is required because deflation_Psqrt_apply constructs its coarse
% matrix from Ahat^2.  On an exactly captured eigenvector, the corrected
% split eigenvalue is therefore relocated to +/-sqrt(tau) = +/-cutoff.
%
% P is the struct returned by make_ildl_precond.  Supplying the explicit C
% avoids rebuilding it when the caller already caches the current coordinate
% map.  The generalized problem K u = lambda (C*C') u has the spectrum of
% Ahat and keeps eigs on symmetric matrices.

    validateattributes(K, {'numeric'}, {'2d', 'square', 'nonempty'}, ...
        mfilename, 'K');
    validateattributes(rankV, {'numeric'}, ...
        {'scalar', 'integer', 'positive', 'finite'}, mfilename, 'rankV');
    n = size(K, 1);
    if rankV + 1 >= n
        error('select_deflation_tau_spectral:rankTooLarge', ...
            'rankV+1 must be smaller than the system size (%d).', n);
    end

    if nargin < 4 || isempty(C)
        required = {'L', 'Dsqrt', 'p', 's'};
        for j = 1:numel(required)
            if ~isfield(P, required{j})
                error('select_deflation_tau_spectral:badPreconditioner', ...
                    'P.%s is required to reconstruct the split factor.', ...
                    required{j});
            end
        end
        Sinv = spdiags(1 ./ P.s, 0, n, n);
        Pt = sparse(P.p, (1:n)', 1, n, n);
        C = Sinv * Pt * P.L * P.Dsqrt;
    end
    if ~isequal(size(C), [n, n])
        error('select_deflation_tau_spectral:badCoordinateMap', ...
            'C must be %d-by-%d.', n, n);
    end

    M = C * C';
    M = (M + M') / 2;
    eig_opts = struct('tol', 1e-6, 'maxit', 1000);
    try
        lambda = eigs((K + K') / 2, M, rankV + 1, ...
            'smallestabs', eig_opts);
    catch err
        failure = MException('select_deflation_tau_spectral:eigsFailed', ...
            ['Could not estimate the %d smallest-magnitude eigenvalues of ' ...
             'the current ILDL-split system: %s'], rankV + 1, err.message);
        failure = addCause(failure, err);
        throw(failure);
    end

    imag_scale = max(1, max(abs(real(lambda))));
    if any(abs(imag(lambda)) > 1e-9 * imag_scale)
        error('select_deflation_tau_spectral:complexSpectrum', ...
            'The symmetric generalized eigensolve returned non-real values.');
    end
    magnitudes = sort(abs(real(lambda(:))), 'ascend');
    cutoff = magnitudes(rankV + 1);
    tau = cutoff^2;
    if ~isfinite(cutoff) || cutoff <= 0 || ~isfinite(tau) || tau <= 0
        error('select_deflation_tau_spectral:invalidCutoff', ...
            'The spectral cutoff must be finite and positive (got %.16g).', ...
            cutoff);
    end

    info = struct('tau', tau, 'cutoff_abs', cutoff, ...
        'deflation_rank', rankV, 'eigenvalues_requested', rankV + 1);
end
