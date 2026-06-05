function [L, alpha, buildTime] = build_ichol_robust(A, baseOpts, alphaList)
%BUILD_ICHOL_ROBUST  Build an incomplete-Cholesky factor, escalating diagcomp.
%   [L, alpha, buildTime] = BUILD_ICHOL_ROBUST(A, BASEOPTS) attempts ichol on
%   sparse(A) using BASEOPTS, retrying with an increasing diagonal-compensation
%   shift alpha (opts.diagcomp) until it succeeds. The shift makes ichol factor
%   the diagonally compensated matrix A + alpha*diag(A); the returned factor L
%   is still used as a preconditioner for the ORIGINAL, unmodified A, so the
%   solved system is unchanged.
%
%   Inputs
%     A         : n-by-n SPD matrix (dense or sparse).
%     baseOpts  : struct of base ichol options (type, droptol, michol, ...).
%     alphaList : (optional) nondecreasing diagcomp values to try, in order.
%                 Default [0 0.01 0.1 0.5 1 5 25].
%
%   Outputs
%     L         : lower-triangular incomplete-Cholesky factor (A ~ L*L').
%     alpha     : the diagcomp value that succeeded.
%     buildTime : wall-clock seconds for the successful ichol call only.
%
%   Example
%     opts = struct('type','ict','droptol',1e-3,'michol','on');
%     [L, alpha] = build_ichol_robust(A, opts);
%
%   Implementation notes
%     - Only the SUCCESSFUL build is timed; the escalation overhead is a
%       one-off tuning cost (in practice alpha is reused across similar systems).
%     - Re-throws the last error if every alpha fails, so callers can record it.

    if nargin < 3 || isempty(alphaList)
        alphaList = [0, 0.01, 0.1, 0.5, 1, 5, 25];
    end

    As = sparse(A);
    lastErr = [];
    for a = alphaList
        opts = baseOpts;
        opts.diagcomp = a;
        try
            t = tic;
            L = ichol(As, opts);
            buildTime = toc(t);
            alpha = a;
            return;
        catch err
            lastErr = err;
        end
    end
    rethrow(lastErr);
end
