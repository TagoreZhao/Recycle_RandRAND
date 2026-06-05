function info = subspace_capture(V_true, V_comp, thresholds)
%SUBSPACE_CAPTURE  Measure how well span(V_comp) captures each column of V_true.
%
%   info = subspace_capture(V_true, V_comp) returns a struct describing
%   how accurately each column of V_true (e.g., a ground-truth eigenvector)
%   is contained in the subspace spanned by V_comp (e.g., a deflation
%   basis produced by AMG / power iteration).
%
%   info = subspace_capture(V_true, V_comp, thresholds) additionally
%   reports, for each entry t of the vector thresholds, the number of
%   columns of V_true with residual_per_vec < t.  Defaults to
%   [1e-2, 1e-3] if omitted or empty.
%
%   Inputs
%     V_true     : n-by-k matrix of "ground-truth" vectors.  Per-vector
%                  residuals are reported for these columns directly, so
%                  their order/identity is preserved.  V_true need not be
%                  orthonormal — residuals are normalized by per-column norm.
%     V_comp     : n-by-m matrix spanning the candidate subspace.  QR-
%                  orthonormalized internally, so V_comp need not be
%                  orthonormal either.
%     thresholds : (optional) vector of relative-residual cutoffs in (0, 1].
%                  Default [1e-2, 1e-3].
%
%   Output (struct)
%     residual_per_vec   k-by-1.  For column v_i of V_true,
%                          ||v_i - P*v_i||_2 / ||v_i||_2
%                        where P = Q_comp * Q_comp' is the orthogonal
%                        projector onto span(V_comp).  Lies in [0, 1];
%                        equals sin(angle(v_i, span(V_comp))) when v_i
%                        is unit-norm.  0 = perfectly captured,
%                        1 = orthogonal to span(V_comp).
%     max_residual       max(residual_per_vec) — worst-case missed column.
%     mean_residual      mean(residual_per_vec).
%     frob_residual_rel  ||V_true - P*V_true||_F / ||V_true||_F.
%     principal_angles   min(k,m)-by-1, principal angles (radians) between
%                        span(V_true) and span(V_comp), ascending.  All
%                        zero when span(V_true) ⊆ span(V_comp).  Computed
%                        via SVD of Q_comp' * Q_true (Björck–Golub).
%     thresholds         numel(thresholds)-by-1, the cutoffs used (column).
%     n_res_below        numel(thresholds)-by-1, # of columns with
%                        residual_per_vec < thresholds(i), one per cutoff.
%     n_res_below_1pct   # of columns with residual_per_vec < 1e-2 (i.e.
%                        captured to better than 1% relative residual).
%                        Always reported regardless of `thresholds`, for
%                        back-compat with existing callers.
%     n_res_below_0p1pct # of columns with residual_per_vec < 1e-3.
%                        Always reported regardless of `thresholds`.
%     k                  size(V_true, 2).
%     m                  size(V_comp, 2).
%
%   Example
%     S = load('data/bcsstk15/bcsstk15_eigs1000.mat', 'V');  % ground truth
%     V_amg = subspace_iter(Mapply, randn(n, 2*sm_eig), q);  % AMG-derived
%     info  = subspace_capture(S.V, V_amg);
%     fprintf('worst residual = %.2e, max angle = %.3f rad\n', ...
%             info.max_residual, max(info.principal_angles));
%     fprintf('%d / %d columns captured to <1%% residual\n', ...
%             info.n_res_below_1pct, info.k);

    if size(V_true, 1) ~= size(V_comp, 1)
        error('subspace_capture:dimMismatch', ...
              'V_true has %d rows but V_comp has %d.', ...
              size(V_true, 1), size(V_comp, 1));
    end

    if nargin < 3 || isempty(thresholds)
        thresholds = [1e-2, 1e-3];
    end
    if ~isnumeric(thresholds) || ~isvector(thresholds) || ...
            any(~isfinite(thresholds)) || any(thresholds <= 0)
        error('subspace_capture:badThresholds', ...
              'thresholds must be a vector of positive finite numbers.');
    end
    thresholds = thresholds(:);

    V_true_f = full(V_true);
    V_comp_f = full(V_comp);

    % Orthonormalize the candidate subspace so that
    %   P = Q_comp * Q_comp'
    % is the orthogonal projector onto span(V_comp).
    [Q_comp, ~] = qr(V_comp_f, 0);

    % Per-column projection residuals (preserves V_true's column identity).
    Proj = Q_comp * (Q_comp' * V_true_f);
    R    = V_true_f - Proj;

    col_norms = vecnorm(V_true_f, 2, 1)';
    raw_res   = vecnorm(R,        2, 1)';
    res       = raw_res ./ max(col_norms, eps);
    res       = max(min(res, 1), 0);          % clamp for floating-point noise

    info.residual_per_vec  = res;
    info.max_residual      = max(res);
    info.mean_residual     = mean(res);
    info.frob_residual_rel = norm(R, 'fro') / max(norm(V_true_f, 'fro'), eps);

    % Principal angles between span(V_true) and span(V_comp).
    [Q_true, ~] = qr(V_true_f, 0);
    s = svd(Q_comp' * Q_true);
    s = max(min(s, 1), 0);
    info.principal_angles = sort(acos(s), 'ascend');

    info.thresholds         = thresholds;
    info.n_res_below        = arrayfun(@(t) sum(res < t), thresholds);
    info.n_res_below_1pct   = sum(res < 1e-2);
    info.n_res_below_0p1pct = sum(res < 1e-3);
    info.k               = size(V_true_f, 2);
    info.m               = size(V_comp_f, 2);
end
