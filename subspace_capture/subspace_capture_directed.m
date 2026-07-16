function info = subspace_capture_directed(V_true, V_comp, thresholds, opts)
%SUBSPACE_CAPTURE_DIRECTED  Basis-invariant capture of a target eigenspace.
%
%   info = subspace_capture_directed(V_true, V_comp) measures how well
%   span(V_comp) captures span(V_true), where V_true is the reference
%   eigenspace and V_comp is the computed deflation subspace.
%
%   LOCAL trial version of the revised metric; once validated it will
%   replace +src/+precond/subspace_capture.m.
%
%   The metrics are the directed principal-angle residuals sin(theta_i),
%   where theta_i measure how well directions in span(V_true) are contained
%   in span(V_comp).  They depend only on the two subspaces, not on the
%   particular basis chosen for V_true (important on the sphere, where
%   near-degenerate clusters make the eigs basis arbitrary within each
%   cluster).
%
%   The sines are the singular values of the residual matrix
%       B = (I - P_comp) Q_true
%   (Knyazev & Argentati sine-based formula).  Computing them instead as
%   sqrt(1 - cos^2) from svd(Q_comp' * Q_true) loses half the digits for
%   well-captured directions (floor ~sqrt(eps) ~ 1e-8); the sine-based form
%   is accurate to ~n*eps absolutely, so log-scale error plots stay
%   meaningful down to machine precision.
%
%   If r_comp < r_true, the r_true - r_comp directions that cannot possibly
%   be captured get sin(theta) = 1 by construction (e.g. a dimension-j
%   Krylov space measured against a k > j dimensional eigenspace).
%
%   Inputs
%     V_true     : n-by-k matrix spanning the target/ground-truth eigenspace.
%     V_comp     : n-by-m matrix spanning the computed candidate subspace.
%                  Neither needs orthonormal columns: each is orthonormalized
%                  and rank-truncated internally via column-pivoted QR --
%                  callers must NOT orth() first (that would just repeat the
%                  work here).
%     thresholds : optional vector of cutoffs in (0,1]. Default [1e-2; 1e-3].
%     opts       : optional struct. Logical fields true_is_orth / comp_is_orth
%                  (default false) declare that the corresponding input
%                  already has orthonormal columns (e.g. eigs output, Lanczos
%                  with full reorthogonalization, orth output), skipping the
%                  internal QR.  The caller guarantees orthonormality AND
%                  full column rank; no check is performed.
%
%   Output (struct):
%     sin_angles_directed       r_true-by-1 directed sine residuals from
%                               span(V_true) into span(V_comp), ascending.
%                               0 = captured, 1 = missed. Basis-invariant.
%     principal_angles_directed asin(sin_angles_directed), radians.
%     eigspace_err_2            max(sin_angles_directed)
%                               = ||(I - P_comp) Q_true||_2 -- worst-case
%                               missed direction (MAIN quality metric).
%     eigspace_err_fro          sqrt(mean(sin_angles_directed.^2))
%                               = ||(I - P_comp) Q_true||_F / sqrt(r_true)
%                               -- average missed energy.
%     n_angle_below             per-threshold counts of sin(theta_i) < t.
%     n_angle_below_1pct        # directions captured to < 1% residual.
%     n_angle_below_0p1pct      # directions captured to < 0.1% residual.
%     thresholds, k, m          inputs echoed back.
%     r_true, r_comp            numerical dims of the two spans.

    if nargin < 3 || isempty(thresholds)
        thresholds = [1e-2; 1e-3];
    else
        thresholds = thresholds(:);
    end
    if ~isnumeric(thresholds) || any(~isfinite(thresholds)) || any(thresholds <= 0)
        error('subspace_capture_directed:badThresholds', ...
              'thresholds must be positive finite numbers.');
    end
    if nargin < 4 || isempty(opts)
        opts = struct();
    end
    true_is_orth = isfield(opts, 'true_is_orth') && opts.true_is_orth;
    comp_is_orth = isfield(opts, 'comp_is_orth') && opts.comp_is_orth;

    [n1, k] = size(V_true);
    [n2, m] = size(V_comp);
    if n1 ~= n2
        error('subspace_capture_directed:dimMismatch', ...
              'V_true has %d rows but V_comp has %d.', n1, n2);
    end

    % Orthonormal bases; pivoted QR with rank truncation unless the caller
    % declared the input orthonormal.
    if true_is_orth
        Q_true = full(V_true);  r_true = k;
    else
        [Q_true, r_true] = local_orth(full(V_true));
    end
    if comp_is_orth
        Q_comp = full(V_comp);  r_comp = m;
    else
        [Q_comp, r_comp] = local_orth(full(V_comp));
    end

    if r_true == 0
        sin_angles_directed       = zeros(0, 1);
        principal_angles_directed = zeros(0, 1);
        eigspace_err_2            = 0;
        eigspace_err_fro          = 0;
    elseif r_comp == 0
        sin_angles_directed       = ones(r_true, 1);
        principal_angles_directed = (pi / 2) * ones(r_true, 1);
        eigspace_err_2            = 1;
        eigspace_err_fro          = 1;
    else
        B = Q_true - Q_comp * (Q_comp' * Q_true);

        % Single-output svd computes singular values only (cheapest path);
        % they are the sines, sorted descending -- report ascending.  Clip
        % rounding excursions above 1 so asin stays real.
        sin_angles_directed = sort(min(svd(B), 1), 'ascend');

        principal_angles_directed = asin(sin_angles_directed);
        eigspace_err_2            = max(sin_angles_directed);
        eigspace_err_fro          = sqrt(mean(sin_angles_directed.^2));
    end

    n_angle_below = arrayfun(@(t) sum(sin_angles_directed < t), thresholds);

    info = struct();
    info.sin_angles_directed       = sin_angles_directed;
    info.principal_angles_directed = principal_angles_directed;
    info.eigspace_err_2            = eigspace_err_2;
    info.eigspace_err_fro          = eigspace_err_fro;
    info.n_angle_below             = n_angle_below;
    info.n_angle_below_1pct        = sum(sin_angles_directed < 1e-2);
    info.n_angle_below_0p1pct      = sum(sin_angles_directed < 1e-3);
    info.thresholds = thresholds;
    info.k      = k;
    info.m      = m;
    info.r_true = r_true;
    info.r_comp = r_comp;
end

function [Q, r] = local_orth(V)
%LOCAL_ORTH  Orthonormal basis of range(V) with numerical rank truncation.
%   Column-pivoted economy QR: |diag(R)| is nonincreasing, so truncating at
%   the rank keeps the columns of Q that actually span range(V). (Unpivoted
%   QR's diag(R) is NOT a rank indicator -- a dependent column in the middle
%   of V would leave a garbage direction inside the kept block.)

    if isempty(V)
        Q = zeros(size(V));
        r = 0;
        return;
    end

    [Q0, R, ~] = qr(V, 0);           % 3-output economy QR => column-pivoted

    d   = abs(diag(R));
    tol = max(size(V)) * eps(max(d));
    r   = sum(d > tol);
    Q   = Q0(:, 1:r);
end
