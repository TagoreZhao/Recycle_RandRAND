function info = subspace_capture_directed(V_true, V_comp, thresholds)
%SUBSPACE_CAPTURE_DIRECTED  Basis-invariant capture of a target eigenspace.
%
%   info = subspace_capture_directed(V_true, V_comp) measures how well
%   span(V_comp) captures span(V_true), where V_true is the reference
%   eigenspace and V_comp is the computed deflation subspace.
%
%   LOCAL trial version of the revised metric; once validated it will
%   replace +src/+precond/subspace_capture.m.
%
%   The main metrics are directed principal-angle residuals
%
%       sin(theta_i),
%
%   where theta_i are the principal angles measuring how well directions in
%   span(V_true) are contained in span(V_comp).  Unlike the old per-column
%   residual ||v_i - P_comp v_i|| / ||v_i||, these depend only on the two
%   subspaces, not on the particular basis chosen for V_true (important on
%   the sphere, where near-degenerate clusters make the eigs basis
%   arbitrary within each cluster).
%
%   If r_comp < r_true, the r_true - r_comp directions that cannot possibly
%   be captured get sin(theta) = 1 by construction (e.g. a dimension-j
%   Krylov space measured against a k > j dimensional eigenspace).
%
%   Inputs
%     V_true     : n-by-k matrix spanning the target/ground-truth eigenspace.
%                  Columns need not be orthonormal.
%     V_comp     : n-by-m matrix spanning the computed candidate subspace.
%                  Columns need not be orthonormal (rank-truncated
%                  internally via column-pivoted QR).
%     thresholds : optional vector of cutoffs in (0,1]. Default [1e-2; 1e-3].
%
%   Output (struct) — new recommended metrics:
%     sin_angles_directed       r_true-by-1 directed sine residuals from
%                               span(V_true) into span(V_comp), ascending.
%                               0 = captured, 1 = missed. Basis-invariant.
%     principal_angles_directed asin(sin_angles_directed), radians.
%     eigspace_err_2            max(sin_angles_directed)
%                               = ||(I - P_comp) Q_true||_2 — worst-case
%                               missed direction (MAIN quality metric).
%     eigspace_err_fro          sqrt(mean(sin_angles_directed.^2))
%                               = ||(I - P_comp) Q_true||_F / sqrt(r_true)
%                               — average missed energy.
%     n_angle_below             per-threshold counts of sin(theta_i) < t.
%     n_angle_below_1pct        # directions captured to < 1% residual.
%     n_angle_below_0p1pct      # directions captured to < 0.1% residual.
%
%   Old basis-dependent diagnostics (kept for debugging / back-compat):
%     residual_per_vec, max_residual, mean_residual, frob_residual_rel,
%     n_res_below, n_res_below_1pct, n_res_below_0p1pct.
%
%   Metadata:
%     principal_angles (alias of principal_angles_directed, back-compat),
%     thresholds, k, m, r_true (numerical dim of span(V_true)),
%     r_comp (numerical dim of span(V_comp)).
%
%   Recommendation: report eigspace_err_2, eigspace_err_fro and the
%   n_angle_below counts; treat residual_per_vec as a diagnostic only.

    if nargin < 3 || isempty(thresholds)
        thresholds = [1e-2; 1e-3];
    else
        thresholds = thresholds(:);
    end
    if ~isnumeric(thresholds) || any(~isfinite(thresholds)) || any(thresholds <= 0)
        error('subspace_capture_directed:badThresholds', ...
              'thresholds must be positive finite numbers.');
    end

    [n1, k] = size(V_true);
    [n2, m] = size(V_comp);
    if n1 ~= n2
        error('subspace_capture_directed:dimMismatch', ...
              'V_true has %d rows but V_comp has %d.', n1, n2);
    end

    V_true_f = full(V_true);
    V_comp_f = full(V_comp);

    % Orthonormal bases with numerical rank truncation.
    [Q_true, r_true] = local_orth(V_true_f);
    [Q_comp, r_comp] = local_orth(V_comp_f);

    % Orthogonal projection onto span(V_comp).
    if r_comp == 0
        Pcomp_apply = @(X) zeros(size(X), 'like', X);
    else
        Pcomp_apply = @(X) Q_comp * (Q_comp' * X);
    end

    % ---------------------------------------------------------------------
    % Old basis-dependent per-column residuals (diagnostics only).
    % ---------------------------------------------------------------------
    R_cols = V_true_f - Pcomp_apply(V_true_f);

    col_norms        = vecnorm(V_true_f, 2, 1).';
    residual_per_vec = vecnorm(R_cols,   2, 1).';

    nonzero_cols = col_norms > 0;
    residual_per_vec(nonzero_cols)  = residual_per_vec(nonzero_cols) ...
                                      ./ col_norms(nonzero_cols);
    residual_per_vec(~nonzero_cols) = NaN;

    max_residual  = max(residual_per_vec, [], 'omitnan');
    mean_residual = mean(residual_per_vec, 'omitnan');

    denom_fro = norm(V_true_f, 'fro');
    if denom_fro > 0
        frob_residual_rel = norm(R_cols, 'fro') / denom_fro;
    else
        frob_residual_rel = NaN;
    end

    % ---------------------------------------------------------------------
    % New basis-invariant directed eigenspace metrics.
    %
    % The directed sines are the singular values of the residual matrix
    %       B = (I - P_comp) Q_true
    % (Knyazev & Argentati sine-based formula).  Computing them instead as
    % sqrt(1 - cos^2) from svd(Q_comp' * Q_true) loses half the digits for
    % well-captured directions (floor ~sqrt(eps) ~ 1e-8); the sine-based
    % form is accurate to ~n*eps absolutely, so log-scale error plots stay
    % meaningful down to machine precision.  When r_comp < r_true the
    % r_true - r_comp uncapturable directions come out as sin = 1
    % automatically.
    % ---------------------------------------------------------------------
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
        % they are the sines, sorted descending -- report ascending.
        sin_angles_directed = sort(min(max(svd(B), 0), 1), 'ascend');

        principal_angles_directed = asin(sin_angles_directed);
        eigspace_err_2            = max(sin_angles_directed);
        eigspace_err_fro          = sqrt(mean(sin_angles_directed.^2));
    end

    % Basis-invariant capture counts.
    n_angle_below = arrayfun(@(t) sum(sin_angles_directed < t), thresholds);
    n_angle_below_1pct   = sum(sin_angles_directed < 1e-2);
    n_angle_below_0p1pct = sum(sin_angles_directed < 1e-3);

    % Old basis-dependent capture counts, retained for compatibility.
    n_res_below = arrayfun(@(t) sum(residual_per_vec < t), thresholds);
    n_res_below_1pct   = sum(residual_per_vec < 1e-2);
    n_res_below_0p1pct = sum(residual_per_vec < 1e-3);

    % Pack output.
    info = struct();

    % New recommended metrics.
    info.sin_angles_directed       = sin_angles_directed;
    info.principal_angles_directed = principal_angles_directed;
    info.eigspace_err_2            = eigspace_err_2;
    info.eigspace_err_fro          = eigspace_err_fro;
    info.n_angle_below             = n_angle_below;
    info.n_angle_below_1pct        = n_angle_below_1pct;
    info.n_angle_below_0p1pct      = n_angle_below_0p1pct;

    % Old diagnostic metrics.
    info.residual_per_vec  = residual_per_vec;
    info.max_residual      = max_residual;
    info.mean_residual     = mean_residual;
    info.frob_residual_rel = frob_residual_rel;
    info.n_res_below       = n_res_below;
    info.n_res_below_1pct  = n_res_below_1pct;
    info.n_res_below_0p1pct = n_res_below_0p1pct;

    % Metadata (principal_angles kept as a back-compat alias).
    info.principal_angles = principal_angles_directed;
    info.thresholds       = thresholds;
    info.k      = k;
    info.m      = m;
    info.r_true = r_true;
    info.r_comp = r_comp;
end

function [Q, r] = local_orth(V)
%LOCAL_ORTH  Orthonormal basis of range(V) with numerical rank truncation.
%   Column-pivoted economy QR: |diag(R)| is nonincreasing, so truncating at
%   the rank keeps the columns of Q that actually span range(V). (Unpivoted
%   QR's diag(R) is NOT a rank indicator — a dependent column in the middle
%   of V would leave a garbage direction inside the kept block.)

    if isempty(V)
        Q = zeros(size(V));
        r = 0;
        return;
    end

    [Q0, R, ~] = qr(V, 0);           % 3-output economy QR => column-pivoted

    d = abs(diag(R));
    if isempty(d)
        Q = Q0(:, []);
        r = 0;
        return;
    end

    tol = max(size(V)) * eps(max(d));
    r   = sum(d > tol);
    Q   = Q0(:, 1:r);
end
