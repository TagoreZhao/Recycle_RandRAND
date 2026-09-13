function [metrics, ritz] = varvisc_arnoldi_metrics(V,W,arn)
%VARVISC_ARNOLDI_METRICS Explicit recurrence/eigenpair residuals, no A calls.
    m = size(W,2);
    metrics = struct('vw_orth',norm(V'*W,'fro'), ...
        'ww_orth',norm(W'*W-eye(m),'fro'),'arnoldi_relation',0, ...
        'arnoldi_coupling',norm(arn.B,'fro'));
    ritz = struct('values',zeros(0,1),'relative_residual',zeros(0,1), ...
        'converged',false(0,1));
    if m == 0, return; end
    residual = arn.AW - V*arn.B - W*arn.H;
    residual(:,end) = residual(:,end) - arn.tail;
    metrics.arnoldi_relation = norm(residual,'fro') / max(norm(arn.AW,'fro'),realmin);
    H = (arn.H + arn.H') / 2;
    [Y,D] = eig(H);
    theta = real(diag(D));
    Z = W*Y;
    AZ = (arn.AW - V*(V'*arn.AW))*Y;
    absolute_residual = vecnorm(AZ - Z.*theta',2,1)';
    scale = max(max(abs(theta)),realmin);
    relative = absolute_residual ./ max(scale + abs(theta),realmin);
    [ritz.values,order] = sort(theta);
    ritz.relative_residual = relative(order);
    ritz.converged = ritz.relative_residual <= 1e-6;
end
