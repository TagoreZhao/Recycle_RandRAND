function [x,flag,relres,iters,info] = varvisc_gaussian_augmented_solve( ...
        K,b,tol,maxit,P,V,m,tau,shared)
%VARVISC_GAUSSIAN_AUGMENTED_SOLVE Fresh projected Arnoldi + original MINRES.
% SHARED holds the same attributed setup costs for all Gaussian variants.
    Afun = @(y) P.applyCinv(K * P.applyCtinv(y));
    timer = tic;
    if m == 0
        W = zeros(size(K,1),0);
        arnoldi = struct('requested',0,'actual',0,'status',0,'start_norm',NaN, ...
            'operator_columns',0,'H',zeros(0),'B',zeros(size(V,2),0), ...
            'AW',W,'tail',zeros(size(K,1),1));
    else
        [W, arnoldi] = varvisc_build_projected_arnoldi(Afun,V,P.applyCinv(b),m);
    end
    S = [V,W];
    arnoldi_s = toc(timer);
    if m == 0, arnoldi_s = 0; end
    [x,flag,relres,iters,core] = src.precond.two_level_split_solve( ...
        K,b,tol,maxit,P,S,tau);
    info = shared;
    info.base_rank = size(V,2);
    info.augment_requested = m;
    info.augment_rank = size(W,2);
    info.total_rank = size(S,2);
    info.arnoldi_status = arnoldi.status;
    info.projected_start_norm = arnoldi.start_norm;
    info.arnoldi_s = arnoldi_s;
    info.coarse_setup_s = core.coarse_setup_s;
    info.minres_s = core.minres_s;
    info.algorithm_s = shared.factor_setup_s + shared.base_setup_s ...
        + arnoldi_s + core.coarse_setup_s + core.minres_s;
    info.arnoldi_operator_columns = arnoldi.operator_columns;
    info.coarse_operator_columns = core.coarse_operator_columns;
    info.minres_operator_columns = core.minres_operator_columns;
    info.operator_columns = arnoldi.operator_columns + core.coarse_operator_columns ...
        + core.minres_operator_columns;
    info.coarse_min_eig = min(core.coarse_eigenvalues);
    info.coarse_max_eig = max(core.coarse_eigenvalues);
    info.coarse_condition = info.coarse_max_eig / info.coarse_min_eig;
    info.details = struct('V',V,'W',W,'arnoldi',arnoldi,'core',core);
end
