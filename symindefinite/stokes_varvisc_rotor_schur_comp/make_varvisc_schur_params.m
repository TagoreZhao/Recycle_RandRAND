function params = make_varvisc_schur_params()
%MAKE_VARVISC_SCHUR_PARAMS  Defaults for the variable-viscosity Schur study.

    params = struct();
    params.h0 = 0.05;
    params.dt = 0.02;
    params.Tstep = 61;
    params.SOLVER_TOL = 1e-8;
    params.SOLVER_MAXIT = 1e5;

    params.sm_eig = 500;
    params.lg_eig = 500;
    params.q = 2;
    params.tau = [];
    % Each arm rebuilds at steps 1, 1+R, 1+2R, ... independently.
    % Inf freezes that arm's step-1 basis.
    params.DEFLAT_SMALL_PREC_REFRESH = Inf;
    params.DEFLAT_LARGE_PREC_REFRESH = Inf;
    params.DEFLAT_BOTH_PREC_REFRESH = Inf;
    params.skip_unprecond = false;
    params.COMPUTE_SPECTRUM = true;
    params.standalone_variants = [ ...
        struct('name', 'deflate_gaussian',       'source', 'gaussian', 'tail', 'small'), ...
        struct('name', 'deflate_gaussian_large', 'source', 'gaussian', 'tail', 'large'), ...
        struct('name', 'deflate_gaussian_both',  'source', 'gaussian', 'tail', 'both')];
end
