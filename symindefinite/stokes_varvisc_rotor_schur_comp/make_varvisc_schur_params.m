function params = make_varvisc_schur_params()
%MAKE_VARVISC_SCHUR_PARAMS  Defaults for the variable-viscosity Schur study.

    params = struct();
    params.h0 = 0.05;
    params.dt = 0.02;
    params.Tstep = 61;
    params.SOLVER_TOL = 1e-8;
    params.SOLVER_MAXIT = 1e5;

    params.sm_eig = 20;
    params.q = 2;
    params.tau = [];
    params.DEFLAT_PREC_REFRESH = 1e6;
    params.skip_unprecond = false;
    params.COMPUTE_SPECTRUM = true;
    params.standalone_variants = [ ...
        struct('name', 'deflate_exact', 'source', 'eig'), ...
        struct('name', 'deflate_gaussian', 'source', 'gaussian')];
end
