function params = varvisc_default_benchmark_params()
%VARVISC_DEFAULT_BENCHMARK_PARAMS Shared production solver/run parameters.
% The original and promoted-geometry drivers both consume this definition so
% an "all current solvers" comparison cannot silently drift between them.

    params.dt           = 0.02;
    params.Tstep        = 61;
    params.SOLVER_TOL   = 1e-8;
    params.SOLVER_MAXIT = 4000;
    params.h0           = 0.03;

    params.BLOCKJAC_PREC_REFRESH = 1;
    params.ILDL_PREC_REFRESH     = 1;
    params.DEFLAT_PREC_REFRESH   = Inf;
    params.DINVERSE_PREC_REFRESH = Inf;
    params.EXACT_PREC_REFRESH    = Inf;
    params.ESKETCH_REF_REFRESH   = Inf;

    params.DEFLAT_SM_EIG      = 500;
    params.DEFLAT_LG_EIG      = 0;
    params.DEFLAT_Q           = 2;
    params.SKETCH_OVERSAMPLE  = 2;
    params.DEFLAT_TAU         = 0.5;
    params.DEFLAT_CHEB_DEGREE = 4;
    params.ILDL_MODE          = 'nofill';
    params.ILDL_DROPTOL       = 1e-3;
    params.GMRES_MAXIT        = 300;
end
