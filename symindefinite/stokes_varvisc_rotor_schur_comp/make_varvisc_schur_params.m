function params = make_varvisc_schur_params()
%MAKE_VARVISC_SCHUR_PARAMS  Defaults for the variable-viscosity Schur study.

    params = struct();
    params.h0 = 0.03;
    params.dt = 0.02;
    params.Tstep = 61;
    params.SOLVER_TOL = 1e-8;
    params.SOLVER_MAXIT = 1e5;

    params.sm_eig = 20;
    params.lg_eig = 100;
    params.q = 1;
    params.sketch_oversampling = 2;
    params.small_basis_source = 'lanczos';
    params.small_basis_q = 1;
    params.small_basis_lanczos_tol = 1e-12;
    params.small_basis_lanczos_check_every = 10;
    params.lift_large_q = 1;
    params.lift_tau = 1e-10;
    % params.lift_tau = [];
    params.tau = [];
    % Step 1 always builds each object. A finite R rebuilds at
    % 1,1+R,1+2R,...; Inf freezes the step-1 object.
    params.SMALL_BASIS_REFRESH = Inf;
    params.DEFLAT_SHARED_LARGE_REFRESH = Inf;
    params.DEFLAT_ADAPTIVE_LIFT_LARGE_REFRESH = Inf;
    params.skip_unprecond = false;
    params.COMPUTE_SPECTRUM = false;
    params.PLOT_EXTREME_EIGENVALUES = false;
    params.EXACT_DENSE_DIAGNOSTICS = false;
    params.SPECTRAL_RITZ_TOL = 1e-10;
    params.SPECTRAL_RITZ_MAXIT = 10000;
    params.standalone_variants = [ ...
        struct('name','deflate_shared_small', ...
               'design','shared_small'), ...
        struct('name','deflate_gaussian_large', ...
               'design','gaussian_large'), ...
        struct('name','deflate_sequential_shared_subspace', ...
               'design','sequential_shared_subspace'), ...
        struct('name','deflate_concatenated_once', ...
               'design','concatenated_once'), ...
        struct('name','deflate_adaptive_small_lift_large', ...
               'design','adaptive_small_lift_large')];
end
