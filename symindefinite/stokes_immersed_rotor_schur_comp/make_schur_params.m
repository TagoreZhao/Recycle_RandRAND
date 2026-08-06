function params = make_schur_params()
%MAKE_SCHUR_PARAMS  Default parameters for the Schur-complement benchmark.
%   PARAMS = MAKE_SCHUR_PARAMS()
%
%   SIZING NOTE.  h0 = 0.05 gives nS = nP + nC - 1 ~ 1959, matching the sibling
%   KKT benchmark so the two studies are directly comparable.
%
%   SM_EIG IS MEASURED, NOT INHERITED.  Deflation moves the k smallest modes to
%   tau = lam_max, so the conditioning a width-k coarse space buys is exactly
%   kappa_defl(k) = lam_max / lam_{k+1}.  run_schur_spectrum measures it; at
%   h0 = 0.05, nS = 1959, kappa(S) = 1.9e5:
%
%       k        5      10      20      30      50     100     200
%       kappa  4.5e3   2.5e3   1.6e3   1.6e3   1.6e3   1.5e3   1.1e3
%
%   The curve knees at k ~ 20 (a 116x reduction) and is essentially flat out to
%   k = 100, which buys a further 7% for five times the coarse space.  sm_eig is
%   therefore 20.  The old value of 100 was copied from an experiment whose
%   deflation acted on an ICHOL-PRECONDITIONED operator and never applied here.
%   Re-run run_schur_spectrum if the geometry or h0 changes.
%
%   See also: solve_schur_sequence, run_schur_spectrum, build_sketch_V.

    params = struct();

    % --- Discretization (identical to the sibling KKT benchmark) ------------
    params.h0    = 0.05;
    params.dt    = 0.02;
    params.Tstep = 61;                  % => 60 solves

    % --- Krylov solver ------------------------------------------------------
    params.SOLVER_TOL   = 1e-8;
    params.SOLVER_MAXIT = 1e5;

    % --- Coarse space / deflation ------------------------------------------
    params.sm_eig = 20;                 % see SM_EIG NOTE above
    params.q      = 2;                  % inverse power-iteration rounds (sketch)

    % TAU.  Deflation moves every captured mode of P*S to exactly tau, so tau
    % belongs at the TOP of the retained spectrum -- put it below lambda_max and
    % the captured modes are simply relocated, not removed.  The old value (0.5)
    % was chosen against the ichol-split operator, whose spectrum tops out near
    % 1.0; on raw S lambda_max ~ 0.762.  Empty means "resolve at step 1 from the
    % measured lambda_max(S_1)"; a numeric value overrides that.
    params.tau = [];

    % --- Preconditioner refresh cadences ------------------------------------
    % Both the exact Cholesky and the deflation basis are built ONCE at step 1
    % and then RECYCLED verbatim.  Their degradation as S(t) moves is the whole
    % measurement; refreshing either one would erase it.
    params.DINVERSE_PREC_REFRESH = 1e6;
    params.DEFLAT_PREC_REFRESH   = 1e6;

    params.skip_unprecond = false;      % nS is small; the baseline is affordable

    % --- Diagnostics --------------------------------------------------------
    % S is dense and modest, so the EXACT spectrum is affordable every step --
    % no eigs tolerance games.  Condition number is the headline metric.
    params.COMPUTE_SPECTRUM = true;

    % --- Deflation arm registry --------------------------------------------
    % One entry per V-building OPERATION.  Both read the SAME operator -- the
    % Schur complement S itself, no split factor -- and differ only in how the
    % smallest-mode subspace is extracted from it:
    %
    %   'eig'       exact smallest eigenvectors of S (eigs on S^-1 through the
    %               frozen Cholesky).  The quality REFERENCE for the sketch, not
    %               a rival method.
    %   'gaussian'  Gaussian sketch of S^-1, q rounds of plain block power
    %               iteration (build_sketch_V).
    %
    % Both are built once at step 1 and reused verbatim afterwards
    % (DEFLAT_PREC_REFRESH).  Because there is no inner preconditioner there are
    % no factor coordinates for V to drift out of: V lives in the physical
    % coordinates of S and needs no transport.  Set to empty for baselines only.
    params.standalone_variants = [ ...
        struct('name', 'deflate_exact',    'source', 'eig'), ...
        struct('name', 'deflate_gaussian', 'source', 'gaussian')];
end
