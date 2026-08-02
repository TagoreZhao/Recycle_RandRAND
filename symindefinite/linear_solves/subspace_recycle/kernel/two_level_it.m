function res = two_level_it(K, b, P, V, opts)
%TWO_LEVEL_IT  One two-level split MINRES solve + coarse-space health metrics.
%
%   res = TWO_LEVEL_IT(K, b, P, V, opts)
%
%   Same recipe as src.precond.two_level_split_solve (MINRES on the split
%   operator Ahat = C^-1 K C^-T with the P^{1/2} coarse correction built on
%   Ahat^2, then x = C^-T y), instrumented with the two things the diagnosis
%   needs and the production path does not return:
%
%     * the TRUE residual ||b - K x|| / ||b||.  MINRES's own relres is the
%       residual of the SPLIT system and can differ by orders of magnitude —
%       the standing convention in this folder is never to trust it.
%     * the coarse matrix health.  E = V' Ahat^2 V is SPD by construction, so a
%       stale V never errors; it just makes E^{-1/2} inject a 1/|Rayleigh|
%       multiple of a direction that is no longer an eigenvector.  cond(E) and
%       min eig(E) are how that shows up (hypothesis H4).
%
%   V = [] runs the ILDL-only baseline (plain MINRES on Ahat, no coarse space).
%
%   opts (optional): .tau (1), .tol (1e-8), .maxit (min(2000,n)),
%                    .record (0 -> no Krylov capture; k -> keep the last k
%                    ILDL-preconditioned residuals via make_recording_pdef).
%
%   res: .x .flag .relres .iters .resvec .true_res .time .setup_time
%        .coarse_dim .condE .minEigE .sqrt_minEigE .maxEigE .W
%
%   sqrt_minEigE is the more interpretable of the two: E's eigenvalues are
%   squared Rayleigh quotients of Ahat, so sqrt(min eig E) is directly
%   comparable to |lambda_min(Ahat)|.
%
%   See also: src.precond.two_level_split_solve, src.precond.deflation_Psqrt_apply,
%             make_recording_pdef, lowrank_update_basis.

    if nargin < 5 || isempty(opts), opts = struct(); end
    tau    = getdef(opts, 'tau',    1);
    tol    = getdef(opts, 'tol',    1e-8);
    n      = size(K, 1);
    maxit  = getdef(opts, 'maxit',  min(2000, n));
    record = getdef(opts, 'record', 0);

    Afun = @(y) P.applyCinv(K * P.applyCtinv(y));   % Ahat = C^-1 K C^-T
    btil = P.applyCinv(b);

    res = struct('coarse_dim', 0, 'condE', NaN, 'minEigE', NaN, ...
                 'sqrt_minEigE', NaN, 'maxEigE', NaN, 'setup_time', 0, ...
                 'W', zeros(n, 0));

    ts = tic;
    if isempty(V)
        Mfun = [];
        getW = @() zeros(n, 0);
    else
        Ahat2      = @(z) Afun(Afun(z));            % Ahat^2 (SPD)
        [Pdef, E]  = src.precond.deflation_Psqrt_apply(V, Ahat2, tau, 'handle');
        eE         = sort(real(eig(full(E))), 'ascend');
        res.coarse_dim   = size(V, 2);
        res.minEigE      = eE(1);
        res.maxEigE      = eE(end);
        res.sqrt_minEigE = sqrt(max(eE(1), 0));
        res.condE        = eE(end) / max(eE(1), realmin);
        if record > 0
            [Mfun, getW] = make_recording_pdef(Pdef, n, record);
        else
            Mfun = Pdef;
            getW = @() zeros(n, 0);
        end
    end
    res.setup_time = toc(ts);

    t0 = tic;
    if isempty(Mfun)
        [y, fl, rr, it, rv] = minres(Afun, btil, tol, maxit);
    else
        [y, fl, rr, it, rv] = minres(Afun, btil, tol, maxit, Mfun);
    end
    res.time = toc(t0);

    res.x        = P.applyCtinv(y);                 % recover x = C^-T y
    res.flag     = fl;
    res.relres   = rr;
    res.iters    = it;
    res.resvec   = rv;
    res.true_res = norm(b - K * res.x) / max(norm(b), eps);
    res.W        = getW();
end

%==========================================================================
function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
