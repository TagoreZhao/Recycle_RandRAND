function Astat = solve_woodbury_sequence(case_name, params)
%SOLVE_WOODBURY_SEQUENCE  Direct-solve benchmark over one immersed-rotor sequence.
%   ASTAT = SOLVE_WOODBURY_SEQUENCE(CASE_NAME, PARAMS)
%
%   Runs the immersed-rotor KKT time sequence and solves every step three ways:
%
%     woodbury  frozen factorization of A_1 = K_1 plus the rank-2nC capacitance
%               correction -- THE METHOD
%     frozen    K_1^{-1} b_n with no correction at all -- the control that shows
%               what the Woodbury term buys
%     fresh     K_n \ b_n, refactorized every step -- the accuracy reference and
%               the wall-clock baseline the method has to beat
%
%   ALL THREE ARE DIRECT SOLVES.  There is no Krylov layer and no iterative
%   refinement, so cost is wall clock and quality is forward error -- iteration
%   counts, the usual currency of the sibling benchmarks, do not exist here.
%
%   NO ARM ADVANCES THE STATE.  u_prev is advanced with the backslash solution
%   inside build_stokes_sequence, exactly as the parent benchmark does, so all
%   three arms see an identical RHS sequence and cannot drift apart.  S.xref{n}
%   is the ground truth; the `fresh` arm's error against it (~1e-15) is the
%   self-check that validates the ground truth itself.
%
%   WHY THERE IS NO CFG STRUCT.  The sibling engines take (cfg, params, save_dir)
%   because they own mesh generation and the motion closure.  Here
%   build_stokes_sequence owns all of that -- and caches it, and re-asserts the
%   rank-2nC identity while doing so -- so the case name plus params is the whole
%   input.
%
%   TIMING FAIRNESS.  Three deliberate choices, all of which cut against the
%   method rather than for it:
%     * every solve is repeated params.TIME_REPEATS times and the MINIMUM taken,
%       because the mean and max measure the OS scheduler as much as the algorithm;
%     * `fresh` is timed as factorization AND solve together, since refactorizing
%       is exactly the cost being avoided;
%     * materializing K_n (seq_K) is EXCLUDED from every arm's clock even though
%       only `fresh` actually needs it -- a production Woodbury solver never forms
%       K_n, so charging it to nobody understates the method's advantage.
%
%   See also: woodbury_solve, woodbury_context_init, run_woodbury_benchmark.

    if nargin < 2 || isempty(params), params = make_woodbury_params(); end

    keys   = {'woodbury', 'frozen', 'fresh'};
    labels = { ...
        'Woodbury update on frozen A_1^{-1} (rank-2nC) [METHOD]', ...
        'Frozen A_1^{-1}, no correction [CONTROL]', ...
        'Fresh LDL of K_n every step [REFERENCE]'};

    % --- Sequence (built or loaded; the low-rank identity is re-asserted) ---
    nsteps_req = params.Tstep - 1;
    if isfield(params, 'max_steps') && ~isempty(params.max_steps)
        nsteps_req = min(nsteps_req, params.max_steps);
    end
    sopts = struct('case_name', case_name, 'h0', params.h0, 'dt', params.dt, ...
                   'Tstep', params.Tstep, 'nsteps', nsteps_req, ...
                   'verify', params.verify_lowrank, ...
                   'use_cache', params.use_cache, 'quiet', false);
    S      = build_stokes_sequence(sopts);
    nsteps = S.nsteps;

    % --- Warm up before anything is timed -----------------------------------
    % The sparse LDL and triangular-solve paths are lazily loaded, so the FIRST
    % factorization in a session costs ~3x a warm one (measured 96 ms vs 30 ms at
    % h0 = 0.05).  Since t_setup is paid exactly once, a cold measurement would
    % inflate it -- and with it the break-even step -- by a factor of three.  This
    % throwaway pair is untimed and its results discarded.
    Kwarm = seq_K(S, 1);
    [~, ~, ~, ~] = ldl(Kwarm, 'vector');
    warmup_x = Kwarm \ S.b{1};                          %#ok<NASGU>
    clear Kwarm warmup_x;

    % --- The one factorization this study is allowed ------------------------
    ctx = woodbury_context_init(S);
    fprintf(['    [ctx] ldl(K_1): nnz(K)=%d nnz(L)=%d fill=%.2f in %.3f s ' ...
             '(+%d backsolves for YSel, %.3f s total)\n'], ...
            ctx.nnzK1, ctx.nnzL, ctx.fill_ratio, ctx.t_factor, ...
            ctx.n_backsolves_setup, ctx.t_setup);

    Astat = local_prealloc(keys, labels, nsteps);
    Astat.case_name       = case_name;
    Astat.h0              = params.h0;
    Astat.dt              = params.dt;
    Astat.Tstep           = params.Tstep;
    Astat.nsteps          = nsteps;
    Astat.ntot            = S.n;
    Astat.nC_const        = S.nC;
    Astat.ref             = ctx.ref;
    Astat.nnzK1           = ctx.nnzK1;
    Astat.nnzL            = ctx.nnzL;
    Astat.fill_ratio      = ctx.fill_ratio;
    Astat.t_factor        = ctx.t_factor;
    Astat.t_setup         = ctx.t_setup;
    Astat.n_backsolves_setup = ctx.n_backsolves_setup;
    Astat.TIME_REPEATS    = params.TIME_REPEATS;
    Astat.coupling_change = S.coupling_change(1:nsteps);

    nrep = max(1, params.TIME_REPEATS);

    for n = 1:nsteps
        b    = S.b{n};
        xref = S.xref{n};
        Kn   = seq_K(S, n);                     % excluded from every arm's clock
        bnrm = max(norm(b), eps);

        Astat.nC(n)   = S.nC;
        Astat.nsys(n) = size(Kn, 1);
        Astat.backslash_relres(n) = norm(Kn * xref - b) / bnrm;

        % --- woodbury ------------------------------------------------------
        t_tot = inf;  t_net = inf;  t_dg = inf;
        for r = 1:nrep
            [xw, info] = woodbury_solve(ctx, S, n, b);
            t_tot = min(t_tot, info.t_solve);
            t_net = min(t_net, info.t_net);
            t_dg  = min(t_dg,  info.t_diag);
        end
        Astat.t_woodbury(n)      = t_tot;
        Astat.t_woodbury_net(n)  = t_net;
        Astat.t_woodbury_diag(n) = t_dg;
        Astat = local_record(Astat, 'woodbury', n, xw, xref, Kn, b, bnrm);

        Astat.cap_cond(n)   = info.cap_cond;
        Astat.cap_smin(n)   = info.cap_smin;
        Astat.cap_smax(n)   = info.cap_smax;
        Astat.cap_rcond(n)  = info.cap_rcond;
        Astat.cap_symres(n) = info.cap_symres;
        Astat.dC_normF(n)       = info.dC_normF;
        Astat.dC_rel(n)         = info.dC_rel;
        Astat.dC_is_zero(n)     = info.dC_is_zero;
        Astat.correction_rel(n) = info.correction_rel;

        % --- frozen (no correction) ----------------------------------------
        t_best = inf;
        for r = 1:nrep
            t1 = tic;  xf = woodbury_apply_ref(ctx, b);
            t_best = min(t_best, toc(t1));
        end
        Astat.t_frozen(n) = t_best;
        Astat = local_record(Astat, 'frozen', n, xf, xref, Kn, b, bnrm);

        % --- fresh (refactorize: factor AND solve) --------------------------
        % Kn\b, not decomposition(Kn)\b: backslash is MATLAB's fastest
        % from-scratch path for a sparse symmetric system (measured 17.8 ms vs
        % 21.2 ms at h0=0.05), so the method faces the toughest honest opponent.
        t_best = inf;
        for r = 1:nrep
            t1 = tic;  xfr = Kn \ b;  t_best = min(t_best, toc(t1));
        end
        Astat.t_fresh(n) = t_best;
        Astat = local_record(Astat, 'fresh', n, xfr, xref, Kn, b, bnrm);

        % --- optional spectrum of K_n --------------------------------------
        if params.COMPUTE_SPECTRUM
            [Astat.lambda_absmin(n), Astat.lambda_absmax(n), Astat.kappa(n)] = ...
                local_spectrum(Kn, params.SPECTRUM_TOL);
        end

        if n == 1 || mod(n, 10) == 0 || n == nsteps
            fprintf(['    step %3d/%3d  dC_rel %5.3f  cond(Cap) %8.2e | ' ...
                     'err  wood %8.2e  froz %8.2e  fresh %8.2e | ' ...
                     't wood %6.1f ms  fresh %6.1f ms\n'], ...
                    n, nsteps, Astat.dC_rel(n), Astat.cap_cond(n), ...
                    Astat.solver_err.woodbury(n), Astat.solver_err.frozen(n), ...
                    Astat.solver_err.fresh(n), ...
                    1e3*Astat.t_woodbury_net(n), 1e3*Astat.t_fresh(n));
        end
    end

    % --- Cumulative cost and the break-even step ---------------------------
    % The method pays for one factorization up front, so "is it cheaper?" is only
    % meaningful cumulatively.  break_even_step is the first step at which the
    % running total (setup INCLUDED) drops below the refactorize-every-step total.
    cum_w = ctx.t_setup + cumsum(Astat.t_woodbury_net);
    cum_f = cumsum(Astat.t_fresh);
    Astat.cum_woodbury = cum_w;
    Astat.cum_fresh    = cum_f;
    idx = find(cum_w < cum_f, 1, 'first');
    if isempty(idx)
        Astat.break_even_step = NaN;         % never pays off over this many steps
    else
        Astat.break_even_step = idx;
    end
end

%==========================================================================
function [labsmin, labsmax, kappa] = local_spectrum(Kn, tol)
%LOCAL_SPECTRUM  Extreme |eigenvalues| of a symmetric INDEFINITE K_n.
%   kappa = max|lambda| / min|lambda|; the signed extremes are not the relevant
%   pair here because K_n straddles zero by construction.
%
%   eigs SILENTLY IGNORES an options struct -- these must stay name-value pairs.
    labsmax = eigs(Kn, 1, 'largestabs',  'Tolerance', tol, 'MaxIterations', 5000);
    labsmin = eigs(Kn, 1, 'smallestabs', 'Tolerance', tol, 'MaxIterations', 5000);
    labsmax = abs(labsmax);
    labsmin = abs(labsmin);
    kappa   = labsmax / max(labsmin, eps);
end

%==========================================================================
function Astat = local_prealloc(keys, labels, nsteps)
    Astat.solver_keys   = keys(:);
    Astat.solver_labels = labels(:);
    Astat.solver_its    = struct();
    Astat.solver_flag   = struct();
    Astat.solver_relres = struct();
    Astat.solver_err    = struct();
    for i = 1:numel(keys)
        k = keys{i};
        % solver_its/flag are kept at their family-contract shape so the CSV
        % round-trip matches the siblings.  These are DIRECT solves: its == 1
        % (one application) and flag == 0 (converged) at every step, by
        % definition rather than by measurement.
        Astat.solver_its.(k)    = ones(nsteps, 1);
        Astat.solver_flag.(k)   = zeros(nsteps, 1);
        Astat.solver_relres.(k) = nan(nsteps, 1);
        Astat.solver_err.(k)    = nan(nsteps, 1);
    end
    z = @() nan(nsteps, 1);
    Astat.backslash_relres  = z();
    Astat.cap_cond          = z();
    Astat.cap_smin          = z();
    Astat.cap_smax          = z();
    Astat.cap_rcond         = z();
    Astat.cap_symres        = z();
    Astat.dC_normF          = z();
    Astat.dC_rel            = z();
    Astat.dC_is_zero        = false(nsteps, 1);
    % ||x_woodbury - K_1^{-1}b|| / ||x_woodbury||: how far the rank-2nC term
    % actually moves the iterate.  Exactly 0 when dC == 0.
    Astat.correction_rel    = z();
    Astat.t_woodbury        = z();
    Astat.t_woodbury_net    = z();
    Astat.t_woodbury_diag   = z();
    Astat.t_frozen          = z();
    Astat.t_fresh           = z();
    Astat.nC                = z();
    Astat.nsys              = z();
    Astat.lambda_absmin     = z();
    Astat.lambda_absmax     = z();
    Astat.kappa             = z();
end

%==========================================================================
function Astat = local_record(Astat, key, n, x, xref, Kn, b, bnrm)
%LOCAL_RECORD  Forward error against the ground truth and the true residual.
    Astat.solver_err.(key)(n)    = norm(x - xref) / max(norm(xref), eps);
    Astat.solver_relres.(key)(n) = norm(Kn * x - b) / bnrm;
end
