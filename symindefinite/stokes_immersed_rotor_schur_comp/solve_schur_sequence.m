function Astat = solve_schur_sequence(cfg, params, save_dir)
%SOLVE_SCHUR_SEQUENCE  PCG benchmark over the Schur-complement sequence.
%   ASTAT = SOLVE_SCHUR_SEQUENCE(CFG, PARAMS, SAVE_DIR)
%
%   Runs the immersed-rotor time sequence, forms the (p,lambda) Schur
%   complement explicitly at each step, and solves it with every registered
%   arm:
%
%     pcg_unprec    no preconditioner -- the floor
%     chol          exact dense Cholesky of S_1, built ONCE and RECYCLED for
%                   every later step.  Its degradation is what makes operator
%                   drift visible at all, and it is the baseline every other arm
%                   is judged against.
%     registry      deflation on S itself, one arm per V-building operation
%                   (params.standalone_variants)
%
%   Every arm uses MATLAB pcg with the SAME tolerance, the SAME warm start and
%   the SAME RHS scaling, all hoisted above every preconditioner build, so no
%   arm can benefit from a different starting point.  Cost is reported as
%   iteration counts, not timings.
%
%   WHY THERE IS NO ichol AND NO AMG.  S is DENSE -- it contains Avel^{-1} --
%   so neither can take it: `ichol(S)` errors outright, and forcing the true
%   pattern through would just be a complete Cholesky.  Both used to run on a
%   sparse BFBt proxy, i.e. on a DIFFERENT MATRIX, which is not a defensible
%   baseline for this operator.  For a dense SPD system the honest preconditioner
%   is a dense Cholesky, and the only real question is how long one factorization
%   can be reused -- which is exactly what the `chol` arm measures.
%
%   THE DEFLATION SCHEME.  The preconditioner is built on S DIRECTLY,
%
%       P = (I - V V') + tau * V (V' S V)^-1 V'
%
%   via src.precond.deflation_P_apply, first power -- no squaring and no square
%   root: those exist only to make the coarse matrix definite when the operator
%   is indefinite, which S is not.  P is symmetric positive definite for tau > 0,
%   so it is a valid pcg preconditioner on its own; there is no split factor and
%   no B = L^-T P L^-1 composition.  Every captured mode of P*S sits at tau.
%
%   RECYCLING, AND WHY IT NEEDS NO TRANSPORT.  params.DEFLAT_PREC_REFRESH is
%   huge, so V is built once at step 1 and reused VERBATIM at every later step.
%   Because no inner factor is involved, V lives in the PHYSICAL coordinates of
%   S -- there are no factor coordinates for it to drift out of, so unlike the
%   indefinite sibling no coordinate map is needed.  What does change is the
%   operator: S(t_n) moves away from S_1, and the recycled V goes stale with it.
%   That is the effect being measured.
%
%   FAIRNESS.  Both registry arms build V from the exact step-1 inverse of S
%   (the frozen Cholesky the `chol` arm already needs), and differ ONLY in how
%   they extract the subspace from it: a full eigensolve versus a Gaussian
%   sketch.  That is the ablation -- neither is a cheaper-information arm, so
%   `deflate_exact` is the quality reference for `deflate_gaussian`, not a
%   different class of method.  All arms share one params.tau.
%
%   See also: schur_step_operator, build_sketch_V, run_schur_recycle.

    import src.precond.*

    if nargin < 3, save_dir = ''; end

    ctx    = schur_context_init(cfg, params);
    nsteps = params.Tstep - 1;
    if isfield(params, 'max_steps') && ~isempty(params.max_steps)
        % Shorten the RUN without shortening Tmax -- Tstep sets the rotor's
        % angular velocity, so trimming it would change the geometry instead of
        % just doing fewer solves.
        nsteps = min(nsteps, params.max_steps);
    end
    nU = ctx.nU;

    tol   = params.SOLVER_TOL;
    maxit = params.SOLVER_MAXIT;

    [keys, labels, variants] = local_registry(params);

    Astat = local_prealloc(keys, labels, nsteps);
    Astat.case_name = cfg.case_name;
    Astat.geometry  = cfg.geometry;
    Astat.dt        = params.dt;

    % --- State carried across the sequence ----------------------------------
    invApply = [];  chol_nS = -1;  Rfrozen = [];
    S_first  = [];  S_prev = [];   C_prev = [];  y_prev = [];  Omega = [];
    tau_eff  = [];

    rs        = struct();
    rs.V_all  = cell(numel(variants), 1);
    rs.Vexact = [];

    u_prev = zeros(nU, 1);

    for n = 1:nsteps
        tcur = n * params.dt;
        st   = schur_step_operator(ctx, tcur, u_prev);

        S   = st.S;
        rhs = st.rhs_S;
        nS  = size(S, 1);

        Astat.nC(n) = st.nC;
        Astat.nS(n) = nS;

        % --- Warm start + scaling, HOISTED above every build ----------------
        if ~isempty(y_prev) && numel(y_prev) == nS
            x0 = y_prev;
        else
            x0 = zeros(nS, 1);
        end
        rhs_norm = norm(rhs);
        if rhs_norm == 0, rhs_norm = 1; end
        rhs_scaled = rhs / rhs_norm;
        x0_scaled  = x0 / rhs_norm;

        % --- Ground truth (also advances the state) -------------------------
        x_ref = st.K \ st.b;
        Astat.backslash_relres(n) = norm(st.K * x_ref - st.b) / max(norm(st.b), eps);
        y_ref      = x_ref(nU + 1 : end);
        y_ref_keep = y_ref(st.keep);

        x_schur = st.recover(S \ rhs);
        Astat.vel_recovery_err(n) = norm(x_schur - x_ref) / max(norm(x_ref), eps);

        % --- Spectrum / conditioning ----------------------------------------
        if params.COMPUTE_SPECTRUM
            ev = eig(S);
            Astat.lambda_min(n) = min(ev);
            Astat.lambda_max(n) = max(ev);
            Astat.kappa(n)      = max(ev) / max(min(ev), eps);
        end

        % --- Drift diagnostics ----------------------------------------------
        if isempty(S_first)
            S_first = S;
            Omega   = randn(nS, 10) / sqrt(10);
        end
        if ~isempty(S_prev) && size(S_prev, 1) == nS
            Astat.ReldiffF(n) = norm(S / norm(S, 'fro') - ...
                                     S_prev / norm(S_prev, 'fro'), 'fro');
        end
        if size(S_first, 1) == nS
            Astat.RelInitdiffF(n) = norm(S / norm(S, 'fro') - ...
                                         S_first / norm(S_first, 'fro'), 'fro');
        end
        if ~isempty(C_prev) && isequal(size(C_prev), size(st.C))
            Astat.coupling_change(n) = norm(st.C - C_prev, 'fro') / ...
                                       max(norm(C_prev, 'fro'), eps);
        end

        % ================= BASELINE ARMS ===================================
        if ~params.skip_unprecond
            [xs, fl, rr, it] = pcg(S, rhs_scaled, tol, maxit, [], [], x0_scaled);
            Astat = local_record(Astat, 'pcg_unprec', n, xs, fl, rr, it, ...
                                 rhs_norm, y_ref_keep);
        end

        % --- chol, FROZEN at step 1 (the recycled exact inverse) ------------
        if isempty(invApply) || chol_nS ~= nS
            [Rf, cflag] = chol(S, 'lower');
            if cflag ~= 0
                error('solve_schur_sequence:notSPD', ...
                      'chol(S) failed at step %d -- S is not SPD.', n);
            end
            Rfrozen  = Rf;
            invApply = @(r) Rf' \ (Rf \ r);
            chol_nS  = nS;
            Astat.chol_built_step(end+1) = n;
        end
        [xs, fl, rr, it] = pcg(S, rhs_scaled, tol, maxit, invApply, [], x0_scaled);
        Astat = local_record(Astat, 'chol', n, xs, fl, rr, it, rhs_norm, y_ref_keep);

        % --- staleness of the frozen inverse --------------------------------
        if size(Omega, 1) == nS
            Aref = S \ Omega;
            Astat.InvRelDiff(n) = norm(invApply(Omega) - Aref, 'fro') / ...
                                  max(norm(Aref, 'fro'), eps);
            if ~isempty(rs.Vexact) && size(rs.Vexact, 1) == nS
                Astat.LowRankInvRelDiff(n) = ...
                    norm(S \ (rs.Vexact * (rs.Vexact' * Omega) - Omega), 'fro') / ...
                    max(norm(Aref, 'fro'), eps);
            end
        end

        % ================= REGISTRY ARMS ===================================
        if ~isempty(variants)
            if isempty(tau_eff)
                tau_eff   = local_resolve_tau(params, Astat, S, n);
                Astat.tau = tau_eff;
            end
            [rs, Astat] = local_registry_step(S, rhs_scaled, x0_scaled, ...
                variants, rs, params, n, Astat, rhs_norm, y_ref_keep, ...
                tol, maxit, Rfrozen, tau_eff);
        end

        % --- advance ---------------------------------------------------------
        S_prev = S;
        C_prev = st.C;
        y_prev = y_ref_keep;
        u_prev = x_ref(1:nU);

        if mod(n, 10) == 0 || n == 1
            fprintf('    step %3d/%3d  nS=%d nC=%d | chol %3d', ...
                    n, nsteps, nS, st.nC, Astat.solver_its.chol(n));
            for vi = 1:numel(variants)
                fprintf('  %s %3d', variants(vi).name, ...
                        Astat.solver_its.(variants(vi).name)(n));
            end
            fprintf('\n');
        end
    end

    Astat.nsteps = nsteps;
    if ~isempty(save_dir) && ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
end

%==========================================================================
function tau = local_resolve_tau(params, Astat, S, n)
%LOCAL_RESOLVE_TAU  Fix tau once, at the first step that needs it.
%   Deflation relocates every captured mode of P*S to exactly tau, so tau
%   belongs at the top of the retained spectrum: tau = lambda_max(S_1).  Put it
%   lower and the captured modes are merely moved, widening the effective
%   spectrum instead of narrowing it.
    if isfield(params, 'tau') && ~isempty(params.tau)
        tau = params.tau;
        return;
    end
    if isfinite(Astat.lambda_max(n))
        tau = Astat.lambda_max(n);          % free: the spectrum was computed
    else
        tau = eigs(S, 1, 'largestabs', 'Tolerance', 1e-8, ...
                   'IsSymmetricDefinite', true);
    end
    fprintf('    [tau] auto-set to lambda_max(S_%d) = %.4e\n', n, tau);
end

%==========================================================================
function [rs, Astat] = local_registry_step(S, rhs_scaled, x0_scaled, ...
        variants, rs, params, n, Astat, rhs_norm, y_ref, tol, maxit, ...
        Rfrozen, tau)
%LOCAL_REGISTRY_STEP  Build (once) and apply the deflation arms.
    import src.precond.*

    nS = size(S, 1);
    nv = numel(variants);

    % --- Build every coarse space (once; DEFLAT_PREC_REFRESH is huge) -------
    % Both operations read the SAME operator -- S itself.  'eig' takes its
    % smallest eigenvectors, 'gaussian' sketches its inverse.  Rfrozen is the
    % step-1 Cholesky of S the `chol` arm already built, so no extra
    % factorization is performed here.
    if mod(n - 1, params.DEFLAT_PREC_REFRESH) == 0
        % eigs convention: the handle applies S^-1, so 'smallestabs' returns the
        % SMALLEST eigenvalues of S itself.
        Sinv = @(x) Rfrozen' \ (Rfrozen \ x);
        k    = min(params.sm_eig, nS - 2);

        for vi = 1:nv
            switch variants(vi).source
                case 'eig'
                    [Vs, Ds] = eigs(Sinv, nS, k, 'smallestabs', ...
                                    'Tolerance', 1e-8, 'MaxIterations', 5000, ...
                                    'IsFunctionSymmetric', true);
                    [~, ord] = sort(real(diag(Ds)), 'ascend');
                    rs.Vexact    = orth(real(Vs(:, ord)));
                    rs.V_all{vi} = rs.Vexact;

                case 'gaussian'
                    rs.V_all{vi} = build_sketch_V(Sinv, nS, k, params.q);

                otherwise
                    error('solve_schur_sequence:badSource', ...
                          'Unsupported variant source ''%s''.', variants(vi).source);
            end

            % orth() can drop numerically dependent sketch columns, so the
            % REALIZED width is recorded rather than assumed to be sm_eig.
            Astat.deflat_dim.(variants(vi).name) = size(rs.V_all{vi}, 2);
        end
    end

    % --- Solve every registry arm -------------------------------------------
    for vi = 1:nv
        key = variants(vi).name;
        V   = rs.V_all{vi};
        if isempty(V)
            Papply = @(x) x;                  % no basis => plain PCG
        else
            % P built on S DIRECTLY (first power) -- S is SPD, so the
            % squared/square-root form of the indefinite sibling must not be used.
            Papply = deflation_P_apply(V, S, tau, [], 0);
        end
        [xs, fl, rr, it] = pcg(S, rhs_scaled, tol, maxit, Papply, [], x0_scaled);
        Astat = local_record(Astat, key, n, xs, fl, rr, it, rhs_norm, y_ref);
    end
end

%==========================================================================
function [keys, labels, variants] = local_registry(params)
%LOCAL_REGISTRY  Arm keys/labels in reporting order.
    keys   = {};
    labels = {};
    if ~params.skip_unprecond
        keys{end+1}   = 'pcg_unprec';
        labels{end+1} = 'PCG (unpreconditioned)';
    end
    keys{end+1}   = 'chol';
    labels{end+1} = 'PCG (exact chol of S_1, frozen) [BASELINE]';

    variants = struct('name', {}, 'source', {});
    if isfield(params, 'standalone_variants') && ~isempty(params.standalone_variants)
        variants = params.standalone_variants;
        for i = 1:numel(variants)
            keys{end+1}   = variants(i).name;                       %#ok<AGROW>
            labels{end+1} = local_variant_label(variants(i), params); %#ok<AGROW>
        end
    end
end

%==========================================================================
function lab = local_variant_label(v, params)
    switch v.source
        case 'eig'
            lab = sprintf('PCG (deflation on S, exact V, m=%d) [REFERENCE]', ...
                          params.sm_eig);
        case 'gaussian'
            lab = sprintf('PCG (deflation on S, Gaussian sketch of S^{-1}, m=%d, q=%d)', ...
                          params.sm_eig, params.q);
        otherwise
            lab = v.name;
    end
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
        Astat.solver_its.(k)    = nan(nsteps, 1);
        Astat.solver_flag.(k)   = nan(nsteps, 1);
        Astat.solver_relres.(k) = nan(nsteps, 1);
        Astat.solver_err.(k)    = nan(nsteps, 1);
    end
    Astat.backslash_relres  = nan(nsteps, 1);
    Astat.vel_recovery_err  = nan(nsteps, 1);
    Astat.ReldiffF          = nan(nsteps, 1);
    Astat.RelInitdiffF      = nan(nsteps, 1);
    Astat.InvRelDiff        = nan(nsteps, 1);
    Astat.LowRankInvRelDiff = nan(nsteps, 1);
    Astat.coupling_change   = nan(nsteps, 1);
    Astat.lambda_min        = nan(nsteps, 1);
    Astat.lambda_max        = nan(nsteps, 1);
    Astat.kappa             = nan(nsteps, 1);
    Astat.nC                = nan(nsteps, 1);
    Astat.nS                = nan(nsteps, 1);
    % the tau actually used (params.tau, or lambda_max(S_1) when auto-resolved)
    Astat.tau               = NaN;
    % realized coarse-space width per registry arm; `orth` may drop numerically
    % dependent sketch columns, so this is measured, never assumed to be sm_eig
    Astat.deflat_dim        = struct();
    Astat.chol_built_step   = [];
end

%==========================================================================
function Astat = local_record(Astat, key, n, x_scaled, fl, rr, it, rhs_norm, y_ref)
%LOCAL_RECORD  Store one arm's result, un-scaling the iterate first.
    Astat.solver_its.(key)(n)    = it;
    Astat.solver_flag.(key)(n)   = fl;
    Astat.solver_relres.(key)(n) = rr;
    x = x_scaled * rhs_norm;
    Astat.solver_err.(key)(n) = norm(x - y_ref) / max(norm(y_ref), eps);
end
