function Astat = solve_varvisc_schur_sequence(cfg, params, save_dir)
%SOLVE_VARVISC_SCHUR_SEQUENCE  PCG benchmark on dense Schur complements.
%   The current A_n factorization is used only to construct the exact S_n and
%   recover velocity. Solver arm `chol` separately freezes chol(S_1), while
%   both deflation arms freeze their step-1 physical-coordinate basis.

    import src.precond.*
    if nargin < 3, save_dir = ''; end

    ctx = varvisc_schur_context_init(cfg, params);
    nsteps = params.Tstep - 1;
    if isfield(params,'max_steps') && ~isempty(params.max_steps)
        nsteps = min(nsteps, params.max_steps);
    end
    [keys,labels,variants] = local_registry(params);
    Astat = local_prealloc(keys,labels,nsteps);
    Astat.case_name = cfg.case_name;
    Astat.geometry = cfg.geometry;
    Astat.dt = params.dt;

    u_prev = zeros(ctx.nU,1);
    if isfield(cfg,'u0') && ~isempty(cfg.u0), u_prev = cfg.u0; end
    y_prev = [];
    S_first = []; S_prev = []; A_prev = []; D_prev = []; C_prev = [];
    Rfrozen = []; invApply = []; nS_frozen = -1; Omega = [];
    tau_eff = [];
    V_all = cell(numel(variants),1);

    for n = 1:nsteps
        tcur = n * params.dt;
        st = varvisc_schur_step_operator(ctx,tcur,u_prev);
        S = st.S; rhs = st.rhs_S; nS = st.nS;
        Astat.nC(n) = st.nC;
        Astat.nS(n) = nS;
        Astat.nu_contrast(n) = max(st.nu_e) / min(st.nu_e);

        if isempty(S_first)
            S_first = S;
            Omega = randn(nS,10) / sqrt(10);
        elseif size(S_first,1) ~= nS
            error('solve_varvisc_schur_sequence:changingSize', ...
                  'Schur dimension changed from %d to %d at step %d.', ...
                  size(S_first,1),nS,n);
        end

        if ~isempty(S_prev)
            Astat.ReldiffF(n) = local_normalized_change(S,S_prev);
            Astat.A_change(n) = local_normalized_change(st.A_bc,A_prev);
            Astat.D_change(n) = local_normalized_change(st.D,D_prev);
            Spp = S(1:ctx.nP-1,1:ctx.nP-1);
            SppPrev = S_prev(1:ctx.nP-1,1:ctx.nP-1);
            Astat.pressure_schur_change(n) = local_normalized_change(Spp,SppPrev);
        end
        Astat.RelInitdiffF(n) = local_normalized_change(S,S_first);
        if ~isempty(C_prev) && isequal(size(C_prev),size(st.C))
            Astat.coupling_change(n) = norm(st.C-C_prev,'fro') / ...
                max(norm(C_prev,'fro'),eps);
        end

        % Full KKT ground truth is the only state-advancing solution.
        x_ref = st.K \ st.b;
        Astat.backslash_relres(n) = norm(st.K*x_ref-st.b) / max(norm(st.b),eps);
        y_ref = x_ref(ctx.nU+1:end);
        y_ref_keep = y_ref(st.keep);
        x_schur = st.recover(S \ rhs);
        Astat.vel_recovery_err(n) = norm(x_schur-x_ref) / max(norm(x_ref),eps);
        Astat.symmetry_res(n) = norm(S-S','fro') / max(norm(S,'fro'),eps);
        [~,Astat.chol_flag(n)] = chol(S);

        need_eigvecs = n == 1 && any(strcmp({variants.source},'eig'));
        if params.COMPUTE_SPECTRUM || need_eigvecs
            if need_eigvecs
                [Ue,De] = eig(S,'vector');
                [ev,ord] = sort(real(De),'ascend');
                Ue = real(Ue(:,ord));
            else
                ev = sort(real(eig(S)),'ascend');
                Ue = [];
            end
            Astat.lambda_min(n) = ev(1);
            Astat.lambda_max(n) = ev(end);
            Astat.kappa(n) = ev(end)/ev(1);
        else
            Ue = [];
        end

        if isempty(y_prev), x0 = zeros(nS,1); else, x0 = y_prev; end
        rhs_norm = norm(rhs); if rhs_norm == 0, rhs_norm = 1; end
        rhs_scaled = rhs/rhs_norm;
        x0_scaled = x0/rhs_norm;

        if ~params.skip_unprecond
            [xs,fl,rr,it] = pcg(S,rhs_scaled,params.SOLVER_TOL, ...
                                params.SOLVER_MAXIT,[],[],x0_scaled);
            Astat = local_record(Astat,'pcg_unprec',n,xs,fl,rr,it, ...
                                 rhs_norm,y_ref_keep);
        end

        if isempty(Rfrozen)
            [Rfrozen,cflag] = chol(S,'lower');
            if cflag ~= 0
                error('solve_varvisc_schur_sequence:notSPD', ...
                      'chol(S) failed at step %d.',n);
            end
            invApply = @(X) Rfrozen' \ (Rfrozen \ X);
            nS_frozen = nS;
            Astat.chol_built_step(end+1) = n;
        elseif nS ~= nS_frozen
            error('solve_varvisc_schur_sequence:changingFrozenSize', ...
                  'Cannot reuse frozen factors after a dimension change.');
        end
        [xs,fl,rr,it] = pcg(S,rhs_scaled,params.SOLVER_TOL, ...
                            params.SOLVER_MAXIT,invApply,[],x0_scaled);
        Astat = local_record(Astat,'chol',n,xs,fl,rr,it,rhs_norm,y_ref_keep);
        Aref = S \ Omega;
        Astat.InvRelDiff(n) = norm(invApply(Omega)-Aref,'fro') / ...
                              max(norm(Aref,'fro'),eps);

        if ~isempty(variants)
            if isempty(tau_eff)
                if isfield(params,'tau') && ~isempty(params.tau)
                    tau_eff = params.tau;
                elseif isfinite(Astat.lambda_max(n))
                    tau_eff = Astat.lambda_max(n);
                else
                    tau_eff = max(eig(S));
                end
                Astat.tau = tau_eff;
            end

            if n == 1
                kdim = min(params.sm_eig,nS-1);
                for vi = 1:numel(variants)
                    switch variants(vi).source
                        case 'eig'
                            if isempty(Ue)
                                [Uf,Df] = eig(S,'vector');
                                [~,ord] = sort(real(Df),'ascend');
                                Ue = real(Uf(:,ord));
                            end
                            V_all{vi} = Ue(:,1:kdim);
                        case 'gaussian'
                            V_all{vi} = varvisc_schur_build_sketch_V( ...
                                invApply,nS,kdim,params.q);
                        otherwise
                            error('solve_varvisc_schur_sequence:badSource', ...
                                  'Unsupported basis source %s.',variants(vi).source);
                    end
                    Astat.deflat_dim.(variants(vi).name) = size(V_all{vi},2);
                    Astat.basis_built_step.(variants(vi).name) = n;
                end
            end

            for vi = 1:numel(variants)
                key = variants(vi).name;
                % V is passed through exactly as constructed.  In particular,
                % the Gaussian arm's single orth(real(Y)) is the only
                % numerical-range selection; there is no post-orth truncation.
                V = V_all{vi};
                Papply = deflation_P_apply(V,S,tau_eff,[],0);
                [xs,fl,rr,it] = pcg(S,rhs_scaled,params.SOLVER_TOL, ...
                                    params.SOLVER_MAXIT,Papply,[],x0_scaled);
                Astat = local_record(Astat,key,n,xs,fl,rr,it, ...
                                     rhs_norm,y_ref_keep);
            end
        end

        S_prev = S; A_prev = st.A_bc; D_prev = st.D; C_prev = st.C;
        y_prev = y_ref_keep;
        u_prev = x_ref(1:ctx.nU);

        if n == 1 || mod(n,10) == 0
            fprintf('    step %3d/%3d nS=%d contrast=%.1f |', ...
                    n,nsteps,nS,Astat.nu_contrast(n));
            for ki = 1:numel(keys)
                fprintf(' %s %d',keys{ki},Astat.solver_its.(keys{ki})(n));
            end
            fprintf('\n');
        end
    end

    Astat.nsteps = nsteps;
    if ~isempty(save_dir) && ~exist(save_dir,'dir'), mkdir(save_dir); end
end

function value = local_normalized_change(A,B)
    value = norm(A/max(norm(A,'fro'),eps)-B/max(norm(B,'fro'),eps),'fro');
end

function [keys,labels,variants] = local_registry(params)
    keys = {}; labels = {};
    if ~params.skip_unprecond
        keys{end+1} = 'pcg_unprec';
        labels{end+1} = 'PCG (unpreconditioned)';
    end
    keys{end+1} = 'chol';
    labels{end+1} = 'PCG (exact chol of S_1, frozen) [BASELINE]';
    variants = struct('name',{},'source',{});
    if isfield(params,'standalone_variants') && ~isempty(params.standalone_variants)
        variants = params.standalone_variants;
        for i = 1:numel(variants)
            keys{end+1} = variants(i).name; %#ok<AGROW>
            switch variants(i).source
                case 'eig'
                    labels{end+1} = sprintf('PCG (deflation on S, exact V, m=%d)', ...
                                            params.sm_eig); %#ok<AGROW>
                case 'gaussian'
                    labels{end+1} = sprintf(['PCG (deflation on S, Gaussian ' ...
                        'sketch of S_1^{-1}, m=%d, q=%d)'],params.sm_eig,params.q); %#ok<AGROW>
            end
        end
    end
end

function Astat = local_prealloc(keys,labels,nsteps)
    Astat.solver_keys = keys(:); Astat.solver_labels = labels(:);
    Astat.solver_its = struct(); Astat.solver_flag = struct();
    Astat.solver_relres = struct(); Astat.solver_err = struct();
    for i = 1:numel(keys)
        key = keys{i};
        Astat.solver_its.(key) = nan(nsteps,1);
        Astat.solver_flag.(key) = nan(nsteps,1);
        Astat.solver_relres.(key) = nan(nsteps,1);
        Astat.solver_err.(key) = nan(nsteps,1);
    end
    fields = {'backslash_relres','vel_recovery_err','symmetry_res','chol_flag', ...
        'ReldiffF','RelInitdiffF','InvRelDiff','coupling_change','A_change', ...
        'D_change','pressure_schur_change','nu_contrast','lambda_min', ...
        'lambda_max','kappa','nC','nS'};
    for i = 1:numel(fields), Astat.(fields{i}) = nan(nsteps,1); end
    Astat.LowRankInvRelDiff = nan(nsteps,1);
    Astat.tau = NaN;
    Astat.deflat_dim = struct();
    Astat.basis_built_step = struct();
    Astat.chol_built_step = [];
end

function Astat = local_record(Astat,key,n,xscaled,fl,rr,it,rhsnorm,yref)
    Astat.solver_its.(key)(n) = it;
    Astat.solver_flag.(key)(n) = fl;
    Astat.solver_relres.(key)(n) = rr;
    x = xscaled*rhsnorm;
    Astat.solver_err.(key)(n) = norm(x-yref)/max(norm(yref),eps);
end
