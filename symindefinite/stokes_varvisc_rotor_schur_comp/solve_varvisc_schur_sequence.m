function Astat = solve_varvisc_schur_sequence(cfg, params, save_dir)
%SOLVE_VARVISC_SCHUR_SEQUENCE  PCG benchmark on dense Schur complements.
%   The current A_n factorization is used only to construct the exact S_n and
%   recover velocity. Solver arm `chol` separately freezes chol(S_1). The
%   Gaussian small-, large- and both-tail bases have independent refresh
%   cadences and are otherwise reused in physical coordinates.

    import src.precond.*
    if nargin < 3, save_dir = ''; end

    refresh_intervals = local_refresh_intervals(params);
    ctx = varvisc_schur_context_init(cfg, params);
    nsteps = params.Tstep - 1;
    if isfield(params,'max_steps') && ~isempty(params.max_steps)
        nsteps = min(nsteps, params.max_steps);
    end
    [keys,labels,variants] = local_registry(params);
    Astat = local_prealloc(keys,labels,nsteps);
    Astat.deflat_requested_dim = struct( ...
        'small',params.sm_eig,'large',params.lg_eig, ...
        'both',params.sm_eig+params.lg_eig);
    Astat.case_name = cfg.case_name;
    Astat.geometry = cfg.geometry;
    Astat.dt = params.dt;

    u_prev = zeros(ctx.nU,1);
    if isfield(cfg,'u0') && ~isempty(cfg.u0), u_prev = cfg.u0; end
    y_prev = [];
    S_first = []; S_prev = []; A_prev = []; D_prev = []; C_prev = [];
    Rfrozen = []; invApply = []; nS_frozen = -1; Omega = [];
    V_all = cell(numel(variants),1);
    tau_all = repmat(struct('small',NaN,'large',NaN, ...
                            'both_stage1',NaN,'both_stage2',NaN), ...
                     numel(variants),1);

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
        [Rcurrent,Astat.chol_flag(n)] = chol(S,'lower');

        need_deflation_spectrum = ~isempty(variants);
        if params.COMPUTE_SPECTRUM || need_deflation_spectrum
            ev = sort(real(eig(S)),'ascend');
            Astat.lambda_min(n) = ev(1);
            Astat.lambda_max(n) = ev(end);
            Astat.kappa(n) = ev(end)/ev(1);
        else
            ev = [];
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
            Rfrozen = Rcurrent;
            if Astat.chol_flag(n) ~= 0
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
            for vi = 1:numel(variants)
                key = variants(vi).name;
                tail = variants(vi).tail;
                refresh_basis = local_refresh_due( ...
                    n,refresh_intervals.(tail));
                if refresh_basis
                    if Astat.chol_flag(n) ~= 0
                        error('solve_varvisc_schur_sequence:notSPD', ...
                              ['Cannot refresh %s-tail deflation from ', ...
                               'non-SPD S at step %d.'],tail,n);
                    end
                    [V_all{vi},tau_all(vi),dims] = local_refresh_basis( ...
                        S,Rcurrent,ev,params,tail);
                    Astat.deflat_dim.(key) = size(V_all{vi},2);
                    Astat.deflat_tail_dim.(tail) = size(V_all{vi},2);
                    if strcmp(tail,'both')
                        Astat.deflat_tail_dim.both_small = dims.small;
                        Astat.deflat_tail_dim.both_large = dims.large;
                    end
                    if ~isfield(Astat.basis_built_step,key)
                        Astat.basis_built_step.(key) = [];
                    end
                    Astat.basis_built_step.(key)(end+1) = n;
                    if strcmp(tail,'small') && isnan(Astat.tau)
                        Astat.tau = tau_all(vi).small;
                    end
                end

                V = V_all{vi};
                Astat.deflat_dim_history.(key)(n) = size(V,2);
                switch tail
                    case 'small'
                        Astat.deflation_tau.small(n) = tau_all(vi).small;
                        Papply = deflation_P_apply(V,S,tau_all(vi).small,[],0);
                    case 'large'
                        Astat.deflation_tau.large(n) = tau_all(vi).large;
                        Papply = deflation_P_apply(V,S,tau_all(vi).large,[],0);
                    case 'both'
                        Astat.deflation_tau.both_stage1(n) = ...
                            tau_all(vi).both_stage1;
                        Astat.deflation_tau.both_stage2(n) = ...
                            tau_all(vi).both_stage2;
                        % Symmetric two-stage composition from the standalone
                        % Schur experiment. The same combined V is used twice.
                        P1half = deflation_Psqrt_apply( ...
                            V,S,tau_all(vi).both_stage1,'handle');
                        S1apply = @(X) P1half(S*P1half(X));
                        P2apply = deflation_P_apply( ...
                            V,S1apply,tau_all(vi).both_stage2,'handle',0);
                        Papply = @(X) P1half(P2apply(P1half(X)));
                    otherwise
                        error('solve_varvisc_schur_sequence:badTail', ...
                              'Unsupported tail selection %s.',variants(vi).tail);
                end
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
    variants = struct('name',{},'source',{},'tail',{});
    if isfield(params,'standalone_variants') && ~isempty(params.standalone_variants)
        variants = params.standalone_variants;
        if ~isfield(variants,'tail')
            [variants.tail] = deal('small');
        end
        for i = 1:numel(variants)
            if ~strcmp(variants(i).source,'gaussian')
                error('solve_varvisc_schur_sequence:badSource', ...
                      'Only Gaussian deflation bases are supported; got %s.', ...
                      variants(i).source);
            end
            keys{end+1} = variants(i).name; %#ok<AGROW>
            switch variants(i).tail
                case 'small'
                    labels{end+1} = sprintf(['PCG (deflation on S, Gaussian ' ...
                        'small-tail sketch, m=%d, q=%d)'], ...
                        params.sm_eig,params.q); %#ok<AGROW>
                case 'large'
                    labels{end+1} = sprintf(['PCG (deflation on S, Gaussian ' ...
                        'large-tail sketch, m=%d, q=%d)'], ...
                        params.lg_eig,params.q); %#ok<AGROW>
                case 'both'
                    labels{end+1} = sprintf(['PCG (two-stage deflation on S, ' ...
                        'Gaussian two-tail sketch, m=%d+%d, q=%d)'], ...
                        params.sm_eig,params.lg_eig,params.q); %#ok<AGROW>
                otherwise
                    error('solve_varvisc_schur_sequence:badTail', ...
                          'Unsupported tail selection %s.',variants(i).tail);
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
    Astat.deflation_tau = struct( ...
        'small',nan(nsteps,1),'large',nan(nsteps,1), ...
        'both_stage1',nan(nsteps,1),'both_stage2',nan(nsteps,1));
    Astat.deflat_dim = struct();
    Astat.deflat_dim_history = struct();
    for i = 1:numel(keys)
        if startsWith(keys{i},'deflate_')
            Astat.deflat_dim_history.(keys{i}) = nan(nsteps,1);
        end
    end
    Astat.deflat_requested_dim = struct();
    Astat.deflat_tail_dim = struct( ...
        'small',NaN,'large',NaN,'both',NaN, ...
        'both_small',NaN,'both_large',NaN);
    Astat.basis_built_step = struct();
    Astat.chol_built_step = [];
end

function refresh = local_refresh_intervals(params)
    names = {'small','large','both'};
    fields = {'DEFLAT_SMALL_PREC_REFRESH', ...
              'DEFLAT_LARGE_PREC_REFRESH', ...
              'DEFLAT_BOTH_PREC_REFRESH'};
    refresh = struct();
    for i = 1:numel(names)
        value = Inf;
        if isfield(params,fields{i}) && ~isempty(params.(fields{i}))
            value = params.(fields{i});
        elseif isfield(params,'DEFLAT_PREC_REFRESH') && ...
                ~isempty(params.DEFLAT_PREC_REFRESH)
            % Compatibility with parameter structs saved before the three
            % independent cadence controls were introduced.
            value = params.DEFLAT_PREC_REFRESH;
        end
        valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
            ((isinf(value) && value > 0) || ...
             (isfinite(value) && value >= 1 && value == floor(value)));
        if ~valid
            error('solve_varvisc_schur_sequence:badDeflatRefresh', ...
                  '%s must be a positive integer or Inf.',fields{i});
        end
        refresh.(names{i}) = value;
    end
end

function due = local_refresh_due(step,interval)
    due = step == 1 || ...
        (isfinite(interval) && mod(step-1,interval) == 0);
end

function [V,tau_state,dims] = local_refresh_basis( ...
        S,Rcurrent,ev,params,tail)
    nS = size(S,1);
    validateattributes(params.sm_eig,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'params.sm_eig');
    validateattributes(params.lg_eig,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'params.lg_eig');

    currentInv = @(X) Rcurrent' \ (Rcurrent \ X);
    tau_state = struct('small',NaN,'large',NaN, ...
                       'both_stage1',NaN,'both_stage2',NaN);
    dims = struct('small',0,'large',0);
    switch tail
        case 'small'
            kSmall = min(params.sm_eig,nS-1);
            V = varvisc_schur_build_sketch_V( ...
                currentInv,nS,kSmall,params.q);
            dims.small = size(V,2);
            tau_state.small = ev(end);
            if isfield(params,'tau') && ~isempty(params.tau)
                validateattributes(params.tau,{'numeric'}, ...
                    {'scalar','real','positive'},mfilename,'params.tau');
                tau_state.small = params.tau;
            end
        case 'large'
            kLarge = min(params.lg_eig,nS-1);
            V = varvisc_schur_build_sketch_V( ...
                @(X) S*X,nS,kLarge,params.q);
            dims.large = size(V,2);
            tau_state.large = ev(nS-dims.large);
        case 'both'
            [kSmall,kLarge] = local_two_tail_widths( ...
                nS,params.sm_eig,params.lg_eig);
            Vsmall = varvisc_schur_build_sketch_V( ...
                currentInv,nS,kSmall,params.q);
            Vlarge = varvisc_schur_build_sketch_V( ...
                @(X) S*X,nS,kLarge,params.q);
            dims.small = size(Vsmall,2);
            dims.large = size(Vlarge,2);
            V = orth(real([Vlarge,Vsmall]));
            if isempty(V)
                error('solve_varvisc_schur_sequence:emptyBothBasis', ...
                      'The combined two-tail sketch collapsed to rank zero.');
            end
            tau_state.both_stage1 = ev(nS-dims.large);
            tau_state.both_stage2 = sqrt( ...
                ev(dims.small+1)*ev(nS-dims.large));
        otherwise
            error('solve_varvisc_schur_sequence:badTail', ...
                  'Unsupported tail selection %s.',tail);
    end
end

function [kSmall,kLarge] = local_two_tail_widths(nS,smRequested,lgRequested)
    available = nS-1;
    if available < 2
        error('solve_varvisc_schur_sequence:schurTooSmall', ...
              'Two-tail deflation requires at least a 3-by-3 Schur matrix.');
    end
    kSmall = min(smRequested,available);
    kLarge = min(lgRequested,available);
    if kSmall+kLarge <= available
        return
    end

    % Proportionally reduce both requested widths while guaranteeing at least
    % one column for each tail and one unresolved Schur direction.
    totalRequested = smRequested+lgRequested;
    kSmall = max(1,min(smRequested,floor(available*smRequested/totalRequested)));
    kLarge = max(1,min(lgRequested,available-kSmall));
    remaining = available-kSmall-kLarge;
    addSmall = min(remaining,smRequested-kSmall);
    kSmall = kSmall+addSmall;
    remaining = available-kSmall-kLarge;
    kLarge = kLarge+min(remaining,lgRequested-kLarge);
end

function Astat = local_record(Astat,key,n,xscaled,fl,rr,it,rhsnorm,yref)
    Astat.solver_its.(key)(n) = it;
    Astat.solver_flag.(key)(n) = fl;
    Astat.solver_relres.(key)(n) = rr;
    x = xscaled*rhsnorm;
    Astat.solver_err.(key)(n) = norm(x-yref)/max(norm(yref),eps);
end
