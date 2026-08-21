function Astat = solve_varvisc_schur_sequence(cfg, params, save_dir)
%SOLVE_VARVISC_SCHUR_SEQUENCE  PCG benchmark on dense Schur complements.
%   All two-tail designs share one centrally refreshed smallest-mode basis.
%   Each design that needs a largest-mode sketch owns an independent cache.

    import src.precond.*
    if nargin < 3, save_dir = ''; end

    settings = local_settings(params);
    refresh = local_refresh_intervals(params);
    ctx = varvisc_schur_context_init(cfg,params);
    nsteps = params.Tstep-1;
    if isfield(params,'max_steps') && ~isempty(params.max_steps)
        nsteps = min(nsteps,params.max_steps);
    end
    [keys,labels,variants] = local_registry(params,settings);
    Astat = local_prealloc(keys,labels,nsteps,settings);
    Astat.case_name = cfg.case_name;
    Astat.geometry = cfg.geometry;
    Astat.dt = params.dt;

    u_prev = zeros(ctx.nU,1);
    if isfield(cfg,'u0') && ~isempty(cfg.u0), u_prev = cfg.u0; end
    y_prev = [];
    S_first = []; S_prev = []; A_prev = []; D_prev = []; C_prev = [];
    Rfrozen = []; invApply = []; nS_frozen = -1; Omega = [];
    smallState = local_empty_small_state();
    states = local_empty_arm_states();

    for step = 1:nsteps
        tcur = step*params.dt;
        st = varvisc_schur_step_operator(ctx,tcur,u_prev);
        S = st.S; rhs = st.rhs_S; nS = st.nS;
        Astat.nC(step) = st.nC;
        Astat.nS(step) = nS;
        Astat.nu_contrast(step) = max(st.nu_e)/min(st.nu_e);

        if isempty(S_first)
            S_first = S;
            Omega = randn(nS,10)/sqrt(10);
        elseif size(S_first,1) ~= nS
            error('solve_varvisc_schur_sequence:changingSize', ...
                  'Schur dimension changed from %d to %d at step %d.', ...
                  size(S_first,1),nS,step);
        end

        if ~isempty(S_prev)
            Astat.ReldiffF(step) = local_normalized_change(S,S_prev);
            Astat.A_change(step) = local_normalized_change(st.A_bc,A_prev);
            Astat.D_change(step) = local_normalized_change(st.D,D_prev);
            Spp = S(1:ctx.nP-1,1:ctx.nP-1);
            SppPrev = S_prev(1:ctx.nP-1,1:ctx.nP-1);
            Astat.pressure_schur_change(step) = ...
                local_normalized_change(Spp,SppPrev);
        end
        Astat.RelInitdiffF(step) = local_normalized_change(S,S_first);
        if ~isempty(C_prev) && isequal(size(C_prev),size(st.C))
            Astat.coupling_change(step) = norm(st.C-C_prev,'fro') / ...
                max(norm(C_prev,'fro'),eps);
        end

        x_ref = st.K\st.b;
        Astat.backslash_relres(step) = ...
            norm(st.K*x_ref-st.b)/max(norm(st.b),eps);
        y_ref = x_ref(ctx.nU+1:end);
        y_ref_keep = y_ref(st.keep);
        x_schur = st.recover(S\rhs);
        Astat.vel_recovery_err(step) = ...
            norm(x_schur-x_ref)/max(norm(x_ref),eps);
        Astat.symmetry_res(step) = norm(S-S','fro')/max(norm(S,'fro'),eps);
        [Rcurrent,Astat.chol_flag(step)] = chol(S,'lower');

        needDeflationSpectrum = ~isempty(variants);
        if settings.computeSpectrum || needDeflationSpectrum
            ev = sort(real(eig(S)),'ascend');
            Astat.lambda_min(step) = ev(1);
            Astat.lambda_max(step) = ev(end);
            Astat.kappa(step) = ev(end)/ev(1);
        else
            ev = [];
        end

        if isempty(y_prev), x0 = zeros(nS,1); else, x0 = y_prev; end
        rhsNorm = norm(rhs); if rhsNorm == 0, rhsNorm = 1; end
        rhsScaled = rhs/rhsNorm;
        x0Scaled = x0/rhsNorm;

        if ~settings.skipUnpreconditioned
            [xs,fl,rr,it] = pcg(S,rhsScaled,settings.solverTol, ...
                settings.solverMaxit,[],[],x0Scaled);
            Astat = local_record(Astat,'pcg_unprec',step,xs,fl,rr,it, ...
                rhsNorm,y_ref_keep);
        end

        if isempty(Rfrozen)
            Rfrozen = Rcurrent;
            if Astat.chol_flag(step) ~= 0
                error('solve_varvisc_schur_sequence:notSPD', ...
                      'chol(S) failed at step %d.',step);
            end
            invApply = @(X) Rfrozen'\(Rfrozen\X);
            nS_frozen = nS;
            Astat.chol_built_step(end+1) = step;
        elseif nS ~= nS_frozen
            error('solve_varvisc_schur_sequence:changingFrozenSize', ...
                  'Cannot reuse frozen factors after a dimension change.');
        end
        [xs,fl,rr,it] = pcg(S,rhsScaled,settings.solverTol, ...
            settings.solverMaxit,invApply,[],x0Scaled);
        Astat = local_record(Astat,'chol',step,xs,fl,rr,it, ...
            rhsNorm,y_ref_keep);
        Aref = S\Omega;
        Astat.InvRelDiff(step) = norm(invApply(Omega)-Aref,'fro') / ...
            max(norm(Aref,'fro'),eps);

        if ~isempty(variants)
            if Astat.chol_flag(step) ~= 0
                error('solve_varvisc_schur_sequence:notSPD', ...
                      'Cannot construct deflation from non-SPD S at step %d.',step);
            end
            widths = local_two_tail_widths(nS,settings.smEig,settings.lgEig);
            cutoffs = local_target_cutoffs(ev,widths);
            Astat.lambda_lo_target(step) = cutoffs.lambdaLo;
            Astat.lambda_hi_target(step) = cutoffs.lambdaHi;
            Astat.tau_star_target(step) = cutoffs.tauStar;
            currentInv = @(X) Rcurrent'\(Rcurrent\X);

            smallNeeded = local_small_needed(variants);
            smallRefreshed = false;
            if smallNeeded && (isempty(smallState.V) || ...
                    local_refresh_due(step,refresh.small))
                smallState = local_build_small_state( ...
                    S,currentInv,ev,widths.small,settings,step);
                smallRefreshed = true;
                Astat.small_basis_built_step(end+1) = step;
                Astat.small_basis_info{end+1} = smallState.info;
                Astat.small_basis_ritz_values{end+1} = smallState.theta;
                if isnan(Astat.tau), Astat.tau = smallState.directTau; end
            end

            [states,Astat] = local_refresh_arm_states( ...
                states,Astat,S,smallState,smallRefreshed,widths,cutoffs, ...
                settings,refresh,variants,step);
            if smallNeeded
                Astat.lift_tau(step) = smallState.liftTau;
                Astat.small_basis_dim_history(step) = size(smallState.V,2);
            end

            for variantIndex = 1:numel(variants)
                variant = variants(variantIndex);
                [Papply,Vdim,largeDim,tauValues,basisStep,largeStep] = ...
                    local_make_preconditioner( ...
                    variant.design,S,smallState,states);
                key = variant.name;
                Astat.deflat_dim.(key) = Vdim;
                Astat.deflat_dim_history.(key)(step) = Vdim;
                Astat.large_basis_dim_history.(key)(step) = largeDim;
                Astat = local_record_build_steps( ...
                    Astat,key,step,basisStep,largeStep);
                Astat = local_record_tau( ...
                    Astat,variant.design,step,tauValues);
                [xs,fl,rr,it] = pcg(S,rhsScaled,settings.solverTol, ...
                    settings.solverMaxit,Papply,[],x0Scaled);
                Astat = local_record(Astat,key,step,xs,fl,rr,it, ...
                    rhsNorm,y_ref_keep);
            end
        end

        S_prev = S; A_prev = st.A_bc; D_prev = st.D; C_prev = st.C;
        y_prev = y_ref_keep;
        u_prev = x_ref(1:ctx.nU);

        if step == 1 || mod(step,10) == 0
            fprintf('    step %3d/%3d nS=%d contrast=%.1f |', ...
                step,nsteps,nS,Astat.nu_contrast(step));
            for keyIndex = 1:numel(keys)
                fprintf(' %s %d',keys{keyIndex}, ...
                    Astat.solver_its.(keys{keyIndex})(step));
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

function settings = local_settings(params)
    settings.smEig = local_param(params,'sm_eig',20);
    settings.lgEig = local_param(params,'lg_eig',500);
    settings.largeQ = local_param(params,'q',2);
    settings.oversampling = local_param(params,'sketch_oversampling',2);
    settings.smallSource = lower(char( ...
        local_param(params,'small_basis_source','lanczos')));
    settings.smallQ = local_param(params,'small_basis_q',2);
    settings.lanczosTolerance = local_param( ...
        params,'small_basis_lanczos_tol',1e-12);
    settings.lanczosCheckEvery = local_param( ...
        params,'small_basis_lanczos_check_every',10);
    settings.liftLargeQ = local_param(params,'lift_large_q',1);
    settings.liftTauOverride = local_param(params,'lift_tau',[]);
    settings.tauOverride = local_param(params,'tau',[]);
    settings.skipUnpreconditioned = local_param(params,'skip_unprecond',false);
    settings.computeSpectrum = local_param(params,'COMPUTE_SPECTRUM',true);
    settings.solverTol = local_param(params,'SOLVER_TOL',1e-8);
    settings.solverMaxit = local_param(params,'SOLVER_MAXIT',1e5);

    validateattributes(settings.smEig,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'params.sm_eig');
    validateattributes(settings.lgEig,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'params.lg_eig');
    validateattributes(settings.largeQ,{'numeric'}, ...
        {'scalar','integer','nonnegative'},mfilename,'params.q');
    validateattributes(settings.smallQ,{'numeric'}, ...
        {'scalar','integer','nonnegative'},mfilename,'params.small_basis_q');
    validateattributes(settings.liftLargeQ,{'numeric'}, ...
        {'scalar','integer','nonnegative'},mfilename,'params.lift_large_q');
    validateattributes(settings.oversampling,{'numeric'}, ...
        {'scalar','real','finite','>=',1},mfilename,'params.sketch_oversampling');
    if ~ismember(settings.smallSource,{'lanczos','inverse_gaussian'})
        error('solve_varvisc_schur_sequence:badSmallSource', ...
              ['small_basis_source must be ''lanczos'' or ', ...
               '''inverse_gaussian''; got %s.'],settings.smallSource);
    end
    if ~isempty(settings.tauOverride)
        validateattributes(settings.tauOverride,{'numeric'}, ...
            {'scalar','real','finite','positive'},mfilename,'params.tau');
    end
    if ~isempty(settings.liftTauOverride)
        validateattributes(settings.liftTauOverride,{'numeric'}, ...
            {'scalar','real','finite','positive'},mfilename,'params.lift_tau');
    end
end

function value = local_param(params,name,defaultValue)
    value = defaultValue;
    if isfield(params,name) && ~isempty(params.(name)), value = params.(name); end
end

function [keys,labels,variants] = local_registry(params,settings)
    keys = {}; labels = {};
    if ~settings.skipUnpreconditioned
        keys{end+1} = 'pcg_unprec';
        labels{end+1} = 'PCG (unpreconditioned)';
    end
    keys{end+1} = 'chol';
    labels{end+1} = 'PCG (exact chol of S_1, frozen) [BASELINE]';
    variants = struct('name',{},'design',{});
    if ~isfield(params,'standalone_variants') || isempty(params.standalone_variants)
        return
    end

    requested = params.standalone_variants;
    for index = 1:numel(requested)
        if ~isfield(requested,'name') || ~isvarname(requested(index).name)
            error('solve_varvisc_schur_sequence:badVariantName', ...
                  'Every deflation variant needs a valid MATLAB field name.');
        end
        design = local_variant_design(requested(index));
        variants(end+1) = struct( ...
            'name',requested(index).name,'design',design); %#ok<AGROW>
        keys{end+1} = requested(index).name; %#ok<AGROW>
        labels{end+1} = local_variant_label(design,settings); %#ok<AGROW>
    end
end

function design = local_variant_design(variant)
    if isfield(variant,'design') && ~isempty(variant.design)
        design = char(variant.design);
    elseif isfield(variant,'tail')
        switch char(variant.tail)
            case 'small', design = 'shared_small';
            case 'large', design = 'gaussian_large';
            case 'both', design = 'sequential_shared_subspace';
            otherwise
                error('solve_varvisc_schur_sequence:badTail', ...
                      'Unsupported legacy tail selection %s.',variant.tail);
        end
    else
        error('solve_varvisc_schur_sequence:missingDesign', ...
              'Every deflation variant needs a design.');
    end
    allowed = {'shared_small','gaussian_large', ...
        'sequential_shared_subspace','concatenated_once', ...
        'adaptive_small_lift_large'};
    if ~ismember(design,allowed)
        error('solve_varvisc_schur_sequence:badDesign', ...
              'Unsupported deflation design %s.',design);
    end
end

function label = local_variant_label(design,settings)
    smallDescription = settings.smallSource;
    switch design
        case 'shared_small'
            label = sprintf('PCG (shared %s small-tail deflation, k=%d)', ...
                smallDescription,settings.smEig);
        case 'gaussian_large'
            label = sprintf('PCG (Gaussian large-tail deflation, k=%d)', ...
                settings.lgEig);
        case 'sequential_shared_subspace'
            label = sprintf(['PCG (two-stage shared-subspace deflation, ', ...
                'k=%d+%d)'],settings.smEig,settings.lgEig);
        case 'concatenated_once'
            label = sprintf('PCG (one concatenated two-tail deflator, k=%d+%d)', ...
                settings.smEig,settings.lgEig);
        case 'adaptive_small_lift_large'
            label = sprintf(['PCG (adaptive small lift plus transformed ', ...
                'large-tail deflation, k=%d+%d)'], ...
                settings.smEig,settings.lgEig);
    end
end

function Astat = local_prealloc(keys,labels,nsteps,settings)
    Astat.solver_keys = keys(:); Astat.solver_labels = labels(:);
    Astat.solver_its = struct(); Astat.solver_flag = struct();
    Astat.solver_relres = struct(); Astat.solver_err = struct();
    for index = 1:numel(keys)
        key = keys{index};
        Astat.solver_its.(key) = nan(nsteps,1);
        Astat.solver_flag.(key) = nan(nsteps,1);
        Astat.solver_relres.(key) = nan(nsteps,1);
        Astat.solver_err.(key) = nan(nsteps,1);
    end
    fields = {'backslash_relres','vel_recovery_err','symmetry_res','chol_flag', ...
        'ReldiffF','RelInitdiffF','InvRelDiff','coupling_change','A_change', ...
        'D_change','pressure_schur_change','nu_contrast','lambda_min', ...
        'lambda_max','kappa','nC','nS','lambda_lo_target', ...
        'lambda_hi_target','tau_star_target','lift_tau', ...
        'small_basis_dim_history'};
    for index = 1:numel(fields), Astat.(fields{index}) = nan(nsteps,1); end
    Astat.LowRankInvRelDiff = nan(nsteps,1);
    Astat.tau = NaN;
    tauFields = {'shared_small','gaussian_large','sequential_stage1', ...
        'sequential_stage2','concatenated_once','adaptive_lift', ...
        'adaptive_large'};
    Astat.deflation_tau = struct();
    for index = 1:numel(tauFields)
        Astat.deflation_tau.(tauFields{index}) = nan(nsteps,1);
    end
    Astat.deflat_dim = struct();
    Astat.deflat_dim_history = struct();
    Astat.large_basis_dim_history = struct();
    Astat.basis_built_step = struct();
    Astat.large_basis_built_step = struct();
    for index = 1:numel(keys)
        key = keys{index};
        if ~ismember(key,{'pcg_unprec','chol'})
            Astat.deflat_dim_history.(key) = nan(nsteps,1);
            Astat.large_basis_dim_history.(key) = nan(nsteps,1);
            Astat.basis_built_step.(key) = [];
            Astat.large_basis_built_step.(key) = [];
        end
    end
    Astat.deflat_requested_dim = struct( ...
        'small',settings.smEig,'large',settings.lgEig, ...
        'combined',settings.smEig+settings.lgEig);
    Astat.deflat_tail_dim = struct();
    Astat.small_basis_source = settings.smallSource;
    Astat.small_basis_built_step = [];
    Astat.small_basis_info = {};
    Astat.small_basis_ritz_values = {};
    Astat.sketch_oversampling = settings.oversampling;
    Astat.chol_built_step = [];
end

function refresh = local_refresh_intervals(params)
    refresh.small = local_refresh_value(params,'SMALL_BASIS_REFRESH', ...
        {'DEFLAT_SMALL_PREC_REFRESH','DEFLAT_PREC_REFRESH'});
    refresh.gaussianLarge = local_refresh_value( ...
        params,'DEFLAT_GAUSSIAN_LARGE_REFRESH', ...
        {'DEFLAT_LARGE_PREC_REFRESH','DEFLAT_PREC_REFRESH'});
    refresh.sequential = local_refresh_value( ...
        params,'DEFLAT_SEQUENTIAL_SHARED_LARGE_REFRESH', ...
        {'DEFLAT_BOTH_PREC_REFRESH','DEFLAT_PREC_REFRESH'});
    refresh.concatenated = local_refresh_value( ...
        params,'DEFLAT_CONCATENATED_ONCE_LARGE_REFRESH', ...
        {'DEFLAT_BOTH_PREC_REFRESH','DEFLAT_PREC_REFRESH'});
    refresh.adaptive = local_refresh_value( ...
        params,'DEFLAT_ADAPTIVE_LIFT_LARGE_REFRESH', ...
        {'DEFLAT_BOTH_PREC_REFRESH','DEFLAT_PREC_REFRESH'});
end

function value = local_refresh_value(params,newField,legacyFields)
    value = Inf;
    if isfield(params,newField) && ~isempty(params.(newField))
        value = params.(newField);
    else
        for index = 1:numel(legacyFields)
            field = legacyFields{index};
            if isfield(params,field) && ~isempty(params.(field))
                value = params.(field);
                break
            end
        end
    end
    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        ((isinf(value) && value > 0) || ...
         (isfinite(value) && value >= 1 && value == floor(value)));
    if ~valid
        error('solve_varvisc_schur_sequence:badDeflatRefresh', ...
              '%s must be a positive integer or Inf.',newField);
    end
end

function due = local_refresh_due(step,interval)
    due = step == 1 || ...
        (isfinite(interval) && mod(step-1,interval) == 0);
end

function needed = local_small_needed(variants)
    designs = {variants.design};
    needed = any(~strcmp(designs,'gaussian_large'));
end

function enabled = local_design_enabled(variants,design)
    enabled = any(strcmp({variants.design},design));
end

function state = local_empty_small_state()
    state = struct('V',[],'theta',[],'liftTau',NaN,'directTau',NaN, ...
                   'info',struct(),'basisBuildStep',NaN);
end

function states = local_empty_arm_states()
    empty = struct('V',[],'largeV',[],'tau1',NaN,'tau2',NaN, ...
                   'basisBuildStep',NaN,'largeBuildStep',NaN);
    states.gaussian_large = empty;
    states.sequential_shared_subspace = empty;
    states.concatenated_once = empty;
    states.adaptive_small_lift_large = empty;
end

function smallState = local_build_small_state( ...
        S,currentInv,ev,kSmall,settings,step)
    nS = size(S,1);
    switch settings.smallSource
        case 'lanczos'
            computedRank = min(kSmall+1,nS);
            options = struct('maxSteps',nS, ...
                'checkEvery',settings.lanczosCheckEvery, ...
                'tolerance',settings.lanczosTolerance, ...
                'operatorNorm',ev(end));
            [Vall,thetaAll,info] = ...
                fully_reorthogonalized_lanczos_smallest( ...
                S,computedRank,options);
            V = Vall(:,1:kSmall);
            theta = thetaAll(1:kSmall);
            info.computedRank = computedRank;
        case 'inverse_gaussian'
            [V,theta,info] = gaussian_rayleigh_ritz_basis( ...
                currentInv,@(X) S*X,nS,kSmall,settings.smallQ, ...
                settings.oversampling,'smallest');
    end

    E = V'*(S*V);
    E = (E+E')/2;
    lambdaHat = min(real(eig(E)));
    denominator = ev(end)-lambdaHat;
    if ~(lambdaHat > 0 && denominator > 0)
        error('solve_varvisc_schur_sequence:badLiftTau', ...
              'Cannot form a positive dynamic lift tau at step %d.',step);
    end
    liftTau = lambdaHat/denominator;
    if ~isempty(settings.liftTauOverride)
        liftTau = settings.liftTauOverride;
    end
    directTau = ev(end);
    if ~isempty(settings.tauOverride), directTau = settings.tauOverride; end
    info.lambdaHat = lambdaHat;
    info.liftedLambdaHat = (1+1/liftTau)*lambdaHat;
    info.lambdaMax = ev(end);
    info.dynamicLiftTau = lambdaHat/denominator;
    info.usedFixedLiftTau = ~isempty(settings.liftTauOverride);
    smallState = struct('V',V,'theta',theta,'liftTau',liftTau, ...
        'directTau',directTau,'info',info,'basisBuildStep',step);
end

function [states,Astat] = local_refresh_arm_states( ...
        states,Astat,S,smallState,smallRefreshed,widths,cutoffs, ...
        settings,refresh,variants,step)
    nS = size(S,1);

    if local_design_enabled(variants,'gaussian_large') && ...
            (isempty(states.gaussian_large.V) || ...
             local_refresh_due(step,refresh.gaussianLarge))
        [V,~,~] = gaussian_rayleigh_ritz_basis( ...
            @(X) S*X,@(X) S*X,nS,widths.large,settings.largeQ, ...
            settings.oversampling,'largest');
        states.gaussian_large.V = V;
        states.gaussian_large.tau1 = cutoffs.lambdaHi;
        states.gaussian_large.basisBuildStep = step;
        states.gaussian_large.largeBuildStep = step;
    end

    if local_design_enabled(variants,'sequential_shared_subspace')
        largeRefreshed = isempty(states.sequential_shared_subspace.largeV) || ...
            local_refresh_due(step,refresh.sequential);
        if largeRefreshed
            [Vlarge,~,~] = gaussian_rayleigh_ritz_basis( ...
                @(X) S*X,@(X) S*X,nS,widths.large,settings.largeQ, ...
                settings.oversampling,'largest');
            states.sequential_shared_subspace.largeV = Vlarge;
            states.sequential_shared_subspace.largeBuildStep = step;
        end
        if largeRefreshed || smallRefreshed || ...
                isempty(states.sequential_shared_subspace.V)
            states.sequential_shared_subspace.V = local_combine_bases( ...
                states.sequential_shared_subspace.largeV,smallState.V);
            states.sequential_shared_subspace.tau1 = cutoffs.lambdaHi;
            states.sequential_shared_subspace.tau2 = cutoffs.tauStar;
            states.sequential_shared_subspace.basisBuildStep = step;
        end
    end

    if local_design_enabled(variants,'concatenated_once')
        largeRefreshed = isempty(states.concatenated_once.largeV) || ...
            local_refresh_due(step,refresh.concatenated);
        if largeRefreshed
            [Vlarge,~,~] = gaussian_rayleigh_ritz_basis( ...
                @(X) S*X,@(X) S*X,nS,widths.large,settings.largeQ, ...
                settings.oversampling,'largest');
            states.concatenated_once.largeV = Vlarge;
            states.concatenated_once.largeBuildStep = step;
        end
        if largeRefreshed || smallRefreshed || isempty(states.concatenated_once.V)
            states.concatenated_once.V = local_combine_bases( ...
                states.concatenated_once.largeV,smallState.V);
            states.concatenated_once.tau1 = cutoffs.tauStar;
            states.concatenated_once.basisBuildStep = step;
        end
    end

    if local_design_enabled(variants,'adaptive_small_lift_large')
        rebuild = isempty(states.adaptive_small_lift_large.V) || ...
            smallRefreshed || local_refresh_due(step,refresh.adaptive);
        if rebuild
            [~,liftHalf] = small_eigenvalue_lift( ...
                smallState.V,smallState.liftTau);
            transformedApply = @(X) liftHalf(S*liftHalf(X));
            [Vlarge,~,~] = gaussian_rayleigh_ritz_basis( ...
                transformedApply,transformedApply,nS,widths.large, ...
                settings.liftLargeQ,settings.oversampling,'largest');
            states.adaptive_small_lift_large.V = Vlarge;
            states.adaptive_small_lift_large.tau1 = cutoffs.tauStar;
            states.adaptive_small_lift_large.tau2 = smallState.liftTau;
            states.adaptive_small_lift_large.basisBuildStep = step;
            states.adaptive_small_lift_large.largeBuildStep = step;
        end
    end

    Astat.deflat_tail_dim.shared_small = size(smallState.V,2);
    Astat.deflat_tail_dim.gaussian_large = ...
        size(states.gaussian_large.V,2);
    Astat.deflat_tail_dim.sequential_large = ...
        size(states.sequential_shared_subspace.largeV,2);
    Astat.deflat_tail_dim.concatenated_large = ...
        size(states.concatenated_once.largeV,2);
    Astat.deflat_tail_dim.adaptive_large = ...
        size(states.adaptive_small_lift_large.V,2);
end

function V = local_combine_bases(Vlarge,Vsmall)
    V = orth(real([Vlarge,Vsmall]));
    if isempty(V)
        error('solve_varvisc_schur_sequence:emptyCombinedBasis', ...
              'The combined two-tail basis collapsed to rank zero.');
    end
end

function [Papply,Vdim,largeDim,tauValues,basisStep,largeStep] = ...
        local_make_preconditioner(design,S,smallState,states)
    tauValues = struct('first',NaN,'second',NaN);
    largeDim = 0;
    switch design
        case 'shared_small'
            V = smallState.V;
            tauValues.first = smallState.directTau;
            Papply = src.precond.deflation_P_apply( ...
                V,S,tauValues.first,'handle',0);
            Vdim = size(V,2);
            basisStep = smallState.basisBuildStep;
            largeStep = NaN;
        case 'gaussian_large'
            state = states.gaussian_large;
            tauValues.first = state.tau1;
            Papply = src.precond.deflation_P_apply( ...
                state.V,S,tauValues.first,'handle',0);
            Vdim = size(state.V,2); largeDim = Vdim;
            basisStep = state.basisBuildStep;
            largeStep = state.largeBuildStep;
        case 'sequential_shared_subspace'
            state = states.sequential_shared_subspace;
            tauValues.first = state.tau1;
            tauValues.second = state.tau2;
            P1half = src.precond.deflation_Psqrt_apply( ...
                state.V,S,tauValues.first,'handle');
            S1apply = @(X) P1half(S*P1half(X));
            P2apply = src.precond.deflation_P_apply( ...
                state.V,S1apply,tauValues.second,'handle',0);
            Papply = @(X) P1half(P2apply(P1half(X)));
            Vdim = size(state.V,2);
            largeDim = size(state.largeV,2);
            basisStep = state.basisBuildStep;
            largeStep = state.largeBuildStep;
        case 'concatenated_once'
            state = states.concatenated_once;
            tauValues.first = state.tau1;
            Papply = src.precond.deflation_P_apply( ...
                state.V,S,tauValues.first,'handle',0);
            Vdim = size(state.V,2);
            largeDim = size(state.largeV,2);
            basisStep = state.basisBuildStep;
            largeStep = state.largeBuildStep;
        case 'adaptive_small_lift_large'
            state = states.adaptive_small_lift_large;
            tauValues.first = state.tau2;
            tauValues.second = state.tau1;
            [~,liftHalf] = small_eigenvalue_lift( ...
                smallState.V,tauValues.first);
            transformedApply = @(X) liftHalf(S*liftHalf(X));
            P2apply = src.precond.deflation_P_apply( ...
                state.V,transformedApply,tauValues.second,'handle',0);
            Papply = @(X) liftHalf(P2apply(liftHalf(X)));
            largeDim = size(state.V,2);
            Vdim = size(smallState.V,2)+largeDim;
            basisStep = state.basisBuildStep;
            largeStep = state.largeBuildStep;
    end
end

function Astat = local_record_build_steps( ...
        Astat,key,step,basisStep,largeStep)
    if basisStep == step
        Astat.basis_built_step.(key)(end+1) = step;
    end
    if largeStep == step
        Astat.large_basis_built_step.(key)(end+1) = step;
    end
end

function Astat = local_record_tau(Astat,design,step,tauValues)
    switch design
        case 'shared_small'
            Astat.deflation_tau.shared_small(step) = tauValues.first;
        case 'gaussian_large'
            Astat.deflation_tau.gaussian_large(step) = tauValues.first;
        case 'sequential_shared_subspace'
            Astat.deflation_tau.sequential_stage1(step) = tauValues.first;
            Astat.deflation_tau.sequential_stage2(step) = tauValues.second;
        case 'concatenated_once'
            Astat.deflation_tau.concatenated_once(step) = tauValues.first;
        case 'adaptive_small_lift_large'
            Astat.deflation_tau.adaptive_lift(step) = tauValues.first;
            Astat.deflation_tau.adaptive_large(step) = tauValues.second;
    end
end

function cutoffs = local_target_cutoffs(ev,widths)
    nS = numel(ev);
    cutoffs.lambdaLo = ev(widths.small+1);
    cutoffs.lambdaHi = ev(nS-widths.large);
    cutoffs.tauStar = sqrt(cutoffs.lambdaLo*cutoffs.lambdaHi);
end

function widths = local_two_tail_widths(nS,smRequested,lgRequested)
    available = nS-1;
    if available < 2
        error('solve_varvisc_schur_sequence:schurTooSmall', ...
              'Two-tail deflation requires at least a 3-by-3 Schur matrix.');
    end
    widths.small = min(smRequested,available);
    widths.large = min(lgRequested,available);
    if widths.small+widths.large <= available
        return
    end
    totalRequested = smRequested+lgRequested;
    widths.small = max(1,min(smRequested, ...
        floor(available*smRequested/totalRequested)));
    widths.large = max(1,min(lgRequested,available-widths.small));
    remaining = available-widths.small-widths.large;
    addSmall = min(remaining,smRequested-widths.small);
    widths.small = widths.small+addSmall;
    remaining = available-widths.small-widths.large;
    widths.large = widths.large+min(remaining,lgRequested-widths.large);
end

function Astat = local_record(Astat,key,step,xscaled,fl,rr,it,rhsnorm,yref)
    Astat.solver_its.(key)(step) = it;
    Astat.solver_flag.(key)(step) = fl;
    Astat.solver_relres.(key)(step) = rr;
    x = xscaled*rhsnorm;
    Astat.solver_err.(key)(step) = norm(x-yref)/max(norm(yref),eps);
end
