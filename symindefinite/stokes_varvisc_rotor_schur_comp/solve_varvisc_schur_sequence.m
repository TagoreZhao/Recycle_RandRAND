function Astat = solve_varvisc_schur_sequence(cfg, params, save_dir)
%SOLVE_VARVISC_SCHUR_SEQUENCE  PCG benchmark on Schur operator handles.
%   All two-tail designs share one centrally refreshed smallest-mode basis.
%   The original-operator arms also share one largest-mode Gaussian sketch;
%   the adaptive arm owns a separate sketch of its lifted operator.

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
    SFirstApply = []; SPrevApply = []; nSFirst = -1;
    SFirstDense = []; SPrevDense = [];
    A_prev = []; D_prev = []; C_prev = [];
    Rfrozen = []; invApply = []; nS_frozen = -1; Omega = [];
    pressureOmega = [];
    smallState = local_empty_small_state();
    sharedLargeState = local_empty_shared_large_state();
    states = local_empty_arm_states();

    for step = 1:nsteps
        tcur = step*params.dt;
        st = varvisc_schur_step_operator(ctx,tcur,u_prev);
        Sapply = st.apply; rhs = st.rhs_S; nS = st.nS;
        Sdense = []; Rcurrent = [];
        rawSpectrumRecorded = false;
        Astat.nC(step) = st.nC;
        Astat.nS(step) = nS;
        Astat.nu_contrast(step) = max(st.nu_e)/min(st.nu_e);

        if isempty(SFirstApply)
            SFirstApply = Sapply;
            nSFirst = nS;
            Omega = randn(nS,10)/sqrt(10);
            pressureOmega = randn(ctx.nP-1,min(10,ctx.nP-1))/sqrt(10);
        elseif nSFirst ~= nS
            error('solve_varvisc_schur_sequence:changingSize', ...
                  'Schur dimension changed from %d to %d at step %d.', ...
                  nSFirst,nS,step);
        end

        Astat.RelInitdiffProbe(step) = ...
            local_operator_change(Sapply,SFirstApply,Omega);
        Astat.symmetry_probe_res(step) = local_symmetry_probe(Sapply,Omega);
        if ~isempty(SPrevApply)
            Astat.ReldiffProbe(step) = ...
                local_operator_change(Sapply,SPrevApply,Omega);
            Astat.A_change(step) = local_normalized_change(st.A_bc,A_prev);
            Astat.D_change(step) = local_normalized_change(st.D,D_prev);
            Astat.pressure_schur_change_probe(step) = ...
                local_pressure_operator_change( ...
                Sapply,SPrevApply,pressureOmega,nS,ctx.nP-1);
        end
        if ~isempty(C_prev) && isequal(size(C_prev),size(st.C))
            Astat.coupling_change(step) = norm(st.C-C_prev,'fro') / ...
                max(norm(C_prev,'fro'),eps);
        end

        x_ref = st.K\st.b;
        Astat.backslash_relres(step) = ...
            norm(st.K*x_ref-st.b)/max(norm(st.b),eps);
        y_ref = x_ref(ctx.nU+1:end);
        y_ref_keep = y_ref(st.keep);
        x_schur = st.recover(y_ref_keep);
        Astat.vel_recovery_err(step) = ...
            norm(x_schur-x_ref)/max(norm(x_ref),eps);
        Astat.schur_ref_relres(step) = ...
            norm(Sapply(y_ref_keep)-rhs)/max(norm(rhs),eps);

        if settings.exactDenseDiagnostics
            [Sdense,Astat] = local_require_dense(Sdense,st,Astat,step);
            if isempty(SFirstDense), SFirstDense = Sdense; end
            Astat.RelInitdiffF(step) = ...
                local_normalized_change(Sdense,SFirstDense);
            Astat.symmetry_res(step) = ...
                norm(Sdense-Sdense','fro')/max(norm(Sdense,'fro'),eps);
            if ~isempty(SPrevDense)
                Astat.ReldiffF(step) = ...
                    local_normalized_change(Sdense,SPrevDense);
                Spp = Sdense(1:ctx.nP-1,1:ctx.nP-1);
                SppPrev = SPrevDense(1:ctx.nP-1,1:ctx.nP-1);
                Astat.pressure_schur_change(step) = ...
                    local_normalized_change(Spp,SppPrev);
            end
            [Rcurrent,Astat.chol_flag(step)] = chol(Sdense,'lower');
            if Astat.chol_flag(step) ~= 0
                error('solve_varvisc_schur_sequence:notSPD', ...
                      'chol(S) failed at step %d.',step);
            end
        end

        if isempty(y_prev), x0 = zeros(nS,1); else, x0 = y_prev; end
        rhsNorm = norm(rhs); if rhsNorm == 0, rhsNorm = 1; end
        rhsScaled = rhs/rhsNorm;
        x0Scaled = x0/rhsNorm;

        if ~settings.skipUnpreconditioned
            [xs,fl,rr,it] = pcg(Sapply,rhsScaled,settings.solverTol, ...
                settings.solverMaxit,[],[],x0Scaled);
            Astat = local_record(Astat,'pcg_unprec',step,xs,fl,rr,it, ...
                rhsNorm,y_ref_keep);
        end

        if isempty(Rfrozen)
            [Sdense,Astat] = local_require_dense(Sdense,st,Astat,step);
            if isempty(Rcurrent)
                [Rcurrent,Astat.chol_flag(step)] = chol(Sdense,'lower');
            end
            if Astat.chol_flag(step) ~= 0
                error('solve_varvisc_schur_sequence:notSPD', ...
                      'chol(S) failed at step %d.',step);
            end
            Rfrozen = Rcurrent;
            invApply = @(X) Rfrozen'\(Rfrozen\X);
            nS_frozen = nS;
            Astat.chol_built_step(end+1) = step;
        elseif nS ~= nS_frozen
            error('solve_varvisc_schur_sequence:changingFrozenSize', ...
                  'Cannot reuse frozen factors after a dimension change.');
        end
        [xs,fl,rr,it] = pcg(Sapply,rhsScaled,settings.solverTol, ...
            settings.solverMaxit,invApply,[],x0Scaled);
        Astat = local_record(Astat,'chol',step,xs,fl,rr,it, ...
            rhsNorm,y_ref_keep);
        if settings.plotExtremeEigenvalues
            cholSpectrumApply = @(X) Rfrozen\(Sapply(Rfrozen'\X));
            Astat = local_record_extreme_spectrum( ...
                Astat,'chol',step,cholSpectrumApply,nS,settings);
        end
        if settings.exactDenseDiagnostics
            Aref = Sdense\Omega;
            Astat.InvRelDiff(step) = norm(invApply(Omega)-Aref,'fro') / ...
                max(norm(Aref,'fro'),eps);
        end

        if ~isempty(variants)
            smallNeeded = local_small_needed(variants);
            largeNeeded = local_large_needed(variants);
            widths = local_two_tail_widths( ...
                nS,local_small_basis_width(settings,smallNeeded), ...
                local_large_basis_width(settings,largeNeeded));
            [spectrum,~] = local_spectral_summary( ...
                Sapply,nS,widths,Sdense,settings);
            Astat.lambda_min(step) = spectrum.lambdaMin;
            Astat.lambda_max(step) = spectrum.lambdaMax;
            Astat.kappa(step) = spectrum.lambdaMax/spectrum.lambdaMin;
            Astat.spectrum_is_exact(step) = spectrum.isExact;
            if settings.plotExtremeEigenvalues && ...
                    isfield(Astat.system_lambda_min,'pcg_unprec')
                Astat = local_store_extreme_spectrum( ...
                    Astat,'pcg_unprec',step,spectrum.lambdaMin, ...
                    spectrum.lambdaMax,spectrum.flag,NaN,spectrum.isExact);
                rawSpectrumRecorded = true;
            end
            cutoffs = spectrum.cutoffs;
            Astat.lambda_lo_target(step) = cutoffs.lambdaLo;
            Astat.lambda_hi_target(step) = cutoffs.lambdaHi;
            Astat.tau_star_target(step) = cutoffs.tauStar;

            [sharedLargeState,sharedLargeRefreshed,Astat] = ...
                local_refresh_shared_large_state( ...
                sharedLargeState,Astat,Sapply,nS,cutoffs,settings, ...
                refresh,variants,step);

            smallRefreshed = false;
            refreshSmall = smallNeeded && (isempty(smallState.V) || ...
                local_refresh_due(step,refresh.small));
            currentInv = [];
            if refreshSmall && strcmp(settings.smallSource,'inverse_gaussian')
                [Sdense,Astat] = local_require_dense(Sdense,st,Astat,step);
                if isempty(Rcurrent)
                    [Rcurrent,Astat.chol_flag(step)] = chol(Sdense,'lower');
                end
                if Astat.chol_flag(step) ~= 0
                    error('solve_varvisc_schur_sequence:notSPD', ...
                          'chol(S) failed at step %d.',step);
                end
                currentInv = @(X) Rcurrent'\(Rcurrent\X);
            end
            if refreshSmall
                smallState = local_build_small_state( ...
                    Sapply,currentInv,spectrum,nS,widths.small,settings,step);
                smallRefreshed = true;
                Astat.small_basis_built_step(end+1) = step;
                Astat.small_basis_info{end+1} = smallState.info;
                Astat.small_basis_ritz_values{end+1} = smallState.theta;
                if isnan(Astat.tau), Astat.tau = smallState.directTau; end
            end

            [states,Astat] = local_refresh_arm_states( ...
                sharedLargeState,states,Astat,Sapply,nS,smallState, ...
                smallRefreshed,sharedLargeRefreshed,cutoffs,settings, ...
                refresh,variants,step);
            if smallNeeded
                Astat.lift_tau(step) = smallState.liftTau;
                Astat.small_basis_dim_history(step) = size(smallState.V,2);
            end

            for variantIndex = 1:numel(variants)
                variant = variants(variantIndex);
                [Papply,spectralApply,Vdim,largeDim,tauValues, ...
                    basisStep,largeStep] = ...
                    local_make_preconditioner( ...
                    variant.design,Sapply,smallState,sharedLargeState,states, ...
                    settings.plotExtremeEigenvalues);
                key = variant.name;
                Astat.deflat_dim.(key) = Vdim;
                Astat.deflat_dim_history.(key)(step) = Vdim;
                Astat.large_basis_dim_history.(key)(step) = largeDim;
                Astat = local_record_build_steps( ...
                    Astat,key,step,basisStep,largeStep);
                Astat = local_record_tau( ...
                    Astat,variant.design,step,tauValues);
                [xs,fl,rr,it] = pcg(Sapply,rhsScaled,settings.solverTol, ...
                    settings.solverMaxit,Papply,[],x0Scaled);
                Astat = local_record(Astat,key,step,xs,fl,rr,it, ...
                    rhsNorm,y_ref_keep);
                if settings.plotExtremeEigenvalues
                    Astat = local_record_extreme_spectrum( ...
                        Astat,key,step,spectralApply,nS,settings);
                end
            end
        elseif settings.computeSpectrum
            widths = struct('small',0,'large',0);
            [spectrum,~] = local_spectral_summary( ...
                Sapply,nS,widths,Sdense,settings);
            Astat.lambda_min(step) = spectrum.lambdaMin;
            Astat.lambda_max(step) = spectrum.lambdaMax;
            Astat.kappa(step) = spectrum.lambdaMax/spectrum.lambdaMin;
            Astat.spectrum_is_exact(step) = spectrum.isExact;
            if settings.plotExtremeEigenvalues && ...
                    isfield(Astat.system_lambda_min,'pcg_unprec')
                Astat = local_store_extreme_spectrum( ...
                    Astat,'pcg_unprec',step,spectrum.lambdaMin, ...
                    spectrum.lambdaMax,spectrum.flag,NaN,spectrum.isExact);
                rawSpectrumRecorded = true;
            end
        end

        if settings.plotExtremeEigenvalues && ~rawSpectrumRecorded && ...
                isfield(Astat.system_lambda_min,'pcg_unprec')
            Astat = local_record_extreme_spectrum( ...
                Astat,'pcg_unprec',step,Sapply,nS,settings);
        end

        SPrevApply = Sapply;
        if settings.exactDenseDiagnostics, SPrevDense = Sdense; end
        A_prev = st.A_bc; D_prev = st.D; C_prev = st.C;
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

function value = local_operator_change(Aapply,Bapply,Omega)
    value = local_normalized_change(Aapply(Omega),Bapply(Omega));
end

function value = local_pressure_operator_change( ...
        Aapply,Bapply,Omega,nS,nPressure)
    sample = zeros(nS,size(Omega,2));
    sample(1:nPressure,:) = Omega;
    Apressure = Aapply(sample); Apressure = Apressure(1:nPressure,:);
    Bpressure = Bapply(sample); Bpressure = Bpressure(1:nPressure,:);
    value = local_normalized_change(Apressure,Bpressure);
end

function value = local_symmetry_probe(Aapply,Omega)
    projected = Omega'*Aapply(Omega);
    value = norm(projected-projected','fro')/max(norm(projected,'fro'),eps);
end

function [Sdense,Astat] = local_require_dense(Sdense,st,Astat,step)
    if ~isempty(Sdense), return; end
    Sdense = st.to_dense();
    if isempty(Astat.dense_materialized_step) || ...
            Astat.dense_materialized_step(end) ~= step
        Astat.dense_materialized_step(end+1) = step;
    end
end

function [spectrum,ev] = local_spectral_summary( ...
        Sapply,nS,widths,Sdense,settings)
    ev = [];
    if ~isempty(Sdense)
        ev = sort(real(eig(Sdense)),'ascend');
        spectrum.lambdaMin = ev(1);
        spectrum.lambdaMax = ev(end);
        spectrum.isExact = true;
        spectrum.flag = 0;
        spectrum.cutoffs = local_target_cutoffs(ev,widths);
        return
    end

    options = struct('issym',true,'isreal',true, ...
        'tol',settings.spectralTolerance,'maxit',settings.spectralMaxit, ...
        'v0',ones(nS,1)/sqrt(nS));
    smallCount = max(1,widths.small+1);
    largeCount = max(1,widths.large+1);
    [~,largeValues,largeFlag] = eigs( ...
        Sapply,nS,largeCount,'largestreal',options);
    largeValues = sort(real(diag(largeValues)),'descend');
    [~,smallValues,smallFlag] = local_smallest_abs_ritz( ...
        Sapply,nS,smallCount,largeValues(1),options,settings);
    spectrum.lambdaMin = smallValues(1);
    spectrum.lambdaMax = largeValues(1);
    spectrum.isExact = false;
    spectrum.flag = max(smallFlag,largeFlag);
    if spectrum.lambdaMin <= 0
        error('solve_varvisc_schur_sequence:notSPD', ...
              'The estimated smallest Schur eigenvalue is nonpositive.');
    end
    spectrum.cutoffs = struct('lambdaLo',NaN,'lambdaHi',NaN,'tauStar',NaN);
    if widths.small > 0
        spectrum.cutoffs.lambdaLo = smallValues(end);
    end
    if widths.large > 0
        spectrum.cutoffs.lambdaHi = largeValues(end);
    end
    if widths.small > 0 && widths.large > 0
        spectrum.cutoffs.tauStar = sqrt( ...
            spectrum.cutoffs.lambdaLo*spectrum.cutoffs.lambdaHi);
    end
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
    settings.plotExtremeEigenvalues = local_param( ...
        params,'PLOT_EXTREME_EIGENVALUES',false);
    settings.exactDenseDiagnostics = local_param( ...
        params,'EXACT_DENSE_DIAGNOSTICS',false);
    settings.spectralTolerance = local_param( ...
        params,'SPECTRAL_RITZ_TOL',1e-10);
    settings.spectralMaxit = local_param(params,'SPECTRAL_RITZ_MAXIT',1000);
    settings.solverTol = local_param(params,'SOLVER_TOL',1e-8);
    settings.solverMaxit = local_param(params,'SOLVER_MAXIT',1e5);

    validateattributes(settings.smEig,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'params.sm_eig');
    validateattributes(settings.lgEig,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'params.lg_eig');
    validateattributes(settings.largeQ,{'numeric'}, ...
        {'scalar','integer','nonnegative'},mfilename,'params.q');
    validateattributes(settings.oversampling,{'numeric'}, ...
        {'scalar','real','finite','>=',1},mfilename,'params.sketch_oversampling');
    validateattributes(settings.smallQ,{'numeric'}, ...
        {'scalar','integer','nonnegative'},mfilename,'params.small_basis_q');
    validateattributes(settings.liftLargeQ,{'numeric'}, ...
        {'scalar','integer','nonnegative'},mfilename,'params.lift_large_q');
    validateattributes(settings.exactDenseDiagnostics,{'logical','numeric'}, ...
        {'scalar'},mfilename,'params.EXACT_DENSE_DIAGNOSTICS');
    validateattributes(settings.plotExtremeEigenvalues,{'logical','numeric'}, ...
        {'scalar'},mfilename,'params.PLOT_EXTREME_EIGENVALUES');
    validateattributes(settings.spectralTolerance,{'numeric'}, ...
        {'scalar','real','finite','positive'},mfilename,'params.SPECTRAL_RITZ_TOL');
    validateattributes(settings.spectralMaxit,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'params.SPECTRAL_RITZ_MAXIT');
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
    smallWidth = local_small_basis_width(settings,true);
    largeWidth = local_large_basis_width(settings,true);
    switch design
        case 'shared_small'
            label = sprintf('PCG (shared %s small-tail deflation, k=%d)', ...
                smallDescription,smallWidth);
        case 'gaussian_large'
            label = sprintf('PCG (Gaussian large-tail deflation, k=%d)', ...
                largeWidth);
        case 'sequential_shared_subspace'
            label = sprintf(['PCG (two-stage shared-subspace deflation, ', ...
                'k=%d+%d)'],smallWidth,largeWidth);
        case 'concatenated_once'
            label = sprintf('PCG (one concatenated two-tail deflator, k=%d+%d)', ...
                smallWidth,largeWidth);
        case 'adaptive_small_lift_large'
            label = sprintf(['PCG (adaptive small lift plus transformed ', ...
                'large-tail deflation, k=%d+%d)'], ...
                smallWidth,largeWidth);
    end
end

function Astat = local_prealloc(keys,labels,nsteps,settings)
    Astat.solver_keys = keys(:); Astat.solver_labels = labels(:);
    Astat.solver_its = struct(); Astat.solver_flag = struct();
    Astat.solver_relres = struct(); Astat.solver_err = struct();
    Astat.system_lambda_min = struct();
    Astat.system_lambda_max = struct();
    Astat.system_kappa = struct();
    Astat.system_spectrum_flag = struct();
    Astat.system_spectrum_residual = struct();
    Astat.system_spectrum_is_exact = struct();
    for index = 1:numel(keys)
        key = keys{index};
        Astat.solver_its.(key) = nan(nsteps,1);
        Astat.solver_flag.(key) = nan(nsteps,1);
        Astat.solver_relres.(key) = nan(nsteps,1);
        Astat.solver_err.(key) = nan(nsteps,1);
        Astat.system_lambda_min.(key) = nan(nsteps,1);
        Astat.system_lambda_max.(key) = nan(nsteps,1);
        Astat.system_kappa.(key) = nan(nsteps,1);
        Astat.system_spectrum_flag.(key) = nan(nsteps,1);
        Astat.system_spectrum_residual.(key) = nan(nsteps,1);
        Astat.system_spectrum_is_exact.(key) = false(nsteps,1);
    end
    fields = {'backslash_relres','vel_recovery_err','schur_ref_relres', ...
        'symmetry_res','symmetry_probe_res','chol_flag','ReldiffF', ...
        'RelInitdiffF','ReldiffProbe','RelInitdiffProbe','InvRelDiff', ...
        'coupling_change','A_change','D_change','pressure_schur_change', ...
        'pressure_schur_change_probe','nu_contrast','lambda_min', ...
        'lambda_max','kappa','spectrum_is_exact','nC','nS', ...
        'lambda_lo_target','lambda_hi_target','tau_star_target','lift_tau', ...
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
    smallWidth = local_small_basis_width(settings,true);
    largeWidth = local_large_basis_width(settings,true);
    adaptiveLargeWidth = local_adaptive_large_basis_width(settings);
    Astat.deflat_requested_dim = struct( ...
        'small',smallWidth,'large',largeWidth, ...
        'combined',smallWidth+largeWidth, ...
        'adaptive_large',adaptiveLargeWidth, ...
        'adaptive_combined',smallWidth+adaptiveLargeWidth);
    Astat.deflat_tail_dim = struct();
    Astat.small_basis_source = settings.smallSource;
    Astat.small_basis_built_step = [];
    Astat.shared_large_basis_built_step = [];
    Astat.small_basis_info = {};
    Astat.small_basis_ritz_values = {};
    Astat.sketch_oversampling = settings.oversampling;
    Astat.chol_built_step = [];
    Astat.dense_materialized_step = [];
    Astat.exact_dense_diagnostics = logical(settings.exactDenseDiagnostics);
    Astat.plot_extreme_eigenvalues = logical(settings.plotExtremeEigenvalues);
end

function refresh = local_refresh_intervals(params)
    refresh.small = local_refresh_value(params,'SMALL_BASIS_REFRESH', ...
        {'DEFLAT_SMALL_PREC_REFRESH','DEFLAT_PREC_REFRESH'});
    refresh.sharedLarge = local_shared_large_refresh_value(params);
    refresh.adaptive = local_refresh_value( ...
        params,'DEFLAT_ADAPTIVE_LIFT_LARGE_REFRESH', ...
        {'DEFLAT_BOTH_PREC_REFRESH','DEFLAT_PREC_REFRESH'});
end

function value = local_shared_large_refresh_value(params)
    newField = 'DEFLAT_SHARED_LARGE_REFRESH';
    legacyFields = {'DEFLAT_GAUSSIAN_LARGE_REFRESH', ...
        'DEFLAT_SEQUENTIAL_SHARED_LARGE_REFRESH', ...
        'DEFLAT_CONCATENATED_ONCE_LARGE_REFRESH', ...
        'DEFLAT_LARGE_PREC_REFRESH','DEFLAT_BOTH_PREC_REFRESH', ...
        'DEFLAT_PREC_REFRESH'};
    value = Inf;
    if isfield(params,newField) && ~isempty(params.(newField))
        value = params.(newField);
        local_validate_refresh_value(value,newField);
    end

    legacyValues = [];
    legacyNames = {};
    for index = 1:numel(legacyFields)
        field = legacyFields{index};
        if isfield(params,field) && ~isempty(params.(field))
            legacyValue = params.(field);
            local_validate_refresh_value(legacyValue,field);
            legacyValues(end+1) = legacyValue; %#ok<AGROW>
            legacyNames{end+1} = field; %#ok<AGROW>
        end
    end
    if isempty(legacyValues), return; end

    commonLegacyValue = legacyValues(1);
    if any(legacyValues ~= commonLegacyValue)
        error('solve_varvisc_schur_sequence:conflictingLargeRefresh', ...
              ['Legacy large-basis refresh fields must agree now that the ', ...
               'original-operator large basis is shared; got %s.'], ...
              strjoin(legacyNames,', '));
    end
    if ~isinf(value) && value ~= commonLegacyValue
        error('solve_varvisc_schur_sequence:conflictingLargeRefresh', ...
              ['%s conflicts with the legacy large-basis refresh value. ', ...
               'Specify only %s.'],newField,newField);
    end
    if isinf(value), value = commonLegacyValue; end
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
    local_validate_refresh_value(value,newField);
end

function local_validate_refresh_value(value,field)
    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        ((isinf(value) && value > 0) || ...
         (isfinite(value) && value >= 1 && value == floor(value)));
    if ~valid
        error('solve_varvisc_schur_sequence:badDeflatRefresh', ...
              '%s must be a positive integer or Inf.',field);
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

function needed = local_large_needed(variants)
    designs = {variants.design};
    needed = any(~strcmp(designs,'shared_small'));
end

function needed = local_shared_large_needed(variants)
    designs = {variants.design};
    needed = any(ismember(designs,{'gaussian_large', ...
        'sequential_shared_subspace','concatenated_once'}));
end

function width = local_small_basis_width(settings,needed)
    if ~needed
        width = 0;
    elseif strcmp(settings.smallSource,'inverse_gaussian')
        width = ceil(settings.oversampling*settings.smEig);
    else
        width = settings.smEig;
    end
end

function width = local_large_basis_width(settings,needed)
    if needed
        width = ceil(settings.oversampling*settings.lgEig);
    else
        width = 0;
    end
end

function width = local_adaptive_large_basis_width(settings)
    width = ceil(settings.oversampling*(settings.smEig+settings.lgEig));
end

function enabled = local_design_enabled(variants,design)
    enabled = any(strcmp({variants.design},design));
end

function state = local_empty_small_state()
    state = struct('V',[],'theta',[],'liftTau',NaN,'directTau',NaN, ...
                   'info',struct(),'basisBuildStep',NaN);
end

function states = local_empty_arm_states()
    combined = struct('V',[],'tau1',NaN,'tau2',NaN, ...
                      'basisBuildStep',NaN);
    adaptive = struct('V',[],'tau1',NaN,'tau2',NaN, ...
                      'basisBuildStep',NaN,'largeBuildStep',NaN);
    states.sequential_shared_subspace = combined;
    states.concatenated_once = combined;
    states.adaptive_small_lift_large = adaptive;
end

function smallState = local_build_small_state( ...
        Sapply,currentInv,spectrum,nS,kSmall,settings,step)
    switch settings.smallSource
        case 'lanczos'
            computedRank = min(kSmall+1,nS);
            options = struct('maxSteps',nS, ...
                'checkEvery',settings.lanczosCheckEvery, ...
                'tolerance',settings.lanczosTolerance, ...
                'operatorNorm',spectrum.lambdaMax, ...
                'dimension',nS);
            [Vall,thetaAll,info] = ...
                fully_reorthogonalized_lanczos_smallest( ...
                Sapply,computedRank,options);
            V = Vall(:,1:kSmall);
            theta = thetaAll(1:kSmall);
            info.computedRank = computedRank;
        case 'inverse_gaussian'
            [V,info] = gaussian_subspace_basis( ...
                currentInv,nS,settings.smEig,settings.smallQ, ...
                settings.oversampling);
            theta = [];
    end

    E = V'*Sapply(V);
    E = (E+E')/2;
    lambdaHat = min(real(eig(E)));
    denominator = spectrum.lambdaMax-lambdaHat;
    if ~(lambdaHat > 0 && denominator > 0)
        error('solve_varvisc_schur_sequence:badLiftTau', ...
              'Cannot form a positive dynamic lift tau at step %d.',step);
    end
    liftTau = lambdaHat/denominator;
    if ~isempty(settings.liftTauOverride)
        liftTau = settings.liftTauOverride;
    end
    directTau = spectrum.lambdaMax;
    if ~isempty(settings.tauOverride), directTau = settings.tauOverride; end
    info.lambdaHat = lambdaHat;
    info.liftedLambdaHat = (1+1/liftTau)*lambdaHat;
    info.lambdaMax = spectrum.lambdaMax;
    info.dynamicLiftTau = lambdaHat/denominator;
    info.usedFixedLiftTau = ~isempty(settings.liftTauOverride);
    smallState = struct('V',V,'theta',theta,'liftTau',liftTau, ...
        'directTau',directTau,'info',info,'basisBuildStep',step);
end

function state = local_empty_shared_large_state()
    state = struct('V',[],'tau',NaN,'basisBuildStep',NaN);
end

function [state,refreshed,Astat] = local_refresh_shared_large_state( ...
        state,Astat,Sapply,nS,cutoffs,settings,refresh,variants,step)
    refreshed = false;
    if ~local_shared_large_needed(variants) || ...
            ~(isempty(state.V) || local_refresh_due(step,refresh.sharedLarge))
        return
    end
    [state.V,~] = gaussian_subspace_basis( ...
        Sapply,nS,settings.lgEig,settings.largeQ,settings.oversampling);
    state.tau = cutoffs.lambdaHi;
    state.basisBuildStep = step;
    refreshed = true;
    Astat.shared_large_basis_built_step(end+1) = step;
end

function [states,Astat] = local_refresh_arm_states( ...
        sharedLargeState,states,Astat,Sapply,nS,smallState,smallRefreshed, ...
        sharedLargeRefreshed,cutoffs,settings,refresh,variants,step)
    if local_design_enabled(variants,'sequential_shared_subspace')
        if sharedLargeRefreshed || smallRefreshed || ...
                isempty(states.sequential_shared_subspace.V)
            states.sequential_shared_subspace.V = local_combine_bases( ...
                sharedLargeState.V,smallState.V);
            states.sequential_shared_subspace.tau1 = cutoffs.lambdaHi;
            states.sequential_shared_subspace.tau2 = cutoffs.tauStar;
            states.sequential_shared_subspace.basisBuildStep = step;
        end
    end

    if local_design_enabled(variants,'concatenated_once')
        if sharedLargeRefreshed || smallRefreshed || ...
                isempty(states.concatenated_once.V)
            states.concatenated_once.V = local_combine_bases( ...
                sharedLargeState.V,smallState.V);
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
            transformedApply = @(X) liftHalf(Sapply(liftHalf(X)));
            [Vlarge,~] = gaussian_subspace_basis( ...
                transformedApply,nS,settings.smEig+settings.lgEig, ...
                settings.liftLargeQ, ...
                settings.oversampling,true);
            states.adaptive_small_lift_large.V = Vlarge;
            states.adaptive_small_lift_large.tau1 = cutoffs.tauStar;
            states.adaptive_small_lift_large.tau2 = smallState.liftTau;
            states.adaptive_small_lift_large.basisBuildStep = step;
            states.adaptive_small_lift_large.largeBuildStep = step;
        end
    end

    Astat.deflat_tail_dim.shared_small = size(smallState.V,2);
    Astat.deflat_tail_dim.shared_large = size(sharedLargeState.V,2);
    Astat.deflat_tail_dim.gaussian_large = size(sharedLargeState.V,2);
    Astat.deflat_tail_dim.sequential_large = size(sharedLargeState.V,2);
    Astat.deflat_tail_dim.concatenated_large = size(sharedLargeState.V,2);
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

function [Papply,spectralApply,Vdim,largeDim,tauValues,basisStep,largeStep] = ...
        local_make_preconditioner( ...
        design,Sapply,smallState,sharedLargeState,states,needSpectralApply)
    tauValues = struct('first',NaN,'second',NaN);
    largeDim = 0;
    spectralApply = [];
    switch design
        case 'shared_small'
            V = smallState.V;
            tauValues.first = smallState.directTau;
            Papply = src.precond.deflation_P_apply( ...
                V,Sapply,tauValues.first,'handle',0);
            if needSpectralApply
                Phalf = src.precond.deflation_Psqrt_apply( ...
                    V,Sapply,tauValues.first,'handle');
                spectralApply = @(X) Phalf(Sapply(Phalf(X)));
            end
            Vdim = size(V,2);
            basisStep = smallState.basisBuildStep;
            largeStep = NaN;
        case 'gaussian_large'
            tauValues.first = sharedLargeState.tau;
            Papply = src.precond.deflation_P_apply( ...
                sharedLargeState.V,Sapply,tauValues.first,'handle',0);
            if needSpectralApply
                Phalf = src.precond.deflation_Psqrt_apply( ...
                    sharedLargeState.V,Sapply,tauValues.first,'handle');
                spectralApply = @(X) Phalf(Sapply(Phalf(X)));
            end
            Vdim = size(sharedLargeState.V,2); largeDim = Vdim;
            basisStep = sharedLargeState.basisBuildStep;
            largeStep = sharedLargeState.basisBuildStep;
        case 'sequential_shared_subspace'
            state = states.sequential_shared_subspace;
            tauValues.first = state.tau1;
            tauValues.second = state.tau2;
            P1half = src.precond.deflation_Psqrt_apply( ...
                state.V,Sapply,tauValues.first,'handle');
            S1apply = @(X) P1half(Sapply(P1half(X)));
            P2apply = src.precond.deflation_P_apply( ...
                state.V,S1apply,tauValues.second,'handle',0);
            Papply = @(X) P1half(P2apply(P1half(X)));
            if needSpectralApply
                P2half = src.precond.deflation_Psqrt_apply( ...
                    state.V,S1apply,tauValues.second,'handle');
                spectralApply = @(X) P2half(S1apply(P2half(X)));
            end
            Vdim = size(state.V,2);
            largeDim = size(sharedLargeState.V,2);
            basisStep = state.basisBuildStep;
            largeStep = sharedLargeState.basisBuildStep;
        case 'concatenated_once'
            state = states.concatenated_once;
            tauValues.first = state.tau1;
            Papply = src.precond.deflation_P_apply( ...
                state.V,Sapply,tauValues.first,'handle',0);
            if needSpectralApply
                Phalf = src.precond.deflation_Psqrt_apply( ...
                    state.V,Sapply,tauValues.first,'handle');
                spectralApply = @(X) Phalf(Sapply(Phalf(X)));
            end
            Vdim = size(state.V,2);
            largeDim = size(sharedLargeState.V,2);
            basisStep = state.basisBuildStep;
            largeStep = sharedLargeState.basisBuildStep;
        case 'adaptive_small_lift_large'
            state = states.adaptive_small_lift_large;
            tauValues.first = state.tau2;
            tauValues.second = state.tau1;
            [~,liftHalf] = small_eigenvalue_lift( ...
                smallState.V,tauValues.first);
            transformedApply = @(X) liftHalf(Sapply(liftHalf(X)));
            P2apply = src.precond.deflation_P_apply( ...
                state.V,transformedApply,tauValues.second,'handle',0);
            Papply = @(X) liftHalf(P2apply(liftHalf(X)));
            if needSpectralApply
                P2half = src.precond.deflation_Psqrt_apply( ...
                    state.V,transformedApply,tauValues.second,'handle');
                spectralApply = @(X) P2half( ...
                    transformedApply(P2half(X)));
            end
            largeDim = size(state.V,2);
            Vdim = size(smallState.V,2)+largeDim;
            basisStep = state.basisBuildStep;
            largeStep = state.largeBuildStep;
    end
end

function Astat = local_record_extreme_spectrum( ...
        Astat,key,step,Aapply,nS,settings)
    try
        isExact = false;
        if nS <= 2
            [lambdaMin,lambdaMax,residual] = ...
                local_dense_extreme_spectrum(Aapply,nS);
            isExact = true;
        else
            options = struct('issym',true,'isreal',true, ...
                'tol',settings.spectralTolerance, ...
                'maxit',settings.spectralMaxit, ...
                'v0',ones(nS,1)/sqrt(nS));
            [largeVector,largeValue,largeFlag] = eigs( ...
                Aapply,nS,1,'largestreal',options);
            lambdaMax = real(largeValue(1,1));
            [smallVector,smallValues,smallFlag] = ...
                local_smallest_abs_ritz( ...
                Aapply,nS,1,lambdaMax,options,settings);
            lambdaMin = smallValues(1);
            flag = max(smallFlag,largeFlag);
            smallImage = Aapply(smallVector);
            largeImage = Aapply(largeVector);
            smallResidual = norm(smallImage-lambdaMin*smallVector) / ...
                max(lambdaMax,eps);
            largeResidual = norm(largeImage-lambdaMax*largeVector) / ...
                max(lambdaMax,eps);
            residual = max(smallResidual,largeResidual);
            residualLimit = max(100*settings.spectralTolerance,1e-6);
            if ~isfinite(residual) || residual > residualLimit
                error('solve_varvisc_schur_sequence:badRitzResidual', ...
                    ['Extreme Ritz solve returned flag %d and ', ...
                     'residual %.3e.'],flag,residual);
            end
        end
        if ~isfinite(lambdaMin) || ~isfinite(lambdaMax) || lambdaMin <= 0
            error('solve_varvisc_schur_sequence:badExtremeSpectrum', ...
                ['Invalid extreme eigenvalue estimates for %s: ', ...
                 'lambda_min=%.16g, lambda_max=%.16g.'], ...
                key,lambdaMin,lambdaMax);
        end
        Astat = local_store_extreme_spectrum( ...
            Astat,key,step,lambdaMin,lambdaMax,0,residual,isExact);
    catch ME
        Astat = local_store_extreme_spectrum( ...
            Astat,key,step,NaN,NaN,-1,NaN,false);
        warning('solve_varvisc_schur_sequence:extremeSpectrumFailed', ...
            ['Extreme-eigenvalue estimation failed for %s at step %d: ', ...
             '%s'],key,step,ME.message);
    end
end

function [vectors,values,flag] = local_smallest_abs_ritz( ...
        Aapply,nS,count,lambdaMax,options,settings)
    shiftedApply = @(X) lambdaMax*X-Aapply(X);
    [vectors,shiftedValues,flag] = eigs( ...
        shiftedApply,nS,count,'largestabs',options);
    [values,order] = sort( ...
        lambdaMax-real(diag(shiftedValues)),'ascend');
    vectors = vectors(:,order);
    if flag ~= 0 || any(~isfinite(values)) || any(values <= 0)
        lanczosOptions = struct( ...
            'maxSteps',min(nS,settings.spectralMaxit), ...
            'checkEvery',settings.lanczosCheckEvery, ...
            'tolerance',max(settings.spectralTolerance,1e-8), ...
            'operatorNorm',lambdaMax,'dimension',nS);
        [vectors,values] = fully_reorthogonalized_lanczos_smallest( ...
            Aapply,count,lanczosOptions);
        flag = 0;
    end
end

function [lambdaMin,lambdaMax,residual] = ...
        local_dense_extreme_spectrum(Aapply,nS)
    denseOperator = Aapply(eye(nS));
    residual = norm(denseOperator-denseOperator','fro') / ...
        max(norm(denseOperator,'fro'),eps);
    denseOperator = (denseOperator+denseOperator')/2;
    values = sort(real(eig(denseOperator)),'ascend');
    lambdaMin = values(1);
    lambdaMax = values(end);
end

function Astat = local_store_extreme_spectrum( ...
        Astat,key,step,lambdaMin,lambdaMax,flag,residual,isExact)
    if nargin < 7, residual = NaN; end
    if nargin < 8, isExact = false; end
    Astat.system_lambda_min.(key)(step) = lambdaMin;
    Astat.system_lambda_max.(key)(step) = lambdaMax;
    Astat.system_kappa.(key)(step) = lambdaMax/lambdaMin;
    Astat.system_spectrum_flag.(key)(step) = flag;
    Astat.system_spectrum_residual.(key)(step) = residual;
    Astat.system_spectrum_is_exact.(key)(step) = isExact;
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
    cutoffs = struct('lambdaLo',NaN,'lambdaHi',NaN,'tauStar',NaN);
    if widths.small > 0, cutoffs.lambdaLo = ev(widths.small+1); end
    if widths.large > 0, cutoffs.lambdaHi = ev(nS-widths.large); end
    if widths.small > 0 && widths.large > 0
        cutoffs.tauStar = sqrt(cutoffs.lambdaLo*cutoffs.lambdaHi);
    end
end

function widths = local_two_tail_widths(nS,smallWidth,largeWidth)
    available = nS-1;
    if available < 1
        error('solve_varvisc_schur_sequence:schurTooSmall', ...
              'Deflation requires at least a 2-by-2 Schur matrix.');
    end
    widths = struct('small',smallWidth,'large',largeWidth);
    if smallWidth+largeWidth > available
        error('solve_varvisc_schur_sequence:deflationTooWide', ...
              ['Requested small/large basis widths %d+%d exceed the %d ', ...
               'available nonoverlapping Schur directions.'], ...
              smallWidth,largeWidth,available);
    end
end

function Astat = local_record(Astat,key,step,xscaled,fl,rr,it,rhsnorm,yref)
    Astat.solver_its.(key)(step) = it;
    Astat.solver_flag.(key)(step) = fl;
    Astat.solver_relres.(key)(step) = rr;
    x = xscaled*rhsnorm;
    Astat.solver_err.(key)(step) = norm(x-yref)/max(norm(yref),eps);
end
