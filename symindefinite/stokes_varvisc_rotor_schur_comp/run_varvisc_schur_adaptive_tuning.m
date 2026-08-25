function result = run_varvisc_schur_adaptive_tuning(options)
%RUN_VARVISC_SCHUR_ADAPTIVE_TUNING  Tune the adaptive two-tail deflator.
%   RESULT = RUN_VARVISC_SCHUR_ADAPTIVE_TUNING() screens lift strength and
%   transformed power count on the coarse
%   checkerboard-drift problem, then evaluates six finalists on the full
%   h0=0.05, 60-step problem. The winner minimizes the worst per-step
%   preconditioned condition number; settings within one percent of the best
%   condition number are ordered by total PCG iterations.
%
%   RESULT = RUN_VARVISC_SCHUR_ADAPTIVE_TUNING(OPTIONS) accepts:
%     smoke            - small, fast integration run (default false)
%     outputDir        - output directory (default adaptive_tuning[_smoke])
%     seed             - RNG seed reset before every candidate (default 1)
%     tauMultipliers   - fixed multiples of the step-1 dynamic tau
%     qList            - transformed power counts (default [0 1 2])
%     finalistCount    - number promoted to the fine stage (default 6)
%
%   The production sweep preserves the effective 20-small/140-post-lift basis and
%   keeps every small and large basis frozen (refresh interval Inf). Smoke
%   mode uses a 2-small/4-post-lift basis solely to test the workflow.

    if nargin < 1 || isempty(options), options = struct(); end
    paths = add_varvisc_schur_paths();
    options = local_options(options,paths);
    if ~exist(options.outputDir,'dir'), mkdir(options.outputDir); end

    fprintf('[adaptive tuning] estimating the reference dynamic lift tau ...\n');
    probeParams = local_stage_params(options,'coarse');
    probeParams.max_steps = 1;
    probeParams.lift_tau = [];
    probeParams.PLOT_EXTREME_EIGENVALUES = false;
    rng(options.seed,'twister');
    probeCfg = varvisc_schur_make_cfg( ...
        'disk_static_nu_checkerboard_shift',probeParams,[]);
    probeStats = solve_varvisc_schur_sequence(probeCfg,probeParams,'');
    dynamicTau = probeStats.small_basis_info{1}.dynamicLiftTau;

    configurations = local_configurations(options,dynamicTau);
    fprintf('[adaptive tuning] coarse screen: %d configurations ...\n', ...
        numel(configurations));
    coarseParams = local_stage_params(options,'coarse');
    [coarseSummary,coarseSteps] = local_run_grid( ...
        configurations,coarseParams,options.seed,'coarse');
    writetable(coarseSummary,fullfile(options.outputDir,'coarse_summary.csv'));
    writetable(coarseSteps,fullfile(options.outputDir,'coarse_timesteps.csv'));

    eligible = local_eligible(coarseSummary);
    if ~any(eligible)
        error('run_varvisc_schur_adaptive_tuning:noCoarseCandidate', ...
              'No coarse candidate converged with a valid spectrum.');
    end
    orderedCoarse = sortrows(coarseSummary(eligible,:), ...
        {'worst_kappa','total_iterations'});
    finalistCount = min(options.finalistCount,height(orderedCoarse));
    finalistIds = orderedCoarse.config_id(1:finalistCount);
    finalistMask = ismember([configurations.id],finalistIds);
    finalists = configurations(finalistMask);

    fprintf('[adaptive tuning] fine evaluation: %d finalists ...\n', ...
        numel(finalists));
    fineParams = local_stage_params(options,'fine');
    [fineSummary,fineSteps] = local_run_grid( ...
        finalists,fineParams,options.seed,'fine');
    writetable(fineSummary,fullfile(options.outputDir,'fine_summary.csv'));
    writetable(fineSteps,fullfile(options.outputDir,'fine_timesteps.csv'));

    winnerRow = local_select_winner(fineSummary);
    winner = configurations([configurations.id] == winnerRow.config_id);
    [comparisonSummary,comparisonSteps] = local_run_comparison( ...
        winner,fineParams,options.seed);
    writetable(comparisonSummary, ...
        fullfile(options.outputDir,'comparison_summary.csv'));
    writetable(comparisonSteps, ...
        fullfile(options.outputDir,'comparison_timesteps.csv'));

    recommendation = local_recommendation(winner,winnerRow,dynamicTau,options);
    local_write_json(fullfile(options.outputDir,'recommended_config.json'), ...
        recommendation);
    save(fullfile(options.outputDir,'adaptive_tuning.mat'), ...
        'options','dynamicTau','configurations','coarseSummary','coarseSteps', ...
        'fineSummary','fineSteps','comparisonSummary','comparisonSteps', ...
        'recommendation');

    result = struct('recommendation',recommendation, ...
        'coarseSummary',coarseSummary,'fineSummary',fineSummary, ...
        'comparisonSummary',comparisonSummary, ...
        'outputDir',options.outputDir);
    fprintf(['[adaptive tuning] winner %s: worst kappa %.6g, ', ...
        'total iterations %d -> %s\n'],winner.label,winnerRow.worst_kappa, ...
        winnerRow.total_iterations,options.outputDir);
end

function options = local_options(options,paths)
    hasTauMultipliers = isfield(options,'tauMultipliers') && ...
        ~isempty(options.tauMultipliers);
    hasQList = isfield(options,'qList') && ~isempty(options.qList);
    options.smoke = logical(local_option(options,'smoke',false));
    options.seed = local_option(options,'seed',1);
    if options.smoke
        if ~hasTauMultipliers, options.tauMultipliers = 1; end
        if ~hasQList, options.qList = [0 1]; end
    else
        if ~hasTauMultipliers
            options.tauMultipliers = [0.01 0.1 1 10 100];
        end
        if ~hasQList, options.qList = [0 1 2]; end
    end
    options.finalistCount = local_option(options,'finalistCount',6);
    defaultName = 'adaptive_tuning';
    if options.smoke, defaultName = 'adaptive_tuning_smoke'; end
    options.outputDir = char(local_option( ...
        options,'outputDir',fullfile(paths.thisDir,defaultName)));
    validateattributes(options.seed,{'numeric'}, ...
        {'scalar','integer','nonnegative'},mfilename,'options.seed');
    validateattributes(options.tauMultipliers,{'numeric'}, ...
        {'vector','real','finite','positive'},mfilename, ...
        'options.tauMultipliers');
    validateattributes(options.qList,{'numeric'}, ...
        {'vector','integer','nonnegative'},mfilename,'options.qList');
    validateattributes(options.finalistCount,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'options.finalistCount');
    if options.smoke, options.finalistCount = min(options.finalistCount,2); end
end

function value = local_option(options,name,defaultValue)
    value = defaultValue;
    if isfield(options,name) && ~isempty(options.(name))
        value = options.(name);
    end
end

function params = local_stage_params(options,stage)
    params = make_varvisc_schur_params();
    params.skip_unprecond = true;
    params.COMPUTE_SPECTRUM = false;
    params.PLOT_EXTREME_EIGENVALUES = true;
    params.EXACT_DENSE_DIAGNOSTICS = false;
    params.standalone_variants = struct( ...
        'name','deflate_adaptive_small_lift_large', ...
        'design','adaptive_small_lift_large');
    if options.smoke
        params.h0 = 0.2;
        params.Tstep = 3;
        params.max_steps = 2;
        params.sm_eig = 2;
        params.lg_eig = 2;
        params.sketch_oversampling = 1;
        params.SPECTRAL_RITZ_TOL = 1e-8;
        params.SPECTRAL_RITZ_MAXIT = 500;
    elseif strcmp(stage,'coarse')
        params.h0 = 0.1;
        params.Tstep = 13;
        params.max_steps = 12;
    else
        params.h0 = 0.05;
        params.Tstep = 61;
        params.max_steps = 60;
    end
end

function configurations = local_configurations(options,dynamicTau)
    tauModes = struct('label',{},'mode',{},'value',{},'multiplier',{});
    tauModes(end+1) = struct( ...
        'label','dynamic','mode','dynamic','value',NaN,'multiplier',NaN);
    [~,multiplierOrder] = sort(abs(log10(options.tauMultipliers)));
    for multiplier = options.tauMultipliers(multiplierOrder)
        tauModes(end+1) = struct( ...
            'label',sprintf('fixed_%g_x_dynamic',multiplier), ...
            'mode','fixed','value',multiplier*dynamicTau, ...
            'multiplier',multiplier); %#ok<AGROW>
    end
    tauModes(end+1) = struct( ...
        'label','current_1e-10','mode','fixed','value',1e-10, ...
        'multiplier',1e-10/dynamicTau);

    configurations = struct('id',{},'label',{},'tauMode',{}, ...
        'tauValue',{},'tauMultiplier',{},'q',{});
    id = 0;
    for tauIndex = 1:numel(tauModes)
        for q = sort(options.qList,'descend')
            id = id+1;
            configurations(end+1) = struct( ...
                'id',id, ...
                'label',sprintf('%s_q%d_RInf', ...
                    tauModes(tauIndex).label,q), ...
                'tauMode',tauModes(tauIndex).mode, ...
                'tauValue',tauModes(tauIndex).value, ...
                'tauMultiplier',tauModes(tauIndex).multiplier, ...
                'q',q); %#ok<AGROW>
        end
    end
end

function [summaryTable,stepTable] = local_run_grid( ...
        configurations,baseParams,seed,stage)
    summaries = cell(numel(configurations),1);
    allSteps = cell(numel(configurations),1);
    for index = 1:numel(configurations)
        config = configurations(index);
        fprintf('  [%s %d/%d] %s\n', ...
            stage,index,numel(configurations),config.label);
        params = local_apply_config(baseParams,config);
        rng(seed,'twister');
        cfg = varvisc_schur_make_cfg( ...
            'disk_static_nu_checkerboard_shift',params,[]);
        stats = solve_varvisc_schur_sequence(cfg,params,'');
        [summaries{index},allSteps{index}] = ...
            local_adaptive_tables(stats,config,stage);
    end
    summaryTable = struct2table([summaries{:}]);
    stepTable = vertcat(allSteps{:});
end

function params = local_apply_config(params,config)
    if strcmp(config.tauMode,'dynamic')
        params.lift_tau = [];
    else
        params.lift_tau = config.tauValue;
    end
    params.lift_large_q = config.q;
    params.SMALL_BASIS_REFRESH = Inf;
    params.DEFLAT_ADAPTIVE_LIFT_LARGE_REFRESH = Inf;
end

function [summary,steps] = local_adaptive_tables(stats,config,stage)
    key = 'deflate_adaptive_small_lift_large';
    lambdaMin = stats.system_lambda_min.(key);
    lambdaMax = stats.system_lambda_max.(key);
    kappa = stats.system_kappa.(key);
    iterations = stats.solver_its.(key);
    solveFlag = stats.solver_flag.(key);
    spectrumFlag = stats.system_spectrum_flag.(key);
    residual = stats.system_spectrum_residual.(key);
    isExact = stats.system_spectrum_is_exact.(key);
    expectedSmall = stats.deflat_requested_dim.small;
    expectedLarge = stats.deflat_requested_dim.adaptive_large;
    if any(stats.small_basis_dim_history ~= expectedSmall) || ...
            any(stats.large_basis_dim_history.(key) ~= expectedLarge)
        error('run_varvisc_schur_adaptive_tuning:dimensionChanged', ...
              'Candidate %s did not preserve the requested basis widths.', ...
              config.label);
    end
    validSpectrum = all(spectrumFlag == 0) && ...
        all(isfinite(lambdaMin)) && all(isfinite(lambdaMax)) && ...
        all(lambdaMin > 0) && all(lambdaMin <= lambdaMax);
    summary = struct('stage',string(stage),'config_id',config.id, ...
        'label',string(config.label),'tau_mode',string(config.tauMode), ...
        'tau_value',config.tauValue,'tau_multiplier',config.tauMultiplier, ...
        'lift_large_q',config.q,'refresh_interval',Inf, ...
        'small_dim',expectedSmall,'large_dim',expectedLarge, ...
        'worst_kappa',max(kappa),'min_lambda',min(lambdaMin), ...
        'max_lambda',max(lambdaMax), ...
        'total_iterations',sum(iterations), ...
        'max_iterations',max(iterations), ...
        'all_converged',all(solveFlag == 0), ...
        'valid_spectrum',validSpectrum, ...
        'max_solution_error',max(stats.solver_err.(key)), ...
        'max_spectrum_residual',max(residual), ...
        'dense_fallback_count',sum(isExact));
    nsteps = stats.nsteps;
    steps = table(repmat(string(stage),nsteps,1), ...
        repmat(config.id,nsteps,1),repmat(string(config.label),nsteps,1), ...
        (1:nsteps)',lambdaMin,lambdaMax,kappa,iterations,solveFlag, ...
        stats.solver_err.(key),spectrumFlag,residual,isExact,stats.lift_tau, ...
        'VariableNames',{'stage','config_id','label','timestep', ...
        'lambda_min','lambda_max','kappa_prec','pcg_iterations', ...
        'pcg_flag','solution_error','spectrum_flag','spectrum_residual', ...
        'spectrum_is_exact','lift_tau_used'});
end

function eligible = local_eligible(summary)
    eligible = summary.all_converged & summary.valid_spectrum & ...
        isfinite(summary.worst_kappa) & summary.max_solution_error < 1e-5;
end

function winner = local_select_winner(summary)
    eligible = local_eligible(summary);
    if ~any(eligible)
        error('run_varvisc_schur_adaptive_tuning:noFineCandidate', ...
              'No fine candidate converged with a valid spectrum.');
    end
    candidates = summary(eligible,:);
    bestKappa = min(candidates.worst_kappa);
    candidates = candidates(candidates.worst_kappa <= 1.01*bestKappa,:);
    candidates = sortrows(candidates,{'total_iterations','worst_kappa'});
    winner = candidates(1,:);
end

function [summary,steps] = local_run_comparison(winner,baseParams,seed)
    params = local_apply_config(baseParams,winner);
    params.q = 1;
    params.DEFLAT_SHARED_LARGE_REFRESH = Inf;
    params.standalone_variants = [ ...
        struct('name','deflate_adaptive_small_lift_large', ...
               'design','adaptive_small_lift_large'), ...
        struct('name','deflate_sequential_shared_subspace', ...
               'design','sequential_shared_subspace'), ...
        struct('name','deflate_concatenated_once', ...
               'design','concatenated_once')];
    rng(seed,'twister');
    cfg = varvisc_schur_make_cfg( ...
        'disk_static_nu_checkerboard_shift',params,[]);
    stats = solve_varvisc_schur_sequence(cfg,params,'');
    keys = {params.standalone_variants.name};
    rows = cell(numel(keys),1); stepCells = cell(numel(keys),1);
    for index = 1:numel(keys)
        key = keys{index};
        lambdaMin = stats.system_lambda_min.(key);
        lambdaMax = stats.system_lambda_max.(key);
        kappa = stats.system_kappa.(key);
        rows{index} = struct('variant',string(key), ...
            'worst_kappa',max(kappa),'min_lambda',min(lambdaMin), ...
            'max_lambda',max(lambdaMax), ...
            'total_iterations',sum(stats.solver_its.(key)), ...
            'max_iterations',max(stats.solver_its.(key)), ...
            'all_converged',all(stats.solver_flag.(key) == 0), ...
            'max_solution_error',max(stats.solver_err.(key)), ...
            'max_spectrum_residual', ...
                max(stats.system_spectrum_residual.(key)));
        nsteps = stats.nsteps;
        stepCells{index} = table(repmat(string(key),nsteps,1), ...
            (1:nsteps)',lambdaMin,lambdaMax,kappa,stats.solver_its.(key), ...
            stats.solver_flag.(key),stats.solver_err.(key), ...
            stats.system_spectrum_flag.(key), ...
            stats.system_spectrum_residual.(key), ...
            'VariableNames',{'variant','timestep','lambda_min', ...
            'lambda_max','kappa_prec','pcg_iterations','pcg_flag', ...
            'solution_error','spectrum_flag','spectrum_residual'});
    end
    summary = struct2table([rows{:}]);
    steps = vertcat(stepCells{:});
end

function recommendation = local_recommendation( ...
        winner,winnerRow,dynamicTau,options)
    recommendation = struct();
    recommendation.case_name = 'disk_static_nu_checkerboard_shift';
    recommendation.selection_rule = ...
        'worst preconditioned kappa, then total PCG iterations within 1 percent';
    recommendation.config_id = winner.id;
    recommendation.label = winner.label;
    recommendation.lift_tau_mode = winner.tauMode;
    if strcmp(winner.tauMode,'dynamic')
        recommendation.lift_tau = [];
    else
        recommendation.lift_tau = winner.tauValue;
    end
    recommendation.step1_dynamic_tau = dynamicTau;
    recommendation.lift_large_q = winner.q;
    recommendation.small_basis_refresh = 'Inf';
    recommendation.adaptive_large_refresh = 'Inf';
    recommendation.small_dim = winnerRow.small_dim;
    recommendation.large_dim = winnerRow.large_dim;
    recommendation.worst_kappa = winnerRow.worst_kappa;
    recommendation.min_lambda = winnerRow.min_lambda;
    recommendation.max_lambda = winnerRow.max_lambda;
    recommendation.total_iterations = winnerRow.total_iterations;
    recommendation.smoke = options.smoke;
end

function local_write_json(filename,value)
    fid = fopen(filename,'w');
    if fid < 0
        error('run_varvisc_schur_adaptive_tuning:cannotWrite', ...
              'Cannot open %s for writing.',filename);
    end
    cleanupFile = onCleanup(@() fclose(fid));
    fprintf(fid,'%s',jsonencode(value,'PrettyPrint',true));
end
