function replot_varvisc_schur(results_root)
%REPLOT_VARVISC_SCHUR  Redraw benchmark figures from all_results.csv.
    paths = add_varvisc_schur_paths();
    if nargin < 1 || isempty(results_root), results_root = paths.outDir; end
    T = readtable(fullfile(results_root,'all_results.csv'));
    vars = T.Properties.VariableNames;
    itscols = vars(endsWith(vars,'_its'));
    keys = cellfun(@(x) x(1:end-4),itscols,'UniformOutput',false);
    labels = keys;
    cfgfile = fullfile(results_root,'run_config.mat');
    if exist(cfgfile,'file')
        L = load(cfgfile,'cfg_dump');
        if isfield(L.cfg_dump,'solver_labels'), labels = L.cfg_dump.solver_labels; end
    end
    cases = unique(T.case_name,'stable'); stats = cell(numel(cases),1);
    diagnostics = {'kappa','lambda_min','lambda_max','ReldiffF', ...
        'RelInitdiffF','ReldiffProbe','RelInitdiffProbe','InvRelDiff', ...
        'coupling_change','A_change','D_change','pressure_schur_change', ...
        'pressure_schur_change_probe','nu_contrast'};
    opts = varvisc_schur_fig_defaults();
    for ci = 1:numel(cases)
        mask = strcmp(T.case_name,cases{ci});
        A = struct('case_name',cases{ci},'geometry',char(T.geometry(find(mask,1))), ...
            'nsteps',sum(mask),'solver_keys',{keys(:)},'solver_labels',{labels(:)});
        A.solver_its = struct();
        A.system_lambda_min = struct();
        A.system_lambda_max = struct();
        A.system_kappa = struct();
        A.system_spectrum_flag = struct();
        A.system_spectrum_residual = struct();
        A.system_spectrum_is_exact = struct();
        for i = 1:numel(keys)
            key = keys{i};
            A.solver_its.(key) = T.(itscols{i})(mask);
            A.system_lambda_min.(key) = read_solver_column( ...
                T,vars,[key '_lambda_min'],mask,A.nsteps);
            A.system_lambda_max.(key) = read_solver_column( ...
                T,vars,[key '_lambda_max'],mask,A.nsteps);
            A.system_kappa.(key) = read_solver_column( ...
                T,vars,[key '_kappa_prec'],mask,A.nsteps);
            A.system_spectrum_flag.(key) = read_solver_column( ...
                T,vars,[key '_spectrum_flag'],mask,A.nsteps);
            A.system_spectrum_residual.(key) = read_solver_column( ...
                T,vars,[key '_spectrum_residual'],mask,A.nsteps);
            A.system_spectrum_is_exact.(key) = read_solver_column( ...
                T,vars,[key '_spectrum_is_exact'],mask,A.nsteps) == 1;
        end
        for i = 1:numel(diagnostics)
            f = diagnostics{i};
            if ismember(f,vars), A.(f) = T.(f)(mask); else, A.(f) = nan(A.nsteps,1); end
        end
        write_varvisc_schur_case_outputs(fullfile(results_root,cases{ci}),A,opts);
        stats{ci} = A;
    end
    write_varvisc_schur_summary(results_root,stats,opts);
end

function values = read_solver_column(T,vars,name,mask,nsteps)
    if ismember(name,vars)
        values = T.(name)(mask);
    else
        values = nan(nsteps,1);
    end
end
