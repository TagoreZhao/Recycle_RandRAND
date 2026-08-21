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
        for i = 1:numel(keys), A.solver_its.(keys{i}) = T.(itscols{i})(mask); end
        for i = 1:numel(diagnostics)
            f = diagnostics{i};
            if ismember(f,vars), A.(f) = T.(f)(mask); else, A.(f) = nan(A.nsteps,1); end
        end
        write_varvisc_schur_case_outputs(fullfile(results_root,cases{ci}),A,opts);
        stats{ci} = A;
    end
    write_varvisc_schur_summary(results_root,stats,opts);
end
