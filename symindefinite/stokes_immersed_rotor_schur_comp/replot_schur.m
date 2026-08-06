function replot_schur(results_root)
%REPLOT_SCHUR  Redraw every figure from all_results.csv, without re-solving.
%   REPLOT_SCHUR()               uses the default results root
%   REPLOT_SCHUR(RESULTS_ROOT)
%
%   A full run costs several minutes; a style or label change should not.  This
%   rebuilds the per-case Astat structs from the CSV plus run_config.mat and
%   replays the same writers the driver uses, so live and replotted figures are
%   produced by identical code.
%
%   See also: run_schur_recycle, write_schur_case_outputs, write_schur_summary.

    paths = add_schur_paths();
    assert_local_helpers();
    if nargin < 1 || isempty(results_root)
        results_root = paths.outDir;
    end

    csv = fullfile(results_root, 'all_results.csv');
    if ~exist(csv, 'file')
        error('replot_schur:noCsv', 'No all_results.csv under %s.', results_root);
    end
    T = readtable(csv);

    cfgfile = fullfile(results_root, 'run_config.mat');
    labels = {};
    if exist(cfgfile, 'file')
        S = load(cfgfile, 'cfg_dump');
        if isfield(S.cfg_dump, 'solver_labels')
            labels = S.cfg_dump.solver_labels;
        end
    end

    % arm keys, in CSV column order
    vars = T.Properties.VariableNames;
    its_cols = vars(endsWith(vars, '_its'));
    keys = cellfun(@(c) c(1:end-4), its_cols, 'UniformOutput', false);
    if numel(labels) ~= numel(keys)
        labels = keys;                       % fall back to raw keys
    end

    cases  = unique(T.case_name, 'stable');
    opts   = benchmark_fig_defaults();
    stats  = cell(numel(cases), 1);

    for k = 1:numel(cases)
        m = strcmp(T.case_name, cases{k});
        A = struct();
        A.case_name     = cases{k};
        A.geometry      = char(T.geometry(find(m, 1)));
        A.nsteps        = sum(m);
        A.solver_keys   = keys(:);
        A.solver_labels = labels(:);
        A.solver_its    = struct();
        for i = 1:numel(keys)
            A.solver_its.(keys{i}) = T.([keys{i} '_its'])(m);
        end
        for f = {'kappa','lambda_min','lambda_max','ReldiffF','RelInitdiffF', ...
                 'InvRelDiff','LowRankInvRelDiff','coupling_change'}
            A.(f{1}) = col_or_nan(T, f{1}, m, A.nsteps);
        end
        write_schur_case_outputs(fullfile(results_root, cases{k}), A, opts);
        stats{k} = A;
    end

    write_schur_summary(results_root, stats, opts);
    fprintf('[replot_schur] redrew %d case(s) under %s\n', numel(cases), results_root);
end

%==========================================================================
function v = col_or_nan(T, name, mask, n)
    if ismember(name, T.Properties.VariableNames)
        v = T.(name)(mask);
    else
        v = nan(n, 1);
    end
end
