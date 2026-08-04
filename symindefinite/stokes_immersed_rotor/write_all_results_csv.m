function write_all_results_csv(results_root, all_stats, geometry, keys)
%WRITE_ALL_RESULTS_CSV  Master per-(case,timestep) table for a benchmark run.
%
%   WRITE_ALL_RESULTS_CSV(RESULTS_ROOT, ALL_STATS, GEOMETRY, KEYS)
%
%   Solver columns follow the registry order (<key>_its / <key>_flag), so the
%   table extends automatically as preconditioners are added.  This file is
%   what replot_benchmark reads back, so its schema is the contract between the
%   expensive live run and cheap figure regeneration.
%
%   solver_err_last holds the last-registered solver's error against the direct
%   backslash solve -- the accuracy figure's primary curve.  It was previously
%   plotted but never stored, which is why the figures for older results
%   directories cannot show it.  The name deliberately does not end in '_its',
%   so make_paper_summary_table's endsWith(vn,'_its') solver discovery is
%   unaffected.
%
%   See also: load_benchmark_stats, make_paper_summary_table.

    nsolv = numel(keys);
    case_col = {}; ts_col = []; relres = []; diffF = [];
    bs = []; constr = []; nCc = []; errLast = [];
    its = repmat({[]}, nsolv, 1); fl = repmat({[]}, nsolv, 1);
    for k = 1:numel(all_stats)
        st = all_stats{k};
        ns = numel(st.solver_its.(keys{1}));
        case_col = [case_col; repmat({st.case_name}, ns, 1)];   %#ok<AGROW>
        ts_col   = [ts_col;   (1:ns)'];                          %#ok<AGROW>
        for s = 1:nsolv
            its{s} = [its{s}; st.solver_its.(keys{s})(:)];
            fl{s}  = [fl{s};  st.solver_flag.(keys{s})(:)];
        end
        relres = [relres; st.solver_relres.(keys{end})(:)];      %#ok<AGROW>
        diffF  = [diffF;  st.coupling_change(:)];                %#ok<AGROW>
        bs     = [bs;     st.backslash_relres(:)];               %#ok<AGROW>
        constr = [constr; st.constraint_res(:)];                 %#ok<AGROW>
        nCc    = [nCc;    st.nC(:)];                             %#ok<AGROW>
        if isfield(st, 'solver_err') && isfield(st.solver_err, keys{end})
            errLast = [errLast; st.solver_err.(keys{end})(:)];   %#ok<AGROW>
        else
            errLast = [errLast; nan(ns, 1)];                     %#ok<AGROW>
        end
    end
    geom_col = repmat({geometry}, numel(ts_col), 1);
    T = table(case_col, geom_col, ts_col, ...
        'VariableNames', {'case_name', 'geometry', 'timestep'});
    for s = 1:nsolv
        T.([keys{s} '_its'])  = its{s};
        T.([keys{s} '_flag']) = fl{s};
    end
    T.relres = relres; T.diffF = diffF; T.backslash_relres = bs;
    T.constraint_res = constr; T.nC = nCc; T.solver_err_last = errLast;
    writetable(T, fullfile(results_root, 'all_results.csv'));
    fprintf('Wrote %s\n', fullfile(results_root, 'all_results.csv'));
end
