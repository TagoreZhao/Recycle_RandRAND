function varvisc_write_all_results_csv(results_root, all_stats, geometry, keys)
%WRITE_ALL_RESULTS_CSV  Master per-(case,timestep) table for a benchmark run.
%
%   WRITE_ALL_RESULTS_CSV(RESULTS_ROOT, ALL_STATS, GEOMETRY, KEYS)
%
%   Solver columns follow the registry order (<key>_its / <key>_flag), so the
%   table extends automatically as preconditioners are added.  This file is
%   what replot_varvisc_benchmark reads back, so its schema is the contract between the
%   expensive live run and cheap figure regeneration.
%
%   solver_err_last holds the last-registered solver's error against the direct
%   backslash solve -- the accuracy figure's primary curve.  It was previously
%   plotted but never stored, which is why the figures for older results
%   directories cannot show it.  The name deliberately does not end in '_its',
%   so varvisc_make_paper_summary_table's endsWith(vn,'_its') solver discovery is
%   unaffected.
%
%   See also: varvisc_load_benchmark_stats, varvisc_make_paper_summary_table.

    nsolv = numel(keys);
    case_col = {}; ts_col = []; relres = []; diffF = [];
    bs = []; constr = []; nCc = []; errLast = [];
    diffKc = []; nuC = []; dKnnz = [];
    its = repmat({[]}, nsolv, 1); fl = repmat({[]}, nsolv, 1);
    times = repmat({[]}, nsolv, 1);
    solver_rr = repmat({[]}, nsolv, 1);
    true_rr = repmat({[]}, nsolv, 1);
    solver_err = repmat({[]}, nsolv, 1);
    for k = 1:numel(all_stats)
        st = all_stats{k};
        ns = numel(st.solver_its.(keys{1}));
        case_col = [case_col; repmat({st.case_name}, ns, 1)];   %#ok<AGROW>
        ts_col   = [ts_col;   (1:ns)'];                          %#ok<AGROW>
        for s = 1:nsolv
            its{s} = [its{s}; st.solver_its.(keys{s})(:)];
            fl{s}  = [fl{s};  st.solver_flag.(keys{s})(:)];
            times{s} = [times{s}; stat_field(st, 'solver_time', keys{s}, ns)];
            solver_rr{s} = [solver_rr{s}; ...
                stat_field(st, 'solver_relres', keys{s}, ns)];
            true_rr{s} = [true_rr{s}; ...
                stat_field(st, 'solver_true_relres', keys{s}, ns)];
            solver_err{s} = [solver_err{s}; ...
                stat_field(st, 'solver_err', keys{s}, ns)];
        end
        relres = [relres; st.solver_relres.(keys{end})(:)];      %#ok<AGROW>
        diffF  = [diffF;  st.coupling_change(:)];                %#ok<AGROW>
        bs     = [bs;     st.backslash_relres(:)];               %#ok<AGROW>
        constr = [constr; st.constraint_res(:)];                 %#ok<AGROW>
        nCc    = [nCc;    st.nC(:)];                             %#ok<AGROW>
        diffKc = [diffKc; st.diffK(:)];                          %#ok<AGROW>
        nuC    = [nuC;    st.nu_contrast(:)];                    %#ok<AGROW>
        dKnnz  = [dKnnz;  st.dK_nnz_frac(:)];                    %#ok<AGROW>
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
        T.([keys{s} '_time_s']) = times{s};
        T.([keys{s} '_relres']) = solver_rr{s};
        T.([keys{s} '_true_relres']) = true_rr{s};
        T.([keys{s} '_err']) = solver_err{s};
    end
    T.relres = relres; T.diffF = diffF; T.backslash_relres = bs;
    T.constraint_res = constr; T.nC = nCc;
    T.diffK = diffKc; T.nu_contrast = nuC; T.dK_nnz_frac = dKnnz;
    T.solver_err_last = errLast;
    for s = 1:nsolv
        info_names = solver_info_names(all_stats, keys{s});
        for j = 1:numel(info_names)
            values = [];
            for k = 1:numel(all_stats)
                st = all_stats{k};
                ns = numel(st.solver_its.(keys{s}));
                values = [values; solver_info_field( ...
                    st, keys{s}, info_names{j}, ns)]; %#ok<AGROW>
            end
            T.([keys{s} '_' info_names{j}]) = values;
        end
    end
    writetable(T, fullfile(results_root, 'all_results.csv'));
    fprintf('Wrote %s\n', fullfile(results_root, 'all_results.csv'));
end

function values = stat_field(st, group, key, n)
    if isfield(st, group) && isfield(st.(group), key)
        values = st.(group).(key)(:);
    else
        values = nan(n, 1);
    end
end

function names = solver_info_names(all_stats, key)
    names = {};
    for k = 1:numel(all_stats)
        st = all_stats{k};
        if isfield(st, 'solver_info') && isfield(st.solver_info, key)
            names = union(names, fieldnames(st.solver_info.(key)), 'stable');
        end
    end
end

function values = solver_info_field(st, key, name, n)
    if isfield(st, 'solver_info') && isfield(st.solver_info, key) && ...
            isfield(st.solver_info.(key), name)
        values = st.solver_info.(key).(name)(:);
    else
        values = nan(n, 1);
    end
end
