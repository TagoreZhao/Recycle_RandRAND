function varvisc_write_case_csvs(run_dir, stats)
%WRITE_CASE_CSVS  Per-solver iteration CSVs for one motion case.
%
%   WRITE_CASE_CSVS(RUN_DIR, STATS)
%   Writes <RUN_DIR>/<key>_solver_iterations.csv, one per solver.
%
%   Split out of the old write_case_outputs so that replot_varvisc_benchmark can
%   regenerate figures without rewriting the data it just read back.
%
%   See also: varvisc_write_case_figures, varvisc_write_all_results_csv.

    if ~exist(run_dir, 'dir'), mkdir(run_dir); end
    keys = stats.solver_keys;
    ns   = numel(stats.solver_its.(keys{1}));
    for s = 1:numel(keys)
        key = keys{s};
        T = table((1:ns)', stats.solver_its.(key)(:), ...
            stat_field(stats, 'solver_flag', key, ns), ...
            stat_field(stats, 'solver_relres', key, ns), ...
            stat_field(stats, 'solver_true_relres', key, ns), ...
            stat_field(stats, 'solver_err', key, ns), ...
            stat_field(stats, 'solver_time', key, ns), ...
            'VariableNames', {'timestep', 'iterations', 'flag', ...
            'split_relres', 'true_relres', 'solution_error', 'time_s'});
        if isfield(stats, 'solver_info') && isfield(stats.solver_info, key)
            info = stats.solver_info.(key);
            names = fieldnames(info);
            for j = 1:numel(names)
                T.(names{j}) = info.(names{j})(:);
            end
        end
        writetable(T, fullfile(run_dir, [keys{s} '_solver_iterations.csv']));
    end
end


function values = stat_field(stats, group, key, n)
    if isfield(stats, group) && isfield(stats.(group), key)
        values = stats.(group).(key)(:);
    else
        values = nan(n, 1);
    end
end
