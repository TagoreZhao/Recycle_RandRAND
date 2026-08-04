function write_case_csvs(run_dir, stats)
%WRITE_CASE_CSVS  Per-solver iteration CSVs for one motion case.
%
%   WRITE_CASE_CSVS(RUN_DIR, STATS)
%   Writes <RUN_DIR>/<key>_solver_iterations.csv, one per solver.
%
%   Split out of the old write_case_outputs so that replot_benchmark can
%   regenerate figures without rewriting the data it just read back.
%
%   See also: write_case_figures, write_all_results_csv.

    if ~exist(run_dir, 'dir'), mkdir(run_dir); end
    keys = stats.solver_keys;
    ns   = numel(stats.solver_its.(keys{1}));
    for s = 1:numel(keys)
        T = table((1:ns)', stats.solver_its.(keys{s})(:), ...
                  'VariableNames', {'timestep', 'iterations'});
        writetable(T, fullfile(run_dir, [keys{s} '_solver_iterations.csv']));
    end
end
