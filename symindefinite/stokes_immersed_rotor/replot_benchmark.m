function replot_benchmark(results_root, varargin)
%REPLOT_BENCHMARK  Redraw every benchmark figure from a finished run's CSVs.
%
%   REPLOT_BENCHMARK
%   REPLOT_BENCHMARK(RESULTS_ROOT)
%   REPLOT_BENCHMARK(RESULTS_ROOT, 'DryRun', true)
%   REPLOT_BENCHMARK(RESULTS_ROOT, 'RewriteCsv', true)
%
%   RESULTS_ROOT defaults to benchmark_no_krylov_recycle and may be given
%   relative to this file's directory.  A full re-run is 3 cases x 61 steps x 8
%   MINRES solves, so figure changes go through here instead.
%
%   FIGURES ONLY by default: all_results.csv, speedup_summary.csv,
%   paper_summary_table.csv, run_config.* and coefficient_movie/ are left
%   untouched.  'RewriteCsv' additionally regenerates the per-solver iteration
%   CSVs from the same data (identical content; opt-in so a replot cannot be
%   blamed for a data change).  'DryRun' lists what would be overwritten and
%   returns.
%
%   See also: load_benchmark_stats, write_case_figures, run_benchmark.

    thisFileDir = fileparts(mfilename('fullpath'));
    repoRoot    = fileparts(fileparts(thisFileDir));
    addpath(repoRoot);
    addpath(thisFileDir);

    if nargin < 1 || isempty(results_root)
        results_root = 'benchmark_no_krylov_recycle';
    end
    if ~isfolder(results_root)
        results_root = fullfile(thisFileDir, results_root);
    end
    if ~isfolder(results_root)
        error('replot_benchmark:noDir', 'No such results directory: %s', results_root);
    end

    p = inputParser;
    p.addParameter('DryRun',     false, @(x) islogical(x) && isscalar(x));
    p.addParameter('RewriteCsv', false, @(x) islogical(x) && isscalar(x));
    p.addParameter('FigOpts',    struct(), @isstruct);
    p.parse(varargin{:});
    dryRun     = p.Results.DryRun;
    rewriteCsv = p.Results.RewriteCsv;
    opts       = benchmark_fig_defaults(p.Results.FigOpts);

    [all_stats, cfg] = load_benchmark_stats(results_root);
    fprintf('[replot_benchmark] %s\n', results_root);
    fprintf('  %d case(s), %d solver(s), %d time step(s), dt = %g\n', ...
        numel(all_stats), numel(cfg.solver_keys), ...
        numel(all_stats{1}.solver_its.(cfg.solver_keys{1})), all_stats{1}.dt);

    if dryRun
        list_targets(results_root, all_stats, cfg, rewriteCsv);
        return
    end

    for k = 1:numel(all_stats)
        st = all_stats{k};
        case_dir = fullfile(results_root, st.case_name);
        if rewriteCsv, write_case_csvs(case_dir, st); end
        write_case_figures(case_dir, st, opts);
    end

    ivt_dir = fullfile(results_root, 'iteration_vs_timestep');
    for k = 1:numel(all_stats)
        write_iteration_vs_timestep(ivt_dir, all_stats{k}, opts);
    end
    write_all_cases_comparison(fullfile(results_root, 'summary_plots'), all_stats, opts);

    fprintf('[replot_benchmark] done.\n');
end

%==========================================================================
function list_targets(results_root, all_stats, cfg, rewriteCsv)
    fprintf('  DryRun -- would overwrite:\n');
    for k = 1:numel(all_stats)
        st = all_stats{k};
        for s = 1:numel(cfg.solver_keys)
            fprintf('    %s\n', fullfile(results_root, st.case_name, ...
                [cfg.solver_keys{s} '_solver_iterations.png']));
            if rewriteCsv
                fprintf('    %s\n', fullfile(results_root, st.case_name, ...
                    [cfg.solver_keys{s} '_solver_iterations.csv']));
            end
        end
        for f = {'all_solvers_comparison.png', 'relative_step_to_step_change.png', ...
                 'accuracy.png'}
            fprintf('    %s\n', fullfile(results_root, st.case_name, f{1}));
        end
        fprintf('    %s\n', fullfile(results_root, 'iteration_vs_timestep', ...
            [st.case_name '.png']));
    end
    fprintf('    %s\n', fullfile(results_root, 'summary_plots', ...
        'all_cases_comparison.png'));
end
