function varvisc_write_upgraded_linear_iteration_figures(results_root, all_stats)
%VARVISC_WRITE_UPGRADED_LINEAR_ITERATION_FIGURES Linear-axis companions to
% the standard logarithmic iteration-versus-timestep figures.
    if nargin < 2 || isempty(all_stats)
        [all_stats, ~] = varvisc_load_benchmark_stats(results_root);
    end
    opts = varvisc_fig_defaults(struct('yscale', 'linear'));
    iteration_dir = fullfile(results_root, 'iteration_vs_timestep');
    for k = 1:numel(all_stats)
        varvisc_write_iteration_vs_timestep( ...
            iteration_dir, all_stats{k}, opts, '_linear');
    end
    varvisc_write_all_cases_comparison( ...
        fullfile(results_root, 'summary_plots'), all_stats, opts, ...
        'all_cases_comparison_linear.png');
end
