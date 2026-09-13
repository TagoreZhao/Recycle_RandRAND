function replot_dynamic_tau_varvisc_benchmark(results_root)
%REPLOT_DYNAMIC_TAU_VARVISC_BENCHMARK Redraw focused plots without solving.
    here = fileparts(mfilename('fullpath'));
    parent = fullfile(fileparts(here), 'stokes_varvisc_rotor');
    addpath(parent, here);
    if nargin < 1 || isempty(results_root)
        results_root = fullfile(here, ...
            'benchmark_varvisc_upgraded_dynamic_tau');
    elseif ~isfolder(results_root)
        results_root = fullfile(here, results_root);
    end
    [all_stats, cfg] = varvisc_load_benchmark_stats(results_root);
    linear = varvisc_fig_defaults(struct('yscale', 'linear'));
    logopts = varvisc_fig_defaults(struct('yscale', 'log'));
    ivt_dir = fullfile(results_root, 'iteration_vs_timestep');
    for k = 1:numel(all_stats)
        varvisc_write_iteration_vs_timestep( ...
            ivt_dir, all_stats{k}, linear, '_linear');
        varvisc_write_iteration_vs_timestep( ...
            ivt_dir, all_stats{k}, logopts, '_log');
    end
    summary_dir = fullfile(results_root, 'summary_plots');
    varvisc_write_all_cases_comparison(summary_dir, all_stats, linear, ...
        'all_cases_comparison_linear.png');
    varvisc_write_all_cases_comparison(summary_dir, all_stats, logopts, ...
        'all_cases_comparison_log.png');
    varvisc_write_dynamic_tau_summary( ...
        results_root, all_stats, cfg.geometry);
end
