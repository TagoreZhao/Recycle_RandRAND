function varvisc_write_iteration_vs_timestep(out_dir, stats, opts)
%WRITE_ITERATION_VS_TIMESTEP  All solvers' iterations vs time step, one case.
%
%   WRITE_ITERATION_VS_TIMESTEP(OUT_DIR, STATS, OPTS)
%   Writes <OUT_DIR>/<case_name>.png.
%
%   Same content as all_solvers_comparison but indexed by step number rather
%   than physical time; kept as a separate figure because it is the one the
%   write-up reads iteration counts off directly.
%
%   The legend lives in its own south tile, so it can no longer cover the
%   deflation cluster the way the old 'Location','best' legend did.
%
%   See also: varvisc_plot_solver_curves, varvisc_place_solver_legend.

    if nargin < 3 || isempty(opts), opts = varvisc_fig_defaults(); end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    ns = numel(stats.solver_its.(stats.solver_keys{1}));

    fh = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', ...
                'Position', [1 1 opts.multi_width opts.multi_height]);
    tl = tiledlayout(fh, 1, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    ax = nexttile(tl);
    [h, legLabels] = varvisc_plot_solver_curves(ax, (1:ns)', stats, 'time step n', opts);
    title(tl, sprintf('%s: iterations vs time step', stats.case_name), ...
          'Interpreter', 'none', 'FontWeight', 'bold', ...
          'FontSize', opts.titlefontsize);
    varvisc_place_solver_legend(tl, h, legLabels, opts);

    save_varvisc_figure(fh, fullfile(out_dir, [stats.case_name '.png']), opts);
end
