function varvisc_write_all_cases_comparison(out_dir, all_stats, opts, output_name)
%WRITE_ALL_CASES_COMPARISON  One panel per motion case, all solvers overlaid.
%
%   WRITE_ALL_CASES_COMPARISON(OUT_DIR, ALL_STATS, OPTS)
%   WRITE_ALL_CASES_COMPARISON(OUT_DIR, ALL_STATS, OPTS, OUTPUT_NAME)
%   OUTPUT_NAME defaults to all_cases_comparison.png; OPTS.yscale selects
%   the logarithmic default or a linear y axis.
%
%   This is the figure the old code broke worst: subplot(1,nc,k) plus a
%   per-panel legend meant three full-width legends inside three 420 px panels,
%   spilling across each other and clipping the y label to 'ES iterations'.
%   Here the panels are tiles, there is exactly ONE legend (in its own south
%   tile), and the axis labels belong to the layout rather than to each panel.
%
%   The panels share y limits so iteration counts are comparable across cases,
%   which also lets the interior tiles drop their tick labels.
%
%   See also: varvisc_plot_solver_curves, varvisc_place_solver_legend.

    if nargin < 3 || isempty(opts), opts = varvisc_fig_defaults(); end
    if nargin < 4 || isempty(output_name), output_name = 'all_cases_comparison.png'; end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    nc    = numel(all_stats);
    width = min(opts.panel_width * nc + opts.panel_margin, opts.max_width);
    gylim = shared_ylim(all_stats, opts);

    fh = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', ...
                'Position', [1 1 width opts.multi_height]);
    if nc > 4
        tl = tiledlayout(fh, 'flow', 'Padding', 'compact', 'TileSpacing', 'compact');
    else
        tl = tiledlayout(fh, 1, nc, 'Padding', 'compact', 'TileSpacing', 'compact');
    end

    % Panels must be read against each other, so they share both axes: MATLAB
    % otherwise picks per-panel ticks (20/40/60 here, 0..60 there) and the eye
    % has to re-anchor on every panel.
    nsAll = cellfun(@(s) numel(s.solver_its.(s.solver_keys{1})), all_stats);
    gxlim = [1 max(nsAll)];
    gxtick = unique([1, 10:10:max(nsAll)]);

    h = gobjects(0); legLabels = {};
    for k = 1:nc
        st = all_stats{k};
        ns = numel(st.solver_its.(st.solver_keys{1}));
        ax = nexttile(tl);
        [hk, lk] = varvisc_plot_solver_curves(ax, (1:ns)', st, '', opts);
        if k == 1, h = hk; legLabels = lk; end
        ylim(ax, gylim);
        xlim(ax, gxlim);
        xticks(ax, gxtick);
        ylabel(ax, '');                       % the layout owns the labels
        if k > 1, ax.YTickLabel = []; end     % shared scale -> label once
        title(ax, st.case_name, 'Interpreter', 'none', 'FontSize', 11);
    end

    tl.XLabel.String    = 'time step n';
    tl.XLabel.FontSize  = opts.labelfontsize;
    tl.YLabel.String    = 'Krylov iterations';
    tl.YLabel.FontSize  = opts.labelfontsize;
    varvisc_place_solver_legend(tl, h, legLabels, opts);

    geom = 'stokes_immersed_rotor';
    if isfield(all_stats{1}, 'geometry') && ~isempty(all_stats{1}.geometry)
        geom = all_stats{1}.geometry;
    end
    scale_note = '';
    if strcmpi(opts.yscale, 'linear'), scale_note = ', linear scale'; end
    title(tl, sprintf('Krylov iterations vs time step (%s%s)', geom, scale_note), ...
          'Interpreter', 'none', 'FontWeight', 'bold', ...
          'FontSize', opts.titlefontsize);

    save_varvisc_figure(fh, fullfile(out_dir, output_name), opts);
end

%==========================================================================
function lims = shared_ylim(all_stats, opts)
%SHARED_YLIM Common limits over every case and solver for the selected scale.
    lo = inf; hi = -inf;
    for k = 1:numel(all_stats)
        st = all_stats{k};
        for s = 1:numel(st.solver_keys)
            v  = max(st.solver_its.(st.solver_keys{s})(:), 1);
            lo = min(lo, min(v));
            hi = max(hi, max(v));
        end
    end
    if ~isfinite(lo) || ~isfinite(hi) || lo <= 0
        lims = [1 10];
        return
    end
    if strcmpi(opts.yscale, 'linear')
        lims = [0, hi * 1.05];
    else
        lims = [lo * 0.85, hi * 1.20];
    end
end
