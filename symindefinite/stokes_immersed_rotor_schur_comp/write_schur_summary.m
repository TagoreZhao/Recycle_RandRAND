function write_schur_summary(results_root, all_stats, opts)
%WRITE_SCHUR_SUMMARY  Cross-case comparison figure with one shared legend.
%   WRITE_SCHUR_SUMMARY(RESULTS_ROOT, ALL_STATS, OPTS)
%
%   One tile per motion case, shared y limits so the panels are comparable,
%   and a single legend in a south tile.
%
%   See also: write_schur_case_outputs.

    if nargin < 3 || isempty(opts), opts = benchmark_fig_defaults(); end
    out_dir = fullfile(results_root, 'summary_plots');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    ncase  = numel(all_stats);
    keys   = all_stats{1}.solver_keys;
    labels = all_stats{1}.solver_labels;
    nk     = numel(keys);
    sty    = solver_style_table(nk);

    % shared y range across every panel and arm
    lo = inf; hi = -inf;
    for k = 1:ncase
        for i = 1:nk
            v = max(all_stats{k}.solver_its.(keys{i}), 1);
            lo = min(lo, min(v)); hi = max(hi, max(v));
        end
    end

    w  = min(opts.max_width, opts.panel_width * ncase + opts.panel_margin);
    fh = figure('Visible', 'off', 'Units', 'inches', ...
                'Position', [1 1 w opts.multi_height], 'Color', 'w');
    tl = tiledlayout(fh, 1, ncase, 'Padding', 'compact', 'TileSpacing', 'compact');

    h = gobjects(nk, 1);
    for k = 1:ncase
        A     = all_stats{k};
        steps = (1:A.nsteps)';
        ax    = nexttile(tl);
        for i = 1:nk
            hh = semilogy(ax, steps, max(A.solver_its.(keys{i}), 1), ...
                'LineWidth', sty(i).linewidth, 'Color', sty(i).color, ...
                'LineStyle', sty(i).linestyle, 'Marker', sty(i).marker, ...
                'MarkerSize', sty(i).markersize, ...
                'MarkerIndices', 1:max(1, round(A.nsteps/opts.marker_targets)):A.nsteps);
            hold(ax, 'on');
            if k == 1, h(i) = hh; end
        end
        grid(ax, 'on');
        ylim(ax, [max(lo*0.8, 0.8), hi*1.25]);
        xlabel(ax, 'time step', 'FontSize', opts.labelfontsize);
        if k == 1
            ylabel(ax, opts.iter_label, 'FontSize', opts.labelfontsize);
        end
        title(ax, A.case_name, 'FontSize', opts.titlefontsize, 'Interpreter', 'none');
        ax.FontSize = opts.fontsize;
    end

    title(tl, sprintf('%s vs time step (%s)', opts.iter_label, ...
                      all_stats{1}.geometry), ...
          'FontSize', opts.titlefontsize, 'Interpreter', 'none');
    lgd = legend(h, labels, 'Interpreter', 'none', ...
                 'FontSize', opts.legendfontsize, ...
                 'NumColumns', min(opts.legend_columns, nk));
    lgd.Layout.Tile = 'south';

    save_benchmark_figure(fh, fullfile(out_dir, 'all_cases_comparison.png'), opts);
end
