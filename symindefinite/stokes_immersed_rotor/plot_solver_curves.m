function [h, legLabels] = plot_solver_curves(ax, xax, stats, xlab, opts)
%PLOT_SOLVER_CURVES  Overlay every solver's iteration curve on one log axes.
%
%   [H, LEGLABELS] = PLOT_SOLVER_CURVES(AX, XAX, STATS, XLAB, OPTS)
%
%   Draws the lines and returns their handles plus the legend text.  It does
%   NOT create a legend and does NOT set a title: the caller owns figure-level
%   decoration.  That split is the fix for the old layout -- this function used
%   to call legend(...,'Location','best') itself, so the three-panel comparison
%   figure got three full-width legends inside three 420 px panels.
%
%   XLAB '' leaves the x label unset (for tiled panels that share one).
%
%   See also: place_solver_legend, solver_style_table, mark_coincident_curves.

    if nargin < 5 || isempty(opts), opts = benchmark_fig_defaults(); end

    keys = stats.solver_keys;
    n    = numel(keys);
    sty  = solver_style_table(n);
    tags = mark_coincident_curves(stats);

    xax  = xax(:);
    ns   = numel(xax);
    % Stagger the markers: each curve gets ~opts.marker_targets of them, on
    % different samples from its neighbours, so coincident curves show
    % interleaved marker trains instead of one solid overprinted row.
    stride = max(1, round(ns / max(opts.marker_targets, 1)));

    h         = gobjects(n, 1);
    legLabels = cell(n, 1);
    hold(ax, 'on');
    for s = 1:n
        y   = max(stats.solver_its.(keys{s})(:), 1);
        idx = (1 + mod(s - 1, stride)) : stride : ns;
        h(s) = plot(ax, xax, y, ...
            'Color',         sty(s).color, ...
            'LineStyle',     sty(s).linestyle, ...
            'LineWidth',     sty(s).linewidth, ...
            'Marker',        sty(s).marker, ...
            'MarkerSize',    sty(s).markersize, ...
            'MarkerIndices', idx);
        legLabels{s} = [solver_short_label(keys{s}) tags{s}];
    end
    hold(ax, 'off');

    set(ax, 'YScale', 'log');
    grid(ax, 'on');
    if ~isempty(xlab), xlabel(ax, xlab); end
    ylabel(ax, 'MINRES iterations');
end
