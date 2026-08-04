function lgd = place_solver_legend(parent, h, labels, opts)
%PLACE_SOLVER_LEGEND  One legend for a solver figure, outside the data area.
%
%   LGD = PLACE_SOLVER_LEGEND(PARENT, H, LABELS, OPTS)
%
%   PARENT is a tiledlayout or an axes.  H are the line handles from
%   plot_solver_curves and LABELS the matching text.
%
%   The placement rule matters and is enforced here rather than left to call
%   sites: inside a tiledlayout, 'Location','southoutside' shrinks the axes but
%   reserves no layout space, so the legend spills over the neighbouring tiles
%   -- exactly the overlap this rewrite exists to remove.  Only Layout.Tile
%   reserves a real tile.  For a bare axes there is no layout, so
%   'southoutside' is correct there.
%
%   Labels are drawn with the TeX interpreter: the registry labels are authored
%   as TeX (L^{-T}PL^{-1}) and contain no underscores.
%
%   See also: plot_solver_curves, save_benchmark_figure.

    if nargin < 4 || isempty(opts), opts = benchmark_fig_defaults(); end

    ncol = max(1, min(opts.legend_columns, numel(labels)));

    if isa(parent, 'matlab.graphics.layout.TiledChartLayout')
        lgd = legend(h, labels, 'Interpreter', 'tex', ...
                     'FontSize', opts.legendfontsize, 'NumColumns', ncol, ...
                     'Box', 'on', 'EdgeColor', [0.65 0.65 0.65]);
        lgd.Layout.Tile = 'south';
    else
        lgd = legend(parent, h, labels, 'Interpreter', 'tex', ...
                     'FontSize', opts.legendfontsize, 'NumColumns', ncol, ...
                     'Location', 'southoutside', ...
                     'Box', 'on', 'EdgeColor', [0.65 0.65 0.65]);
    end
    lgd.ItemTokenSize = [16 8];
end
