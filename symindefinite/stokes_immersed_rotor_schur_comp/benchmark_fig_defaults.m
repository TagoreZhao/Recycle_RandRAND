function opts = benchmark_fig_defaults(overrides)
%BENCHMARK_FIG_DEFAULTS  Single source of truth for benchmark figure geometry.
%
%   OPTS = BENCHMARK_FIG_DEFAULTS()
%   OPTS = BENCHMARK_FIG_DEFAULTS(OVERRIDES)   % struct; fields replace defaults
%
%   Sizes are in INCHES so the point sizes below mean the same thing on every
%   figure -- the old pixel-sized figures rendered 10 pt text at whatever the
%   screen dpi happened to be, which is how a 57-character legend ended up wider
%   than its own axes.
%
%   Height budgets already include room for the legend tile: the multi-solver
%   figures are 5.0 in tall so the south legend does not eat the data area.
%
%   See also: save_benchmark_figure, place_solver_legend, plot_solver_curves.

    opts = struct( ...
        'fontsize',        10,  ...   % axes tick labels
        'labelfontsize',   11,  ...   % x/y labels
        'titlefontsize',   12,  ...   % axes / layout title
        'subtitlefontsize', 9,  ...   % full solver label under a short title
        'legendfontsize',   9,  ...
        'resolution',     200,  ...   % dpi for exportgraphics
        'single_width',   6.5,  ...   % one-curve figures (per-solver, coupling)
        'single_height',  3.6,  ...
        'accuracy_width', 6.5,  ...
        'accuracy_height', 4.0, ...
        'multi_width',    7.5,  ...   % all-solvers figures (legend tile below)
        'multi_height',   5.0,  ...
        'panel_width',    3.9,  ...   % per-case panel in the comparison figure
        'panel_margin',   0.8,  ...
        'max_width',       14,  ...
        'legend_columns',   4,  ...   % 10 arms -> 3 rows, no overlap
        'marker_targets',   8,  ...   % markers drawn per curve (staggered)
        'iter_label', 'PCG iterations', ...   % this study is SPD: PCG, not MINRES
        'close',        true);

    if nargin >= 1 && ~isempty(overrides) && isstruct(overrides)
        f = fieldnames(overrides);
        for i = 1:numel(f)
            opts.(f{i}) = overrides.(f{i});
        end
    end
end
