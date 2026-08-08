function opts = woodbury_fig_defaults(overrides)
%WOODBURY_FIG_DEFAULTS  Single source of truth for this study's figure geometry.
%
%   OPTS = WOODBURY_FIG_DEFAULTS()
%   OPTS = WOODBURY_FIG_DEFAULTS(OVERRIDES)   % struct; fields replace defaults
%
%   Sizes are in INCHES so the point sizes below mean the same thing on every
%   figure, rather than depending on screen dpi.
%
%   LOCAL COPY of the sibling benchmarks' benchmark_fig_defaults, RENAMED.  The
%   rename is not cosmetic: build_stokes_sequence calls add_recycle_paths()
%   internally, which prepends the sibling rotor folder to the path mid-run, so a
%   same-named local copy could be shadowed after add_woodbury_paths ran.  See the
%   hazard note in add_woodbury_paths.
%
%   Differences from the sibling's copy:
%     legend_columns  3, because this study registers exactly three arms
%     value_label     'relative error' -- these are DIRECT solves, so there is no
%                     iteration count to label an axis with
%
%   See also: save_woodbury_figure, woodbury_style_table, add_woodbury_paths.

    opts = struct( ...
        'fontsize',        10,  ...   % axes tick labels
        'labelfontsize',   11,  ...   % x/y labels
        'titlefontsize',   12,  ...   % axes / layout title
        'subtitlefontsize', 9,  ...   % full solver label under a short title
        'legendfontsize',   9,  ...
        'resolution',     200,  ...   % dpi for exportgraphics
        'single_width',   6.5,  ...
        'single_height',  3.6,  ...
        'accuracy_width', 6.5,  ...
        'accuracy_height', 4.0, ...
        'multi_width',    7.5,  ...   % all-arms figures (legend tile below)
        'multi_height',   5.0,  ...
        'panel_width',    3.9,  ...   % per-case panel in the comparison figure
        'panel_margin',   0.8,  ...
        'max_width',       14,  ...
        'legend_columns',   3,  ...   % three arms -> one row
        'marker_targets',   8,  ...   % markers drawn per curve (staggered)
        'value_label', 'relative error', ...  % direct solves: no iteration counts
        'close',        true);

    if nargin >= 1 && ~isempty(overrides) && isstruct(overrides)
        f = fieldnames(overrides);
        for i = 1:numel(f)
            opts.(f{i}) = overrides.(f{i});
        end
    end
end
