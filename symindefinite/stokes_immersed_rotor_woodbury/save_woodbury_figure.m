function outFile = save_woodbury_figure(fh, outFile, opts)
%SAVE_WOODBURY_FIGURE  House style + the single export path for this study.
%
%   OUTFILE = SAVE_WOODBURY_FIGURE(FH, OUTFILE)
%   OUTFILE = SAVE_WOODBURY_FIGURE(FH, OUTFILE, OPTS)
%
%   exportgraphics rather than saveas: a controlled resolution and a white
%   background instead of the screen dpi and the default grey figure colour.
%   Applying the font sizes here rather than at each call site keeps every figure
%   legible at the same physical size.
%
%   Set OPTS.close = false to keep the figure open (used by the tests, which
%   inspect the handle after writing).
%
%   LOCAL COPY of the sibling benchmarks' save_benchmark_figure, RENAMED for the
%   shadowing reason documented in add_woodbury_paths.  Otherwise unchanged.
%
%   See also: woodbury_fig_defaults, write_woodbury_figures.

    if nargin < 3 || isempty(opts), opts = woodbury_fig_defaults(); end

    outDir = fileparts(outFile);
    if ~isempty(outDir) && ~exist(outDir, 'dir'), mkdir(outDir); end

    set(fh, 'Color', 'w');
    for a = findall(fh, 'Type', 'axes')'
        % MinorGridAlpha is turned well down: on a log axis spanning 1e-16..1e0
        % the default minor grid is dense enough to compete with the curves.
        set(a, 'FontSize', opts.fontsize, 'Box', 'on', 'LineWidth', 0.8, ...
               'GridAlpha', 0.15, 'MinorGridAlpha', 0.06, ...
               'MinorGridLineStyle', '-');
        grid(a, 'on');
        set(get(a, 'XLabel'), 'FontSize', opts.labelfontsize);
        set(get(a, 'YLabel'), 'FontSize', opts.labelfontsize);
    end

    exportgraphics(fh, outFile, 'Resolution', opts.resolution, ...
                   'BackgroundColor', 'white');
    fprintf('Wrote %s\n', outFile);
    if opts.close, close(fh); end
end
