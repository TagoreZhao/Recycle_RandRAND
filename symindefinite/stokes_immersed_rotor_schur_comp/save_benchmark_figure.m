function outFile = save_benchmark_figure(fh, outFile, opts)
%SAVE_BENCHMARK_FIGURE  House style + the single export path for this benchmark.
%
%   OUTFILE = SAVE_BENCHMARK_FIGURE(FH, OUTFILE)
%   OUTFILE = SAVE_BENCHMARK_FIGURE(FH, OUTFILE, OPTS)
%
%   Replaces the old saveas(...,'.png') calls, which took the screen dpi and the
%   default grey figure background.  exportgraphics gives a controlled
%   resolution and a white background, and applying the font sizes here (rather
%   than at each of the six call sites) keeps every figure legible at the same
%   physical size.
%
%   Set OPTS.close = false to keep the figure open (used by the tests, which
%   inspect the handle after writing).
%
%   Modelled on symindefinite/coordinate_drift/kernel/save_figure.m, which is
%   bound to that study's figure directory and so cannot be reused directly.
%
%   See also: benchmark_fig_defaults, place_solver_legend.

    if nargin < 3 || isempty(opts), opts = benchmark_fig_defaults(); end

    outDir = fileparts(outFile);
    if ~isempty(outDir) && ~exist(outDir, 'dir'), mkdir(outDir); end

    set(fh, 'Color', 'w');
    for a = findall(fh, 'Type', 'axes')'
        % MinorGridAlpha is turned well down: on a log axis spanning 1e-8..1e-16
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
