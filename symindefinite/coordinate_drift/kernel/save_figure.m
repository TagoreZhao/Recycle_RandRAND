function outFile = save_figure(fh, name, opts)
%SAVE_FIGURE  One style, one export path, for every figure in this study.
%
%   outFile = SAVE_FIGURE(FH, NAME)
%   outFile = SAVE_FIGURE(FH, NAME, OPTS)
%
%   Applies the house style and writes <study>/figures/NAME.png.  The figures
%   directory is COMMITTED (unlike output/), because the document embeds these
%   images inline with ![...](figures/NAME.png) -- a reader of the markdown must
%   see the plots without running anything.
%
%   Style: font size >= 11 pt so text stays legible when the PNG is scaled to a
%   markdown column, a box and a light grid on every axis, and a 150-dpi export.
%
%   OPTS: .width .height (inches, default 9 x 4.5), .fontsize (default 12),
%         .resolution (default 150), .close (default true)
%
%   See also: run_all.

    if nargin < 3 || isempty(opts), opts = struct(); end
    w    = getdef(opts, 'width',      9);
    h    = getdef(opts, 'height',     4.5);
    fs   = getdef(opts, 'fontsize',   12);
    res  = getdef(opts, 'resolution', 150);
    doClose = getdef(opts, 'close',   true);

    p = add_paths();
    if ~exist(p.figDir, 'dir'), mkdir(p.figDir); end

    set(fh, 'Units', 'inches', 'Position', [1 1 w h], 'Color', 'w');
    ax = findall(fh, 'Type', 'axes');
    for a = ax(:)'
        set(a, 'FontSize', fs, 'Box', 'on', 'LineWidth', 0.8);
        grid(a, 'on');
        set(a, 'GridAlpha', 0.15);
        set(get(a, 'XLabel'), 'FontSize', fs);
        set(get(a, 'YLabel'), 'FontSize', fs);
        set(get(a, 'Title'),  'FontSize', fs + 1, 'FontWeight', 'bold');
        lg = get(a, 'Legend');
        if ~isempty(lg), set(lg, 'FontSize', fs - 1); end
    end

    outFile = fullfile(p.figDir, [name '.png']);
    exportgraphics(fh, outFile, 'Resolution', res, 'BackgroundColor', 'white');
    fprintf('[fig] wrote %s\n', outFile);
    if doClose, close(fh); end
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
