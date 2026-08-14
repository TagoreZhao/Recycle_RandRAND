function save_varvisc_schur_figure(fh, outFile, opts)
%SAVE_VARVISC_SCHUR_FIGURE  Apply shared styling and export a PNG.
    if nargin < 3 || isempty(opts), opts = varvisc_schur_fig_defaults(); end
    outDir = fileparts(outFile);
    if ~isempty(outDir) && ~exist(outDir,'dir'), mkdir(outDir); end
    set(fh,'Color','w');
    for ax = findall(fh,'Type','axes')'
        set(ax,'FontSize',opts.fontsize,'Box','on','LineWidth',.8, ...
               'GridAlpha',.15,'MinorGridAlpha',.06);
        grid(ax,'on');
    end
    exportgraphics(fh,outFile,'Resolution',opts.resolution,'BackgroundColor','white');
    if opts.close, close(fh); end
end
