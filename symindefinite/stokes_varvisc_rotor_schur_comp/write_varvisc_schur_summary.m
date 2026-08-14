function write_varvisc_schur_summary(results_root, all_stats, opts)
%WRITE_VARVISC_SCHUR_SUMMARY  Cross-case iteration comparison.
    if nargin < 3 || isempty(opts), opts = varvisc_schur_fig_defaults(); end
    outDir = fullfile(results_root,'summary_plots');
    if ~exist(outDir,'dir'), mkdir(outDir); end
    keys = all_stats{1}.solver_keys; labels = all_stats{1}.solver_labels;
    sty = varvisc_schur_style_table(numel(keys)); ncase = numel(all_stats);
    fh = figure('Visible','off','Units','inches','Position', ...
                [1 1 min(opts.max_width,ncase*opts.panel_width) opts.multi_height]);
    tl = tiledlayout(fh,1,ncase,'Padding','compact','TileSpacing','compact');
    h = gobjects(numel(keys),1);
    for ci = 1:ncase
        A = all_stats{ci}; ax = nexttile(tl); steps = (1:A.nsteps)';
        for i = 1:numel(keys)
            hh = semilogy(ax,steps,max(A.solver_its.(keys{i}),1), ...
                'LineWidth',sty(i).linewidth,'Color',sty(i).color, ...
                'LineStyle',sty(i).linestyle); hold(ax,'on');
            if ci == 1, h(i) = hh; end
        end
        title(ax,A.case_name,'Interpreter','none'); xlabel(ax,'time step');
        if ci == 1, ylabel(ax,'PCG iterations'); end
    end
    lgd = legend(h,labels,'Interpreter','none','NumColumns',2); lgd.Layout.Tile = 'south';
    save_varvisc_schur_figure(fh,fullfile(outDir,'all_cases_comparison.png'),opts);
end
