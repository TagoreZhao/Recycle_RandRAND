function write_varvisc_schur_case_outputs(run_dir, A, opts)
%WRITE_VARVISC_SCHUR_CASE_OUTPUTS  Per-case CSVs and benchmark figures.
    if nargin < 3 || isempty(opts), opts = varvisc_schur_fig_defaults(); end
    if ~exist(run_dir,'dir'), mkdir(run_dir); end
    steps = (1:A.nsteps)'; keys = A.solver_keys; labels = A.solver_labels;
    sty = varvisc_schur_style_table(numel(keys));

    for i = 1:numel(keys)
        its = A.solver_its.(keys{i});
        writematrix(its,fullfile(run_dir,[keys{i} '_solver_iterations.csv']));
        fh = new_fig(opts.single_width,opts.single_height); ax = axes(fh);
        plot(ax,steps,its,'LineWidth',sty(i).linewidth,'Color',sty(i).color, ...
             'Marker',sty(i).marker,'MarkerIndices',marker_idx(A.nsteps,opts));
        xlabel(ax,'time step'); ylabel(ax,'PCG iterations');
        title(ax,labels{i},'Interpreter','none');
        save_varvisc_schur_figure(fh,fullfile(run_dir, ...
            [keys{i} '_solver_iterations.png']),opts);
    end

    fh = new_fig(opts.multi_width,opts.multi_height); ax = axes(fh); h = gobjects(numel(keys),1);
    for i = 1:numel(keys)
        h(i) = semilogy(ax,steps,max(A.solver_its.(keys{i}),1), ...
            'LineWidth',sty(i).linewidth,'Color',sty(i).color, ...
            'LineStyle',sty(i).linestyle,'Marker',sty(i).marker, ...
            'MarkerIndices',marker_idx(A.nsteps,opts)); hold(ax,'on');
    end
    xlabel(ax,'time step'); ylabel(ax,'PCG iterations');
    title(ax,[A.case_name ': all Schur arms'],'Interpreter','none');
    legend(ax,h,labels,'Interpreter','none','Location','best','FontSize',opts.legendfontsize);
    save_varvisc_schur_figure(fh,fullfile(run_dir,'all_solvers_comparison.png'),opts);

    curves = { ...
        'ReldiffF','relative_step_to_step_change.png','normalized ||S_n-S_{n-1}||_F'; ...
        'RelInitdiffF','diff_from_initial.png','normalized ||S_n-S_1||_F'; ...
        'InvRelDiff','relative_inverse_difference.png','frozen inverse error'; ...
        'A_change','velocity_block_change.png','normalized velocity-block change'; ...
        'D_change','stabilization_change.png','normalized stabilization change'; ...
        'pressure_schur_change','pressure_schur_change.png','pressure-Schur-block change'; ...
        'nu_contrast','viscosity_contrast.png','viscosity contrast'};
    for i = 1:size(curves,1)
        if isfield(A,curves{i,1}) && any(isfinite(A.(curves{i,1})))
            single_curve(run_dir,curves{i,2},steps,A.(curves{i,1}),curves{i,3},opts);
        end
    end

    if isfield(A,'kappa') && any(isfinite(A.kappa))
        fh = new_fig(opts.single_width,opts.single_height); ax = axes(fh);
        semilogy(ax,steps,A.kappa,'LineWidth',1.7);
        xlabel(ax,'time step'); ylabel(ax,'kappa(S_n)'); title(ax,[A.case_name ': conditioning'],'Interpreter','none');
        save_varvisc_schur_figure(fh,fullfile(run_dir,'kappa_vs_timestep.png'),opts);
    end
end

function single_curve(run_dir,name,steps,y,ylab,opts)
    fh = new_fig(opts.single_width,opts.single_height); ax = axes(fh);
    semilogy(ax,steps,max(y,eps),'LineWidth',1.7,'Marker','o', ...
             'MarkerIndices',marker_idx(numel(steps),opts));
    xlabel(ax,'time step'); ylabel(ax,ylab,'Interpreter','none');
    title(ax,strrep(name,'_',' '),'Interpreter','none');
    save_varvisc_schur_figure(fh,fullfile(run_dir,name),opts);
end

function idx = marker_idx(n,opts)
    idx = 1:max(1,round(n/opts.marker_targets)):n;
end

function fh = new_fig(w,h)
    fh = figure('Visible','off','Units','inches','Position',[1 1 w h],'Color','w');
end
