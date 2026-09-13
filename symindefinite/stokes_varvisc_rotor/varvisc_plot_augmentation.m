function varvisc_plot_augmentation(results_root)
%VARVISC_PLOT_AUGMENTATION Render saved evidence only; never solve or probe.
    filename = fullfile(results_root,'augmentation_results.csv');
    if ~isfile(filename), return; end
    T = readtable(filename);
    [~,cfg] = varvisc_load_benchmark_stats(results_root);
    mmain = cfg.params.AUGMENT_M;
    opts = varvisc_fig_defaults();
    for c=1:numel(cfg.case_names)
        cname = cfg.case_names{c};
        outdir = fullfile(results_root,cname,'augmentation_plots');
        Tc = T(strcmp(T.case_name,cname),:);
        B = sortrows(Tc(Tc.m==0,:),'timestep');
        A = sortrows(Tc(Tc.m==mmain,:),'timestep');
        t = B.timestep * cfg.params.dt;
        labels = {sprintf('Gaussian %d',B.base_rank(1)), ...
            sprintf('Gaussian %d + Arnoldi %d',A.base_rank(1),mmain)};
        fh = newfigure(9,6); tl = tiledlayout(fh,2,1,'TileSpacing','compact');
        ax = nexttile(tl); paired_handles=plot(ax,t,[B.iters A.iters],'LineWidth',1.5);
        hold(ax,'on'); bad = A.flag~=0 | A.true_relres>1e-6 | A.err>1e-5;
        bad_base = B.flag~=0 | B.true_relres>1e-6 | B.err>1e-5;
        if any(bad | bad_base)
            failed=plot(ax,[t(bad);t(bad_base)],[A.iters(bad);B.iters(bad_base)], ...
                'rx','MarkerSize',7,'LineWidth',1.5);
            paired_handles=[paired_handles;failed];
            labels{end+1}='Accuracy / convergence flag';
        end
        ylabel(ax,'MINRES iterations');
        title(ax,cname,'Interpreter','none');
        ax = nexttile(tl); plot(ax,t,B.iters-A.iters,'LineWidth',1.5); yline(ax,0,':');
        ylabel(ax,'Iterations saved'); xlabel(ax,'Physical time');
        varvisc_place_solver_legend(tl,paired_handles,labels,opts);
        finish(fh,outdir,'gaussian_comparison.png',opts);

        fh = newfigure(12,7); tl = tiledlayout(fh,2,3,'TileSpacing','compact');
        ax = nexttile(tl); semilogy(ax,t,max([A.vv_orth A.vw_orth A.ww_orth],realmin),'LineWidth',1.2);
        legend(ax,{'V orthogonality','V^T W','W orthogonality'},'Location','best'); title(ax,'Orthogonality');
        ax = nexttile(tl); plot(ax,t,[A.base_rank A.augment_rank A.total_rank],'LineWidth',1.2);
        legend(ax,{'Base rank','Arnoldi rank','Total rank'},'Location','best'); title(ax,'Actual dimensions');
        ax = nexttile(tl); semilogy(ax,t,max([A.leakage_before A.leakage_after],realmin),'LineWidth',1.2);
        legend(ax,{'Outside V','Outside [V,W]'},'Location','best'); title(ax,'Relative leakage of Ahat V');
        ax = nexttile(tl); semilogy(ax,t,[B.coarse_condition A.coarse_condition],'LineWidth',1.2);
        legend(ax,{'Gaussian','Gaussian + Arnoldi'},'Location','best'); title(ax,'Coarse matrix condition');
        ax = nexttile(tl); semilogy(ax,t,max(A.arnoldi_relation,realmin),'LineWidth',1.2); title(ax,'Arnoldi relation error');
        ax = nexttile(tl); plot(ax,t,A.arnoldi_coupling,'LineWidth',1.2); title(ax,'||V^T Ahat W||_F');
        for ax = findall(fh,'Type','axes')', xlabel(ax,'Physical time'); end
        finish(fh,outdir,'subspace_diagnostics.png',opts);

        fh = newfigure(9,4); ax = axes(fh);
        semilogy(ax,t,max([B.true_relres A.true_relres B.err A.err],realmin),'LineWidth',1.2);
        legend(ax,{'Gaussian physical residual','Augmented physical residual', ...
            'Gaussian solution error','Augmented solution error'},'Location','best');
        xlabel(ax,'Physical time'); ylabel(ax,'Relative error / residual');
        finish(fh,outdir,'paired_accuracy.png',opts);
        plot_spectra_and_histories(fullfile(results_root,cname),outdir,mmain,opts);
        if numel(unique(Tc.m))>2, plot_sweep(Tc,outdir,cfg.params.dt,opts); end
    end
end

function plot_sweep(T,outdir,dt,opts)
    ms = unique(T.m)'; labels = cell(1,numel(ms));
    meanit = zeros(size(ms)); ranks = meanit;
    fh = newfigure(11,7); tl = tiledlayout(fh,2,2,'TileSpacing','compact');
    ax1 = nexttile(tl); hold(ax1,'on'); ax2 = nexttile(tl); hold(ax2,'on');
    handles=gobjects(numel(ms),1);
    for j=1:numel(ms)
        R = sortrows(T(T.m==ms(j),:),'timestep');
        labels{j}=sprintf('%d + %d',R.base_rank(1),ms(j));
        handles(j)=plot(ax1,R.timestep*dt,R.iters,'LineWidth',1.2);
        plot(ax2,R.timestep*dt,max(R.true_relres,realmin),'LineWidth',1.2);
        meanit(j)=mean(R.iters); ranks(j)=mean(R.total_rank);
    end
    xlabel(ax1,'Physical time'); ylabel(ax1,'MINRES iterations');
    set(ax2,'YScale','log'); xlabel(ax2,'Physical time'); ylabel(ax2,'Physical relative residual');
    ax = nexttile(tl); bar(ax,ms,meanit); xlabel(ax,'Additional Arnoldi dimension m'); ylabel(ax,'Mean MINRES iterations');
    ax = nexttile(tl); plot(ax,ms,ranks,'o-','LineWidth',1.3); ylabel(ax,'Mean actual deflation dimension');
    xlabel(ax,'Additional Arnoldi dimension m');
    title(tl,'Additive dimension sweep: Gaussian base remains fixed');
    lg=varvisc_place_solver_legend(tl,handles,labels,opts); lg.NumColumns=numel(ms);
    finish(fh,outdir,'dimension_sweep.png',opts);
end

function plot_spectra_and_histories(case_dir,outdir,m,opts)
    files = dir(fullfile(case_dir,'diagnostics','spectrum_*.mat'));
    if isempty(files), return; end
    fh = newfigure(max(8,3*numel(files)),7);
    tl = tiledlayout(fh,2,numel(files),'TileSpacing','compact');
    colors = [0 .45 .70; .84 .37 0; .1 .1 .1];
    legend_handles = gobjects(3,1);
    first_ax = [];
    for j=1:numel(files)
        data=load(fullfile(files(j).folder,files(j).name),'spectrum'); s=data.spectrum;
        ax=nexttile(tl,j); hold(ax,'on'); axr=nexttile(tl,numel(files)+j); hold(axr,'on');
        for q=1:3
            sources={'before','after','method'}; r=s.(sources{q});
            idx=(1:numel(r.values))'/max(numel(r.values),1);
            hp=plot(ax,idx,r.values,'o','Color',colors(q,:),'MarkerSize',3,'DisplayName',sources{q});
            if j==1, legend_handles(q)=hp; first_ax=ax; end
            good=r.converged;
            scatter(ax,idx(good),r.values(good),12,colors(q,:),'filled','HandleVisibility','off');
            plot(axr,max(abs(r.values),realmin),max(r.relative_residual,realmin),'.', ...
                'Color',colors(q,:),'DisplayName',sources{q});
        end
        title(ax,sprintf('Step %d\nZeros excluded: %d/%d',s.step,s.forced_zeros_before,s.forced_zeros_after),'FontSize',9);
        xlabel(ax,'Normalized Ritz index'); ylabel(ax,'Signed Ritz value');
        set(axr,'XScale','log','YScale','log'); yline(axr,1e-6,':','HandleVisibility','off');
        xlabel(axr,'|Ritz value|'); ylabel(axr,'Relative eigenpair residual');

        bfile=fullfile(case_dir,'diagnostics',sprintf('m000_step_%03d.mat',s.step));
        afile=fullfile(case_dir,'diagnostics',sprintf('m%03d_step_%03d.mat',m,s.step));
        if isfile(bfile) && isfile(afile)
            bd=load(bfile,'record'); ad=load(afile,'record');
            rh=newfigure(8,4); ah=axes(rh); hold(ah,'on');
            br=bd.record.resvec; ar=ad.record.resvec;
            semilogy(ah,0:numel(br)-1,max(br/max(br(1),realmin),realmin),'LineWidth',1.3);
            semilogy(ah,0:numel(ar)-1,max(ar/max(ar(1),realmin),realmin),'LineWidth',1.3);
            set(ah,'YScale','log'); xlabel(ah,'MINRES iteration'); ylabel(ah,'Reported residual / initial value');
            legend(ah,{'Gaussian','Gaussian + Arnoldi'},'Location','best');
            title(ah,sprintf('Step %d; final physical residuals %.2e / %.2e', ...
                s.step,bd.record.true_relres,ad.record.true_relres),'FontSize',10);
            finish(rh,outdir,sprintf('residual_history_%03d.png',s.step),opts);
        end
    end
    lg=legend(first_ax,legend_handles,{'Before augmentation','After augmentation','Method W'}, ...
        'NumColumns',3,'Visible','on');
    lg.Layout.Tile='south';
    title(tl,'Projected spectral estimates; filled markers meet eigenpair residual 10^{-6}');
    finish(fh,outdir,'projected_spectra.png',opts);
end

function fh=newfigure(w,h)
    fh=figure('Visible','off','Color','w','Units','inches','Position',[.5 .5 w h]);
end
function finish(fh,outdir,name,opts)
    drawnow;
    % Resolve in-axes legends after invisible tiled figures have laid out.
    % 'best' can retain an obsolete location during headless export.
    for lg=findall(fh,'Type','legend')'
        if strcmp(lg.Location,'best'), lg.Location='northeast'; end
        lg.Visible='on'; lg.FontSize=opts.legendfontsize;
    end
    drawnow;
    save_varvisc_figure(fh,fullfile(outdir,name),opts);
end
