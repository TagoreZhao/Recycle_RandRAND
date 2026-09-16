function T = varvisc_schur_analyze_gaussian_refresh(root)
%VARVISC_SCHUR_ANALYZE_GAUSSIAN_REFRESH Validate, tabulate and plot saved runs.
    data=load(fullfile(root,'experiment_config.mat'),'config'); cfg=data.config;
    nsteps=cfg.params.Tstep-1;
    if isfield(cfg.params,'max_steps'), nsteps=min(nsteps,cfg.params.max_steps); end
    width=ceil(cfg.params.sm_eig*cfg.params.sketch_oversampling);
    keys={'gaussian_q2_recycle','gaussian_q1_refresh'}; allrows={};
    for ci=1:numel(cfg.cases)
        reference=[];
        for seed=cfg.seeds
            folder=fullfile(root,sprintf('seed_%d',seed),cfg.cases{ci});
            data=load(fullfile(folder,'solver_stats.mat'),'result','identity'); r=data.result;
            assert(isequaln(data.identity.config,cfg) && data.identity.seed==seed ...
                && strcmp(data.identity.case_name,cfg.cases{ci}),'Checkpoint identity differs.');
            assert(numel(r.rows)==2*nsteps && numel(r.physics)==nsteps);
            if isempty(reference)
                reference=r;
            else
                assert(isequaln(r.physics,reference.physics),'Physical trajectories differ across seeds.');
                assert(abs(r.tau-reference.tau)<1e-12*reference.tau,'Tau differs across seeds.');
            end
            if strcmp(cfg.cases{ci},'disk_static_nu_const')
                assert(all([r.physics(2:end).A_change]==0) && all([r.physics(2:end).D_change]==0) ...
                    && all([r.physics(2:end).coupling_change]==0));
            end
            for j=1:numel(r.rows)
                row=r.rows(j);
                assert(row.seed==seed && strcmp(row.case_name,cfg.cases{ci}));
                assert(row.orthogonality<1e-8 && row.coarse_chol_min_diag>0);
                assert(row.tau==r.tau && row.tau_eigen_residual<=1e-8);
                assert(row.basis_rank>0 && row.basis_rank<=width && row.rank_drop==width-row.basis_rank);
                assert(row.coarse_operator_columns==row.basis_rank);
                assert(isfinite(row.inverse_probe_residual) && row.schur_apply_probe_residual<1e-10);
                if strcmp(row.arm,keys{2})
                    assert(row.q==1 && row.basis_built_step==row.step ...
                        && row.exact_chol_builds==1 && row.inverse_applications==1 ...
                        && row.inverse_rhs_columns==width);
                else
                    assert(strcmp(row.arm,keys{1}) && row.q==2 && row.basis_built_step==1);
                    assert(row.exact_chol_builds==double(row.step==1));
                    assert(row.inverse_rhs_columns==2*width*double(row.step==1));
                end
                record=load(fullfile(folder,'diagnostics', ...
                    sprintf('%s_step_%03d.mat',row.arm,row.step)),'record');
                assert(isequaln(record.record.info,row) && all(isfinite(record.record.resvec)));
            end
            allrows{end+1}=r.rows; %#ok<AGROW>
        end
    end
    T=struct2table(vertcat(allrows{:}));
    assert(height(T)==numel(cfg.cases)*numel(cfg.seeds)*2*nsteps);
    groups=findgroups(T.case_name,T.seed,T.arm,T.step);
    assert(numel(unique(groups))==height(T),'Duplicate solver rows.');
    T.schur_tolerance_exceeded=double(T.schur_relres>cfg.params.SOLVER_TOL);
    T.kkt_tolerance_exceeded=double(T.kkt_relres>cfg.params.SOLVER_TOL);
    writetable(T,fullfile(root,'comparison_results.csv'));
    summaries={};
    for ci=1:numel(cfg.cases)
        for arm=1:2
            for seed=cfg.seeds
                z=T(strcmp(T.case_name,cfg.cases{ci}) & strcmp(T.arm,keys{arm}) & T.seed==seed,:);
                s=struct('case_name',cfg.cases{ci},'arm',keys{arm},'seed',seed, ...
                    'total_iterations',sum(z.iterations),'total_seconds',sum(z.algorithm_s), ...
                    'tau',z.tau(1),'step_operator_seconds',sum(z.step_operator_s), ...
                    'materialization_seconds',sum(z.materialization_s), ...
                    'chol_seconds',sum(z.exact_chol_s),'tau_seconds',sum(z.tau_setup_s), ...
                    'basis_seconds',sum(z.basis_setup_s),'coarse_seconds',sum(z.coarse_setup_s), ...
                    'pcg_seconds',sum(z.pcg_s),'recovery_seconds',sum(z.recovery_s), ...
                    'factorizations',sum(z.exact_chol_builds), ...
                    'inverse_rhs_columns',sum(z.inverse_rhs_columns), ...
                    'materialize_velocity_rhs_columns',sum(z.materialize_velocity_rhs_columns), ...
                    'forward_operator_columns',sum(z.coarse_operator_columns+z.pcg_operator_columns), ...
                    'nonzero_flags',sum(z.flag~=0),'max_schur_relres',max(z.schur_relres), ...
                    'max_kkt_relres',max(z.kkt_relres),'max_solution_error',max(z.solution_error), ...
                    'schur_tolerance_exceeded',sum(z.schur_tolerance_exceeded), ...
                    'kkt_tolerance_exceeded',sum(z.kkt_tolerance_exceeded), ...
                    'min_basis_rank',min(z.basis_rank),'rank_loss_steps',sum(z.rank_drop>0), ...
                    'max_inverse_probe_residual',max(z.inverse_probe_residual));
                summaries{end+1}=s; %#ok<AGROW>
            end
        end
    end
    summary=struct2table(vertcat(summaries{:}));
    writetable(summary,fullfile(root,'comparison_summary.csv'));
    make_plots(root,cfg,T,keys,nsteps);
    write_report(root,cfg,T,summary,keys,nsteps,width);
    fprintf('Validated %d Schur rows: flags=%d, Schur residual > tol=%d, KKT residual > tol=%d.\n', ...
        height(T),sum(T.flag~=0),sum(T.schur_tolerance_exceeded),sum(T.kkt_tolerance_exceeded));
end

function make_plots(root,cfg,T,keys,nsteps)
    folder=fullfile(root,'plots'); if ~isfolder(folder), mkdir(folder); end
    colors=[0 .45 .70; .84 .37 0]; labels={'q=2 recycled','q=1 rebuilt'};
    for ci=1:numel(cfg.cases)
        name=cfg.cases{ci};
        f=figure('Visible','off','Position',[50 50 1100 760]);
        layout=tiledlayout(f,2,2,'TileSpacing','compact');
        fields={'iterations','algorithm_s','inverse_rhs_columns','pcg_operator_columns'};
        titles={'PCG iterations','Cumulative attributed time (s)', ...
            'Cumulative Schur inverse RHS columns','Cumulative PCG Schur operator columns'};
        for panel=1:4
            ax=nexttile(layout); hold(ax,'on'); handles=gobjects(2,1);
            for arm=1:2
                Y=zeros(nsteps,numel(cfg.seeds));
                for si=1:numel(cfg.seeds)
                    z=sortrows(T(strcmp(T.case_name,name) & strcmp(T.arm,keys{arm}) ...
                        & T.seed==cfg.seeds(si),:),'step');
                    y=z.(fields{panel}); if panel>1, y=cumsum(y); end
                    Y(:,si)=y;
                    plot(ax,1:nsteps,y,'Color',.65+.35*colors(arm,:), ...
                        'LineWidth',.7,'HandleVisibility','off');
                end
                t=(1:nsteps)';
                fill(ax,[t;flipud(t)],[min(Y,[],2);flipud(max(Y,[],2))],colors(arm,:), ...
                    'FaceAlpha',.13,'EdgeColor','none','HandleVisibility','off');
                handles(arm)=plot(ax,t,median(Y,2),'Color',colors(arm,:),'LineWidth',1.7);
            end
            title(ax,titles{panel}); xlabel(ax,'Physical timestep'); grid(ax,'on');
        end
        lg=legend(ax,handles,labels,'Orientation','horizontal'); lg.Layout.Tile='south';
        title(layout,strrep(name,'_',' ')); drawnow;
        exportgraphics(f,fullfile(folder,[name '_comparison.png']),'Resolution',150); close(f);
        f=figure('Visible','off','Position',[50 50 1200 700]);
        layout=tiledlayout(f,2,3,'TileSpacing','compact');
        selected=unique([1,15,30,45,nsteps]); selected=selected(selected<=nsteps);
        styles={'-','--',':'};
        for step=selected
            ax=nexttile(layout); hold(ax,'on'); handles=gobjects(2,1);
            for arm=1:2
                for seed=cfg.seeds
                    p=fullfile(root,sprintf('seed_%d',seed),name,'diagnostics', ...
                        sprintf('%s_step_%03d.mat',keys{arm},step));
                    d=load(p,'record'); rv=d.record.resvec; rv=rv/max(rv(1),realmin);
                    handles(arm)=semilogy(ax,0:numel(rv)-1,max(rv,realmin), ...
                        'Color',colors(arm,:),'LineStyle',styles{seed});
                end
            end
            set(ax,'YScale','log'); title(ax,sprintf('Timestep %d',step));
            xlabel(ax,'PCG iteration'); ylabel(ax,'Reported residual / initial'); grid(ax,'on');
        end
        lg=legend(ax,handles,labels,'Orientation','horizontal'); lg.Layout.Tile='south';
        title(layout,[strrep(name,'_',' ') ' (solid: seed 1, dashed: seed 2, dotted: seed 3)']); drawnow;
        exportgraphics(f,fullfile(folder,[name '_residuals.png']),'Resolution',150); close(f);
    end
end

function write_report(root,cfg,T,summary,keys,nsteps,width)
    fid=fopen(fullfile(root,'comparison_report.md'),'w'); assert(fid>=0);
    cleanup=onCleanup(@()fclose(fid));
    fprintf(fid,'# Schur Gaussian q=2 recycling versus q=1 rebuilding\n\n');
    fprintf(fid,['The reduced SPD Schur system is solved with PCG and ' ...
        'P = I - V V^T + tau V (V^T S_i V)^{-1} V^T. ' ...
        'The recycled basis is orth(S_1^{-2} Omega); the rebuilt basis is ' ...
        'orth(S_i^{-1} Omega). q counts inverse applications. ' ...
        'Each rebuild uses the exact current dense Schur Cholesky. ' ...
        'Both arms rebuild their coarse matrices from current S_i.\n\n']);
    fprintf(fid,'Mesh h=%.3g, %d timesteps, width %d, seeds 1-3, zero initial guesses, PCG tolerance %.1g.\n\n', ...
        cfg.params.h0,nsteps,width,cfg.params.SOLVER_TOL);
    fprintf(fid,['The mesh and direct-KKT physical trajectory are shared across seeds. ' ...
        'Within each seed, both arms and all steps share the same Gaussian Omega. ' ...
        'Tau is lambda_max(S_1), held fixed and shared by both arms.\n\n']);
    fprintf(fid,'| Case | q=2 iterations, median [range] | q=1 iterations, median [range] | Paired q=1 iteration change | q=1/q=2 time | Tau |\n');
    fprintf(fid,'|---|---:|---:|---:|---:|---:|\n');
    for ci=1:numel(cfg.cases)
        name=cfg.cases{ci};
        a=sortrows(summary(strcmp(summary.case_name,name)&strcmp(summary.arm,keys{1}),:),'seed');
        b=sortrows(summary(strcmp(summary.case_name,name)&strcmp(summary.arm,keys{2}),:),'seed');
        x=a.total_iterations; y=b.total_iterations;
        fprintf(fid,'| %s | %.0f [%g, %g] | %.0f [%g, %g] | %+.1f%% | %.2fx | %.6g |\n', ...
            name,median(x),min(x),max(x),median(y),min(y),max(y), ...
            100*median(y./x-1),median(b.total_seconds./a.total_seconds),a.tau(1));
    end
    fprintf(fid,'\n| Case | Arm | Nonzero flags | Max Schur residual | Max KKT residual | Max solution error | Min rank |\n');
    fprintf(fid,'|---|---|---:|---:|---:|---:|---:|\n');
    for ci=1:numel(cfg.cases)
        for arm=1:2
            z=T(strcmp(T.case_name,cfg.cases{ci})&strcmp(T.arm,keys{arm}),:);
            fprintf(fid,'| %s | %s | %d/%d | %.3g | %.3g | %.3g | %d |\n', ...
                cfg.cases{ci},keys{arm},sum(z.flag~=0),height(z),max(z.schur_relres), ...
                max(z.kkt_relres),max(z.solution_error),min(z.basis_rank));
        end
    end
    fprintf(fid,'\nValidated %d records. Nonzero flags: %d. Schur residual above tolerance: %d; KKT residual above tolerance: %d.\n\n', ...
        height(T),sum(T.flag~=0),sum(T.schur_tolerance_exceeded),sum(T.kkt_tolerance_exceeded));
    fprintf(fid,'Rank-loss records: %d. Maximum current inverse probe residual: %.3g.\n\n', ...
        sum(T.rank_drop>0),max(T.inverse_probe_residual));
    fprintf(fid,['Two inverse applications emphasize small eigenvalues more strongly than one. ' ...
        'A fresh exact inverse application still yields an approximate finite-width subspace. ' ...
        'The first timestep compares two fresh bases; later differences combine power count and ' ...
        'basis aging. The constant control has no operator drift but its two arms need not coincide ' ...
        'because their power counts differ. This experiment does not isolate the causal effect of ' ...
        'refreshing at a fixed power count.\n\n' ...
        'Attributed time includes current Schur construction, required dense materialization and ' ...
        'Cholesky, initial tau selection, basis orthogonalization, coarse setup, PCG, and velocity ' ...
        'recovery. Shared work is charged fully to every arm requiring it. Direct reference solves, ' ...
        'diagnostics, and plotting are excluded. Cholesky time is not treated as free just because ' ...
        'the experiment has a reference solution. An exact Schur factor also permits direct solution; ' ...
        'these results compare basis strategies, not superiority over a direct solver.\n\n' ...
        'Inverse RHS columns, dense-materialization velocity RHS columns, and matrix-free Schur ' ...
        'operator columns are distinct work units. Timings are measured once per seed. ' ...
        'Thin curves show seeds, thick curves medians, and bands the ranges. ' ...
        'Residual histories are PCG-reported Schur histories; recomputed Schur and full KKT ' ...
        'residuals and direct-reference solution errors are reported separately.\n\n']);
    fprintf(fid,'[Per-step results](comparison_results.csv) · [Per-seed summary](comparison_summary.csv)\n\n');
    for ci=1:numel(cfg.cases)
        name=cfg.cases{ci};
        fprintf(fid,'- [%s comparison](plots/%s_comparison.png), [residual histories](plots/%s_residuals.png)\n',name,name,name);
    end
end
