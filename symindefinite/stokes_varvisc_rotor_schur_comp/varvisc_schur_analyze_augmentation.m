function T = varvisc_schur_analyze_augmentation(root)
%VARVISC_SCHUR_ANALYZE_AUGMENTATION Validate every checkpoint and report pairs.
    data=load(fullfile(root,'experiment_config.mat'),'config'); cfg=data.config;
    nsteps=cfg.params.Tstep-1;
    if isfield(cfg.params,'max_steps'), nsteps=min(nsteps,cfg.params.max_steps); end
    width=ceil(cfg.params.sm_eig*cfg.params.sketch_oversampling);
    keys={'gaussian_q2_recycle','gaussian_q2_augmented'}; allrows={};
    for ci=1:numel(cfg.cases)
        reference=[];
        for seed=cfg.seeds
            folder=fullfile(root,sprintf('seed_%d',seed),cfg.cases{ci});
            checkpoint=fullfile(folder,'solver_stats.mat');
            assert(isfile(checkpoint),'varviscSchur:incompleteExperiment', ...
                'Missing completed checkpoint: %s',checkpoint);
            data=load(checkpoint,'result','identity'); r=data.result;
            assert(isequaln(data.identity.config,cfg) && data.identity.seed==seed ...
                && strcmp(data.identity.case_name,cfg.cases{ci}),'Checkpoint identity differs.');
            assert(numel(r.rows)==2*nsteps && numel(r.physics)==nsteps);
            assert(isequal([r.physics.step],1:nsteps));
            assert(all([r.physics.reference_relres]<1e-8),'Direct reference residual failed.');
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
                z=r.rows(j); first=double(z.step==1);
                assert(z.seed==seed && strcmp(z.case_name,cfg.cases{ci}));
                assert(z.step>=1 && z.step<=nsteps && z.step==fix(z.step));
                arm=find(strcmp(z.arm,keys)); assert(isscalar(arm));
                assert(z.augment_requested==(arm-1)*cfg.params.AUGMENT_M);
                assert(z.base_rank>0 && z.base_rank<=width && z.rank_drop==width-z.base_rank);
                assert(z.total_rank==z.base_rank+z.augment_rank);
                assert(z.augment_rank>=0 && z.augment_rank<=min(z.augment_requested, ...
                    r.physics(z.step).nS-z.base_rank));
                assert(max([z.base_orthogonality,z.vw_orth,z.ww_orth,z.arnoldi_relation])<1e-8);
                assert(z.coarse_chol_min_diag>0 && z.tau==r.tau && z.tau_eigen_residual<=1e-8);
                assert(z.q==2 && z.basis_built_step==1 && z.basis_rebuilt==first);
                assert(z.exact_chol_builds==first && z.inverse_applications==2*first);
                assert(z.inverse_rhs_columns==2*width*first);
                assert(z.materialize_velocity_rhs_columns==r.physics(z.step).nS*first);
                if ~first
                    assert(z.materialization_s==0 && z.exact_chol_s==0 ...
                        && z.basis_setup_s==0 && z.tau_setup_s==0 && z.tau_operator_columns==0);
                end
                assert(z.arnoldi_operator_columns==z.augment_rank);
                assert(z.coarse_operator_columns==z.total_rank);
                assert(z.forward_operator_columns==z.tau_operator_columns ...
                    +z.arnoldi_operator_columns+z.coarse_operator_columns+z.pcg_operator_columns);
                assert(z.inverse_probe_residual<1e-8 && z.schur_apply_probe_residual<1e-10);
                times=[z.step_operator_s,z.materialization_s,z.exact_chol_s,z.tau_setup_s, ...
                    z.basis_setup_s,z.arnoldi_s,z.coarse_setup_s,z.pcg_s,z.recovery_s];
                assert(all(isfinite(times) & times>=0));
                assert(abs(z.algorithm_s-sum(times))<1e-12*max(z.algorithm_s,1));
                d=load(fullfile(folder,'diagnostics',sprintf('%s_step_%03d.mat',z.arm,z.step)),'record');
                assert(isequaln(d.record.info,z) && isvector(d.record.resvec));
                assert(isequal(size(d.record.arnoldi_H),[z.augment_rank,z.augment_rank]));
            end
            allrows{end+1}=r.rows; %#ok<AGROW>
        end
    end
    T=struct2table(vertcat(allrows{:}));
    assert(height(T)==numel(cfg.cases)*numel(cfg.seeds)*2*nsteps);
    groups=findgroups(T.case_name,T.seed,T.arm,T.step);
    assert(numel(unique(groups))==height(T),'Duplicate solver rows.');
    T.schur_tolerance_exceeded=~isfinite(T.schur_relres) | T.schur_relres>cfg.params.SOLVER_TOL;
    T.kkt_tolerance_exceeded=~isfinite(T.kkt_relres) | T.kkt_relres>cfg.params.SOLVER_TOL;
    T.solution_tolerance_exceeded=~isfinite(T.solution_error) | T.solution_error>cfg.solution_error_tol;
    T.valid=T.flag==0 & ~T.schur_tolerance_exceeded & ~T.kkt_tolerance_exceeded ...
        & ~T.solution_tolerance_exceeded;
    T.augmentation_shortfall=T.augment_rank<T.augment_requested;
    writetable(T,fullfile(root,'augmentation_results.csv'));
    validation=T(:,{'case_name','seed','arm','step','flag','schur_relres','kkt_relres', ...
        'solution_error','valid','augmentation_shortfall','base_orthogonality', ...
        'vw_orth','ww_orth','arnoldi_relation'});
    writetable(validation,fullfile(root,'augmentation_validation.csv'));
    summaries={}; pairs={};
    time_fields={'step_operator','materialization','exact_chol','tau_setup','basis_setup', ...
        'arnoldi','coarse_setup','pcg','recovery'};
    for ci=1:numel(cfg.cases)
        for seed=cfg.seeds
            arms=cell(2,1);
            for arm=1:2
                z=sortrows(T(strcmp(T.case_name,cfg.cases{ci}) & T.seed==seed ...
                    & strcmp(T.arm,keys{arm}),:),'step');
                assert(isequal(z.step,(1:nsteps)')); arms{arm}=z;
                s=struct('case_name',cfg.cases{ci},'seed',seed,'arm',keys{arm}, ...
                    'total_iterations',sum(z.iterations),'total_seconds',sum(z.algorithm_s), ...
                    'forward_operator_columns',sum(z.forward_operator_columns), ...
                    'inverse_rhs_columns',sum(z.inverse_rhs_columns), ...
                    'materialize_velocity_rhs_columns',sum(z.materialize_velocity_rhs_columns), ...
                    'factorizations',sum(z.exact_chol_builds),'tau',z.tau(1), ...
                    'valid_steps',sum(z.valid),'nonzero_flags',sum(z.flag~=0), ...
                    'max_schur_relres',max(z.schur_relres),'max_kkt_relres',max(z.kkt_relres), ...
                    'max_solution_error',max(z.solution_error),'min_base_rank',min(z.base_rank), ...
                    'min_augment_rank',min(z.augment_rank),'shortfalls',sum(z.augmentation_shortfall));
                for f=time_fields, s.([f{1} '_seconds'])=sum(z.([f{1} '_s'])); end
                summaries{end+1}=s; %#ok<AGROW>
            end
            a=arms{1}; b=arms{2};
            shared={'base_rank','tau','basis_setup_s','exact_chol_s','tau_setup_s', ...
                'step_operator_s','materialization_s','inverse_rhs_columns','tau_operator_columns'};
            for f=shared, assert(isequal(a.(f{1}),b.(f{1})),'Paired setup differs.'); end
            valid=a.valid & b.valid;
            p=struct('case_name',cfg.cases{ci},'seed',seed,'valid_pairs',sum(valid), ...
                'baseline_iterations',sum(a.iterations),'augmented_iterations',sum(b.iterations), ...
                'iteration_savings_percent',100*(1-sum(b.iterations)/max(sum(a.iterations),realmin)), ...
                'forward_work_ratio',sum(b.forward_operator_columns)/sum(a.forward_operator_columns), ...
                'time_ratio',sum(b.algorithm_s)/sum(a.algorithm_s), ...
                'valid_pair_iteration_savings_percent',NaN);
            if any(valid) && sum(a.iterations(valid))>0
                p.valid_pair_iteration_savings_percent=100*(1-sum(b.iterations(valid))/sum(a.iterations(valid)));
            end
            pairs{end+1}=p; %#ok<AGROW>
        end
    end
    summary=struct2table(vertcat(summaries{:})); paired=struct2table(vertcat(pairs{:}));
    writetable(summary,fullfile(root,'augmentation_summary.csv'));
    writetable(paired,fullfile(root,'augmentation_paired_summary.csv'));
    make_plots(root,cfg,T,keys,nsteps);
    write_report(root,cfg,T,summary,paired,nsteps,width);
    fprintf('Validated %d augmented Schur records; accurate solves %d/%d; shortfalls %d.\n', ...
        height(T),sum(T.valid),height(T),sum(T.augmentation_shortfall));
end

function make_plots(root,cfg,T,keys,nsteps)
    folder=fullfile(root,'plots'); if ~isfolder(folder), mkdir(folder); end
    colors=[0 .45 .70; .84 .37 0]; labels={'Frozen q=2','Frozen q=2 + Arnoldi (m=20)'};
    for ci=1:numel(cfg.cases)
        name=cfg.cases{ci};
        f=figure('Visible','off','Position',[50 50 1200 850]);
        layout=tiledlayout(f,2,3,'TileSpacing','compact');
        fields={'iterations','algorithm_s','forward_operator_columns','schur_relres','kkt_relres','solution_error'};
        titles={'PCG iterations','Cumulative attributed time (s)','Cumulative forward operator columns', ...
            'True Schur relative residual','Recovered KKT relative residual','Relative solution error'};
        for panel=1:6
            ax=nexttile(layout); set(ax,'FontSize',10); hold(ax,'on'); handles=gobjects(2,1);
            for arm=1:2
                Y=zeros(nsteps,numel(cfg.seeds));
                for si=1:numel(cfg.seeds)
                    z=sortrows(T(strcmp(T.case_name,name) & strcmp(T.arm,keys{arm}) ...
                        & T.seed==cfg.seeds(si),:),'step');
                    y=z.(fields{panel}); if panel==2 || panel==3, y=cumsum(y); end
                    if panel>=4, y=max(y,realmin); end
                    Y(:,si)=y;
                    plot(ax,1:nsteps,y,'Color',.65+.35*colors(arm,:), ...
                        'LineWidth',.7,'HandleVisibility','off');
                end
                t=(1:nsteps)';
                fill(ax,[t;flipud(t)],[min(Y,[],2);flipud(max(Y,[],2))],colors(arm,:), ...
                    'FaceAlpha',.13,'EdgeColor','none','HandleVisibility','off');
                handles(arm)=plot(ax,t,median(Y,2),'Color',colors(arm,:),'LineWidth',1.7);
            end
            if panel>=4
                set(ax,'YScale','log'); threshold=cfg.params.SOLVER_TOL;
                if panel==6, threshold=cfg.solution_error_tol; end
                yline(ax,threshold,':k','HandleVisibility','off');
            end
            if panel>=3
                ax.YAxis.Exponent=0;
                ytickformat(ax,'%.1e');
            end
            title(ax,titles{panel}); xlabel(ax,'Physical timestep'); grid(ax,'on');
        end
        lg=legend(ax,handles,labels,'Orientation','horizontal','FontSize',10); lg.Layout.Tile='south';
        title(layout,strrep(name,'_',' '),'FontSize',14); drawnow;
        exportgraphics(f,fullfile(folder,[name '_comparison.png']),'Resolution',150); close(f);
        f=figure('Visible','off','Position',[50 50 1200 700]);
        layout=tiledlayout(f,2,3,'TileSpacing','compact');
        selected=unique([1,15,30,45,nsteps]); selected=selected(selected<=nsteps);
        styles={'-','--',':'};
        for step=selected
            ax=nexttile(layout); set(ax,'FontSize',10); hold(ax,'on'); handles=gobjects(2,1);
            for arm=1:2
                for si=1:numel(cfg.seeds)
                    seed=cfg.seeds(si);
                    p=fullfile(root,sprintf('seed_%d',seed),name,'diagnostics', ...
                        sprintf('%s_step_%03d.mat',keys{arm},step));
                    d=load(p,'record'); rv=d.record.resvec; rv=rv/max(rv(1),realmin);
                    handles(arm)=semilogy(ax,0:numel(rv)-1,max(rv,realmin), ...
                        'Color',colors(arm,:),'LineStyle',styles{si});
                end
            end
            set(ax,'YScale','log'); title(ax,sprintf('Timestep %d',step));
            xlabel(ax,'PCG iteration'); ylabel(ax,'Reported residual / initial'); grid(ax,'on');
        end
        lg=legend(ax,handles,labels,'Orientation','horizontal','FontSize',10); lg.Layout.Tile='south';
        title(layout,[strrep(name,'_',' ') ' (solid: seed 1, dashed: seed 2, dotted: seed 3)'],'FontSize',14);
        drawnow; exportgraphics(f,fullfile(folder,[name '_residuals.png']),'Resolution',150); close(f);
        f=figure('Visible','off','Position',[50 50 1200 380]);
        layout=tiledlayout(f,1,3,'TileSpacing','compact');
        fields={'vw_orth','ww_orth','arnoldi_relation'};
        titles={'||V^T W||_F','||W^T W - I||_F','Relative Arnoldi recurrence error'};
        for panel=1:3
            ax=nexttile(layout); set(ax,'FontSize',10); hold(ax,'on');
            for seed=cfg.seeds
                z=sortrows(T(strcmp(T.case_name,name)&strcmp(T.arm,keys{2})&T.seed==seed,:),'step');
                semilogy(ax,z.step,max(z.(fields{panel}),realmin),'DisplayName',sprintf('Seed %d',seed));
            end
            set(ax,'YScale','log'); title(ax,titles{panel}); xlabel(ax,'Physical timestep'); grid(ax,'on');
        end
        lg=legend(ax,'Orientation','horizontal','FontSize',10); lg.Layout.Tile='south';
        title(layout,strrep(name,'_',' '),'FontSize',14); drawnow;
        exportgraphics(f,fullfile(folder,[name '_orthogonality.png']),'Resolution',150); close(f);
    end
end

function write_report(root,cfg,T,summary,paired,nsteps,width)
    fid=fopen(fullfile(root,'augmentation_report.md'),'w'); assert(fid>=0);
    cleanup=onCleanup(@()fclose(fid));
    fprintf(fid,'# Augmented Gaussian recycling on the SPD Schur system\n\n');
    fprintf(fid,['Both arms share V = orth(S_1^{-2} Omega), retained in fixed reduced coordinates. ' ...
        'The augmented arm adds W = Arnoldi_20((I-VV^T) S_i (I-VV^T), (I-VV^T) b_i) ' ...
        'and discards W after each solve. For Z = V or [V,W], PCG uses ' ...
        'P = I-ZZ^T + tau Z (Z^T S_i Z)^{-1} Z^T, with shared tau = lambda_max(S_1).\n\n']);
    fprintf(fid,['Mesh h=%.3g, dt=%.3g, %d timesteps, %d Gaussian columns, seeds 1-3, ' ...
        'zero starts, PCG tolerance %.1g, cap %d, four MATLAB threads. ' ...
        'The saved mesh and direct-KKT trajectory are identical across seeds.\n\n'], ...
        cfg.params.h0,cfg.params.dt,nsteps,width,cfg.params.SOLVER_TOL,cfg.params.SOLVER_MAXIT);
    fprintf(fid,'| Case | Baseline total iterations, median [range] | Augmented total iterations, median [range] | Iterations saved | Forward work ratio [range] | Time ratio [range] | Valid pairs |\n');
    fprintf(fid,'|---|---:|---:|---:|---:|---:|---:|\n');
    for ci=1:numel(cfg.cases)
        p=paired(strcmp(paired.case_name,cfg.cases{ci}),:);
        a=p.baseline_iterations; b=p.augmented_iterations;
        fprintf(fid,'| %s | %.0f [%g, %g] | %.0f [%g, %g] | %.2f%% | %.3fx [%.3f, %.3f] | %.3fx [%.3f, %.3f] | %d/%d |\n', ...
            cfg.cases{ci},median(a),min(a),max(a),median(b),min(b),max(b), ...
            median(p.iteration_savings_percent),median(p.forward_work_ratio),min(p.forward_work_ratio),max(p.forward_work_ratio), ...
            median(p.time_ratio),min(p.time_ratio),max(p.time_ratio), ...
            sum(p.valid_pairs),numel(cfg.seeds)*nsteps);
    end
    fprintf(fid,['\nRatios are augmented/baseline; values below one indicate savings. ' ...
        'Percentages and ratios are medians of paired per-seed totals. All rows remain in totals, ' ...
        'including failures. Equal-accuracy comparisons require both solves to have flag 0, ' ...
        'true Schur and KKT residuals <= %.1g, and relative solution error <= %.1g. ' ...
        'The paired CSV also reports iteration savings restricted to valid pairs.\n\n'], ...
        cfg.params.SOLVER_TOL,cfg.solution_error_tol);
    fprintf(fid,'Validated %d records; %d valid solves; %d nonzero flags; %d augmentation shortfalls.\n\n', ...
        height(T),sum(T.valid),sum(T.flag~=0),sum(T.augmentation_shortfall));
    fprintf(fid,'| Case | Arm | Max Schur residual | Max KKT residual | Max solution error | Min base / added rank |\n');
    fprintf(fid,'|---|---|---:|---:|---:|---:|\n');
    for j=1:numel(cfg.cases)
        for key={'gaussian_q2_recycle','gaussian_q2_augmented'}
            s=summary(strcmp(summary.case_name,cfg.cases{j}) & strcmp(summary.arm,key{1}),:);
            fprintf(fid,'| %s | %s | %.3g | %.3g | %.3g | %d / %d |\n',cfg.cases{j},key{1}, ...
                max(s.max_schur_relres),max(s.max_kkt_relres),max(s.max_solution_error), ...
                min(s.min_base_rank),min(s.min_augment_rank));
        end
    end
    fprintf(fid,'\nMaximum ||V^T W||_F: %.3g; ||W^T W-I||_F: %.3g; relative Arnoldi recurrence error: %.3g.\n\n', ...
        max(T.vw_orth),max(T.ww_orth),max(T.arnoldi_relation));
    fprintf(fid,['Attributed time includes Schur construction, initial dense materialization, exact Cholesky, ' ...
        'tau selection and basis construction, plus current Arnoldi, coarse setup, PCG and recovery. ' ...
        'Shared initial costs are charged fully to each arm. Reference solves, numerical diagnostics ' ...
        'and output are excluded. Timings are single measurements per seed and can vary with machine load.\n\n' ...
        'Forward work includes Arnoldi, coarse construction, PCG and initial tau iteration. ' ...
        'Inverse RHS columns and dense-materialization velocity RHS columns are reported separately. ' ...
        'Both arms require only one dense Schur factorization per trajectory; the velocity factor ' ...
        'inside the matrix-free Schur operator is rebuilt each step. The initial exact factor also ' ...
        'permits direct solution, so this study compares recycling strategies rather than claiming ' ...
        'superiority over direct solving.\n\n']);
    fprintf(fid,['The static case controls operator aging but its RHS can evolve. Fresh RHS-dependent ' ...
        'augmentation may help even when the operator is constant. No iteration or runtime ' ...
        'improvement is assumed by validation.\n\n']);
    for ci=1:numel(cfg.cases)
        name=cfg.cases{ci};
        fprintf(fid,'- **%s:** [comparison](plots/%s_comparison.png), [residual histories](plots/%s_residuals.png), [orthogonality](plots/%s_orthogonality.png).\n',name,name,name,name);
    end
    fprintf(fid,'\nData: [per step](augmentation_results.csv), [summary](augmentation_summary.csv), [paired summary](augmentation_paired_summary.csv), [validation](augmentation_validation.csv).\n');
end
