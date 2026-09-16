function T = varvisc_analyze_gaussian_refresh(root)
%VARVISC_ANALYZE_GAUSSIAN_REFRESH Rebuild tables/figures from saved evidence.
    loaded=load(fullfile(root,'experiment_config.mat'),'config'); cfg=loaded.config;
    keys={'gaussian_q2_recycle','gaussian_q1_refresh'};
    nsteps=cfg.params.Tstep-1; width=round(cfg.params.DEFLAT_SM_EIG*cfg.params.SKETCH_OVERSAMPLE);
    rows={};
    for j=1:numel(cfg.cases)
        reference=[];
        for seed=cfg.seeds
            folder=fullfile(root,sprintf('seed_%d',seed),cfg.cases{j});
            saved=load(fullfile(folder,'solver_stats.mat'),'st','identity'); st=saved.st;
            assert(isequaln(saved.identity.config,cfg) && saved.identity.seed==seed ...
                && strcmp(saved.identity.case_name,cfg.cases{j}),'Checkpoint identity mismatch.');
            physical_fields={'sys_size','nC','diffK','coupling_change','nu_contrast','backslash_relres'};
            if isempty(reference)
                reference=st;
            else
                for fi=1:numel(physical_fields)
                    field=physical_fields{fi};
                    assert(isequaln(st.(field),reference.(field)), ...
                        'Physical trajectory differs between paired seeds.');
                end
            end
            if strcmp(cfg.cases{j},'disk_static_nu_const')
                assert(all(st.diffK(2:end)==0) && all(st.coupling_change(2:end)==0));
            end
            for a=1:2
                key=keys{a};
                for step=1:nsteps
                    d=load(fullfile(folder,'diagnostics',sprintf('%s_step_%03d.mat',key,step)),'record');
                    rec=d.record; r=rec.info;
                    r.case_name=cfg.cases{j}; r.seed=seed; r.arm=key; r.step=step;
                    r.iterations=st.solver_its.(key)(step);
                    r.flag=st.solver_flag.(key)(step);
                    r.reported_relres=st.solver_relres.(key)(step);
                    r.physical_relres=st.solver_true_relres.(key)(step);
                    r.solution_error=st.solver_err.(key)(step);
                    assert(rec.step==step && rec.iters==r.iterations && rec.flag==r.flag);
                    assert(~isempty(rec.resvec) && all(isfinite(rec.resvec)));
                    assert(r.orthogonality<1e-8 && r.coarse_min_eig>0);
                    assert(r.basis_rank<=width && r.basis_rank>0);
                    if a==2
                        assert(r.basis_built_step==step && r.exact_ldl_builds==1 ...
                            && r.inverse_applications==1 && r.inverse_rhs_columns==width);
                    elseif r.basis_rebuilt
                        assert(step==1 || r.forced_rebuild==1, ...
                            'Recycled arm rebuilt unexpectedly.');
                        assert(r.inverse_applications==2 && r.inverse_rhs_columns==2*width);
                    else
                        assert(r.exact_ldl_builds==0 && r.inverse_applications==0);
                    end
                    r.physical_tolerance_exceeded=double(r.physical_relres>cfg.params.SOLVER_TOL);
                    r.solution_error_above_1e_minus8=double(r.solution_error>1e-8);
                    rows{end+1}=r; %#ok<AGROW>
                end
            end
        end
    end
    T=struct2table(vertcat(rows{:}));
    assert(height(T)==numel(cfg.cases)*numel(cfg.seeds)*2*nsteps);
    writetable(T,fullfile(root,'comparison_results.csv'));
    summaries={};
    for j=1:numel(cfg.cases)
        for a=1:2
            for seed=cfg.seeds
                ix=strcmp(T.case_name,cfg.cases{j}) & strcmp(T.arm,keys{a}) & T.seed==seed;
                z=T(ix,:);
                s=struct('case_name',cfg.cases{j},'arm',keys{a},'seed',seed, ...
                    'total_iterations',sum(z.iterations),'total_seconds',sum(z.algorithm_s), ...
                    'exact_ldl_seconds',sum(z.exact_ldl_s),'basis_seconds',sum(z.basis_setup_s), ...
                    'transport_seconds',sum(z.transport_s),'coarse_seconds',sum(z.coarse_setup_s), ...
                    'minres_seconds',sum(z.minres_s),'factorizations',sum(z.exact_ldl_builds), ...
                    'inverse_rhs_columns',sum(z.inverse_rhs_columns), ...
                    'forward_operator_columns',sum(z.coarse_operator_columns+z.minres_operator_columns), ...
                    'nonzero_flags',sum(z.flag~=0),'max_physical_relres',max(z.physical_relres), ...
                    'max_solution_error',max(z.solution_error), ...
                    'physical_tolerance_exceeded',sum(z.physical_tolerance_exceeded), ...
                    'rank_loss_steps',sum(z.rank_drop>0 | z.basis_numerical_rank<width), ...
                    'forced_rebuilds',sum(z.forced_rebuild));
                summaries{end+1}=s; %#ok<AGROW>
            end
        end
    end
    S=struct2table(vertcat(summaries{:}));
    writetable(S,fullfile(root,'comparison_summary.csv'));
    plots=fullfile(root,'plots'); if ~isfolder(plots), mkdir(plots); end
    colors=[0 .45 .70; .84 .37 0]; labels={'q=2 recycled','q=1 rebuilt'};
    for j=1:numel(cfg.cases)
        cname=cfg.cases{j};
        f=figure('Visible','off','Position',[50 50 1100 760]);
        layout=tiledlayout(f,2,2,'TileSpacing','compact');
        fields={'iterations','algorithm_s','inverse_rhs_columns','minres_operator_columns'};
        titles={'MINRES iterations','Cumulative attributed time (s)', ...
            'Cumulative inverse RHS columns','Cumulative MINRES operator columns'};
        for panel=1:4
            ax=nexttile(layout); hold(ax,'on'); handles=gobjects(2,1);
            for a=1:2
                Y=zeros(nsteps,numel(cfg.seeds));
                for si=1:numel(cfg.seeds)
                    ix=strcmp(T.case_name,cname) & strcmp(T.arm,keys{a}) & T.seed==cfg.seeds(si);
                    z=sortrows(T(ix,:),'step'); y=z.(fields{panel});
                    if panel>1, y=cumsum(y); end
                    Y(:,si)=y;
                    plot(ax,1:nsteps,y,'Color',.65+.35*colors(a,:), ...
                        'LineWidth',.7,'HandleVisibility','off');
                end
                t=(1:nsteps)';
                fill(ax,[t;flipud(t)],[min(Y,[],2);flipud(max(Y,[],2))],colors(a,:), ...
                    'FaceAlpha',.13,'EdgeColor','none','HandleVisibility','off');
                handles(a)=plot(ax,t,median(Y,2),'Color',colors(a,:),'LineWidth',1.7);
            end
            title(ax,titles{panel}); xlabel(ax,'Physical timestep'); grid(ax,'on');
        end
        lg=legend(ax,handles,labels,'Orientation','horizontal');
        lg.Layout.Tile='south';
        title(layout,strrep(cname,'_',' '));
        drawnow;
        exportgraphics(f,fullfile(plots,[cname '_comparison.png']),'Resolution',150); close(f);
        f=figure('Visible','off','Position',[50 50 1200 700]);
        layout=tiledlayout(f,2,3,'TileSpacing','compact');
        selected=unique([1,15,30,45,nsteps]); selected=selected(selected<=nsteps);
        for step=selected
            ax=nexttile(layout); hold(ax,'on'); handles=gobjects(2,1);
            for a=1:2
                for seed=cfg.seeds
                    folder=fullfile(root,sprintf('seed_%d',seed),cname,'diagnostics');
                    d=load(fullfile(folder,sprintf('%s_step_%03d.mat',keys{a},step)),'record');
                    rv=d.record.resvec; rv=rv/max(rv(1),realmin);
                    handles(a)=semilogy(ax,0:numel(rv)-1,max(rv,realmin), ...
                        'Color',colors(a,:),'LineStyle',seed_style(seed));
                end
            end
            set(ax,'YScale','log'); title(ax,sprintf('Timestep %d',step));
            xlabel(ax,'MINRES iteration'); ylabel(ax,'Reported residual / initial'); grid(ax,'on');
        end
        lg=legend(ax,handles,labels,'Orientation','horizontal');
        lg.Layout.Tile='south';
        title(layout,[strrep(cname,'_',' ') ' (solid: seed 1, dashed: seed 2, dotted: seed 3)']);
        drawnow;
        exportgraphics(f,fullfile(plots,[cname '_residuals.png']),'Resolution',150); close(f);
    end
    write_report(root,cfg,T,S,keys);
    fprintf('Validated %d rows. Nonzero flags: %d; physical residual > tolerance: %d.\n', ...
        height(T),sum(T.flag~=0),sum(T.physical_tolerance_exceeded));
end

function style=seed_style(seed)
    styles={'-','--',':'}; style=styles{seed};
end

function write_report(root,cfg,T,S,keys)
    fid=fopen(fullfile(root,'comparison_report.md'),'w'); assert(fid>=0);
    cleanup=onCleanup(@()fclose(fid));
    fprintf(fid,'# Gaussian q=2 recycling versus q=1 rebuilding\n\n');
    fprintf(fid,['Both arms use B_i = C_i^T K_i^{-1} C_i, q counts inverse applications, ' ...
        'and every solve uses the current ILDL smoother and coarse correction. ' ...
        'The recycled arm preserves the step-1 physical subspace; the refreshed arm ' ...
        'uses an exact LDL of the current K_i.\n\n']);
    fprintf(fid,'Mesh h=%.3g; %d timesteps; seeds 1, 2, 3; width %d; tolerance %.1g.\n\n', ...
        cfg.params.h0,cfg.params.Tstep-1,cfg.params.DEFLAT_SM_EIG*cfg.params.SKETCH_OVERSAMPLE,cfg.params.SOLVER_TOL);
    fprintf(fid,['| Case | q=2 total iterations, median [range] | q=1 total iterations, median [range] ' ...
        '| Paired iteration reduction, median | q=1/q=2 time, median |\n|---|---:|---:|---:|---:|\n']);
    for j=1:numel(cfg.cases)
        c=cfg.cases{j}; base=sortrows(S(strcmp(S.case_name,c)&strcmp(S.arm,keys{1}),:),'seed');
        fresh=sortrows(S(strcmp(S.case_name,c)&strcmp(S.arm,keys{2}),:),'seed');
        v=base.total_iterations; w=fresh.total_iterations;
        fprintf(fid,'| %s | %.0f [%g, %g] | %.0f [%g, %g] | %.1f%% | %.2fx |\n', ...
            c,median(v),min(v),max(v),median(w),min(w),max(w), ...
            100*median(1-w./v),median(fresh.total_seconds./base.total_seconds));
    end
    fprintf(fid,'\n| Case | Arm | Nonzero flags | Physical residual > tolerance | Max physical residual | Max solution error |\n');
    fprintf(fid,'|---|---|---:|---:|---:|---:|\n');
    for j=1:numel(cfg.cases)
        for a=1:2
            ix=strcmp(T.case_name,cfg.cases{j}) & strcmp(T.arm,keys{a});
            z=T(ix,:);
            fprintf(fid,'| %s | %s | %d/%d | %d/%d | %.3g | %.3g |\n', ...
                cfg.cases{j},keys{a},sum(z.flag~=0),height(z), ...
                sum(z.physical_tolerance_exceeded),height(z),max(z.physical_relres),max(z.solution_error));
        end
    end
    fprintf(fid,'\nNonzero solver flags: %d/%d. Physical residual above %.1g: %d/%d.\n\n', ...
        sum(T.flag~=0),height(T),cfg.params.SOLVER_TOL,sum(T.physical_tolerance_exceeded),height(T));
    fprintf(fid,'Maximum physical relative residual: %.3g; maximum reference-solution error: %.3g.\n\n', ...
        max(T.physical_relres),max(T.solution_error));
    fprintf(fid,'Rank-loss records: %d; forced dimension rebuilds: %d.\n\n', ...
        sum(S.rank_loss_steps),sum(S.forced_rebuilds));
    fprintf(fid,['Two inverse applications weight an eigencomponent by |lambda|^{-2}; ' ...
        'one weights it by |lambda|^{-1}. The stronger initial filtering can help ' ...
        'capture near-zero modes, but the recycled subspace can become stale as K_i changes. ' ...
        'Rebuilding is optional: it trades spectral refinement for current-operator information ' ...
        'and pays a new factorization at every step. This two-arm experiment changes both q ' ...
        'and refresh cadence, so it cannot separately estimate their causal effects.\n\n' ...
        'The refreshed sketch needs the current inverse: reusing a step-1 LDL with ' ...
        'the current C_i would apply C_i^T K_1^{-1} C_i instead of B_i. An exact LDL ' ...
        'of K_i also permits a direct solve of its RHS. This experiment assesses ' ...
        'basis strategies and does not establish a speed advantage over direct solution.\n\n' ...
        'All timings include attributed ILDL, exact LDL, basis, transport, coarse setup and MINRES ' ...
        'costs. Shared same-step factors are charged fully to every arm needing them. ' ...
        'Basis setup includes numerical-rank certification by SVD of the QR factor R. ' ...
        'Reference solves, assembly, plotting and post-solve diagnostics are excluded. ' ...
        'Timing is measured once per seed, not a repeated timing benchmark. Inverse RHS columns ' ...
        'and forward operator columns are distinct work units and are not added together.\n\n' ...
        'Residual histories are MATLAB-reported histories normalized by their initial values; ' ...
        'they are not physical residual histories. Final physical residual and solution error ' ...
        'are recorded separately. A zero MINRES flag does not certify physical tolerance.\n\n' ...
        'Thin curves show individual seeds, thick curves the median, and bands the range. ' ...
        'The constant-operator control is not expected to make the two arms coincide: q differs.\n\n']);
    for j=1:numel(cfg.cases)
        c=cfg.cases{j};
        fprintf(fid,'- [%s comparison](plots/%s_comparison.png), [residual histories](plots/%s_residuals.png)\n',c,c,c);
    end
end
