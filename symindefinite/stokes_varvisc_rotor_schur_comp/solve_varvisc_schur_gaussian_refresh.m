function result = solve_varvisc_schur_gaussian_refresh(cfg,params,folder)
%SOLVE_VARVISC_SCHUR_GAUSSIAN_REFRESH Focused sequence, direct KKT state advance.
    ctx=varvisc_schur_context_init(cfg,params);
    nsteps=params.Tstep-1;
    if isfield(params,'max_steps'), nsteps=min(nsteps,params.max_steps); end
    width=ceil(params.sm_eig*params.sketch_oversampling);
    keys={'gaussian_q2_recycle','gaussian_q1_refresh'};
    u=zeros(ctx.nU,1); state=[]; Omega=[]; rows={}; physics={}; previous=[];
    diagdir=fullfile(folder,'diagnostics'); if ~isfolder(diagdir), mkdir(diagdir); end
    for step=1:nsteps
        timer=tic; st=varvisc_schur_step_operator(ctx,step*params.dt,u);
        construction_s=toc(timer);
        if isempty(Omega)
            assert(width<st.nS,'Full-width space would trivialize the comparison.');
            stream=RandStream('mt19937ar','Seed',params.PAIRED_SEED);
            Omega=randn(stream,st.nS,width);
        end
        [K,b]=st.materialize_kkt(); xref=K\b;
        ph=struct('step',step,'nS',st.nS,'nC',st.nC, ...
            'reference_relres',norm(K*xref-b)/max(norm(b),eps), ...
            'rhs_norm',norm(st.rhs_S),'nu_contrast',max(st.nu_e)/min(st.nu_e), ...
            'A_change',NaN,'D_change',NaN,'coupling_change',NaN);
        if ~isempty(previous)
            ph.A_change=norm(st.A_bc-previous.A,'fro')/max(norm(previous.A,'fro'),eps);
            ph.D_change=norm(st.D-previous.D,'fro')/max(norm(previous.D,'fro'),eps);
            ph.coupling_change=norm(st.C-previous.C,'fro')/max(norm(previous.C,'fro'),eps);
        end
        previous=struct('A',st.A_bc,'D',st.D,'C',st.C);
        opts=struct('tol',params.SOLVER_TOL,'maxit',params.SOLVER_MAXIT, ...
            'construction_s',construction_s);
        [answers,state]=varvisc_schur_gaussian_refresh_pair(st,Omega,step,state,opts);
        for arm=1:2
            a=answers{arm}; r=a.info;
            r.case_name=cfg.case_name; r.seed=params.PAIRED_SEED; r.arm=keys{arm}; r.step=step;
            r.kkt_relres=norm(K*a.x-b)/max(norm(b),eps);
            r.solution_error=norm(a.x-xref)/max(norm(xref),eps);
            r.schur_solution_error=norm(a.y-xref(ctx.nU+find(st.keep)))/ ...
                max(norm(xref(ctx.nU+find(st.keep))),eps);
            rows{end+1}=r; %#ok<AGROW>
            record=struct('info',r,'resvec',a.resvec);
            save(fullfile(diagdir,sprintf('%s_step_%03d.mat',keys{arm},step)),'record');
        end
        physics{end+1}=ph; %#ok<AGROW>
        u=xref(1:ctx.nU);
        if step==1 || mod(step,10)==0 || step==nsteps
            fprintf('  step %d/%d nS=%d: q2=%d q1=%d, ranks=%d/%d\n', ...
                step,nsteps,st.nS,answers{1}.info.iterations,answers{2}.info.iterations, ...
                answers{1}.info.basis_rank,answers{2}.info.basis_rank);
        end
    end
    result=struct('rows',vertcat(rows{:}),'physics',vertcat(physics{:}), ...
        'tau',state.tau,'tau_eigen_residual',state.tau_eigen_residual);
end
