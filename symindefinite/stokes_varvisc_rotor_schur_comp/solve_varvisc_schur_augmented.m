function result = solve_varvisc_schur_augmented(cfg,params,folder)
%SOLVE_VARVISC_SCHUR_AUGMENTED Paired PCG with a direct-KKT state trajectory.
    ctx=varvisc_schur_context_init(cfg,params);
    nsteps=params.Tstep-1;
    if isfield(params,'max_steps'), nsteps=min(nsteps,params.max_steps); end
    width=ceil(params.sm_eig*params.sketch_oversampling);
    keys={'gaussian_q2_recycle','gaussian_q2_augmented'};
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
            'reference_relres',norm(K*xref-b)/max(norm(b),realmin), ...
            'rhs_norm',norm(st.rhs_S),'nu_contrast',max(st.nu_e)/min(st.nu_e), ...
            'A_change',NaN,'D_change',NaN,'coupling_change',NaN);
        if ~isempty(previous)
            ph.A_change=norm(st.A_bc-previous.A,'fro')/max(norm(previous.A,'fro'),realmin);
            ph.D_change=norm(st.D-previous.D,'fro')/max(norm(previous.D,'fro'),realmin);
            ph.coupling_change=norm(st.C-previous.C,'fro')/max(norm(previous.C,'fro'),realmin);
        end
        previous=struct('A',st.A_bc,'D',st.D,'C',st.C);
        opts=struct('tol',params.SOLVER_TOL,'maxit',params.SOLVER_MAXIT, ...
            'm',params.AUGMENT_M,'construction_s',construction_s);
        [answers,state]=varvisc_schur_augmented_pair(st,Omega,step,state,opts);
        for arm=1:2
            a=answers{arm}; r=a.info;
            r.case_name=cfg.case_name; r.seed=params.PAIRED_SEED; r.arm=keys{arm}; r.step=step;
            r.kkt_relres=norm(K*a.x-b)/max(norm(b),realmin);
            r.solution_error=norm(a.x-xref)/max(norm(xref),realmin);
            yr=xref(ctx.nU+find(st.keep));
            r.schur_solution_error=norm(a.y-yr)/max(norm(yr),realmin);
            rows{end+1}=r; %#ok<AGROW>
            record=struct('info',r,'resvec',a.resvec,'arnoldi_H',a.arnoldi.H, ...
                'arnoldi_B',a.arnoldi.B);
            save(fullfile(diagdir,sprintf('%s_step_%03d.mat',keys{arm},step)),'record');
        end
        physics{end+1}=ph; %#ok<AGROW>
        u=xref(1:ctx.nU);
        if step==1 || mod(step,10)==0 || step==nsteps
            fprintf('  step %d/%d nS=%d: base=%d aug=%d, ranks=%d+%d\n', ...
                step,nsteps,st.nS,answers{1}.info.iterations,answers{2}.info.iterations, ...
                answers{2}.info.base_rank,answers{2}.info.augment_rank);
        end
    end
    result=struct('rows',vertcat(rows{:}),'physics',vertcat(physics{:}), ...
        'tau',state.tau,'tau_eigen_residual',state.tau_eigen_residual);
end
