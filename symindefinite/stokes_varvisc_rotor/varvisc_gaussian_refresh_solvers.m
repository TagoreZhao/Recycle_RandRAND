function solvers = varvisc_gaussian_refresh_solvers(params)
%VARVISC_GAUSSIAN_REFRESH_SOLVERS Two arms sharing only same-step factors/start.
    width = round(params.DEFLAT_SM_EIG*params.SKETCH_OVERSAMPLE);
    opts = struct('width',width,'seed',params.PAIRED_SEED, ...
        'tau',params.DEFLAT_TAU,'mode',params.ILDL_MODE);
    solvers = cell(2,1);
    keys = {'gaussian_q2_recycle','gaussian_q1_refresh'};
    labels = {'Gaussian q=2, recycled','Gaussian q=1, rebuilt each step'};
    for a=1:2
        solvers{a} = struct('key',keys{a},'label',labels{a},'build',[], ...
            'returns_info',true, ...
            'solve',@(K,b,tol,mit,pc) paired_solve(K,b,tol,mit,pc,opts,a), ...
            'diagnose',@(K,b,pc,info,result) diagnose(pc,info,result,keys{a}));
    end
end

function [x,flag,rr,it,info] = paired_solve(K,b,tol,mit,pc,o,arm)
    c = pc.cache; n = size(K,1); key = sprintf('paired_basis_%d',arm);
    assert(o.width<=n,'Sketch width exceeds system dimension.');
    if ~isKey(c,'paired_factor') || c('paired_factor').step~=pc.step
        timer = tic;
        P = src.precond.make_ildl_precond(K,struct('mode',o.mode));
        C = ildl_coordinate_map(P);
        c('paired_factor') = struct('step',pc.step,'P',P,'C',C,'seconds',toc(timer));
    end
    f = c('paired_factor'); P = f.P; C = f.C;
    if ~isKey(c,'paired_omega') || size(c('paired_omega'),1)~=n
        % Independent stream: neither solver order nor mesh RNG affects Omega.
        stream = RandStream('mt19937ar','Seed',o.seed);
        c('paired_omega') = randn(stream,n,o.width);
    end
    forced = isKey(c,key) && size(c(key).U,1)~=n;
    build = arm==2 || ~isKey(c,key) || forced;
    if forced
        warning('varvisc:pairedShapeChanged', ...
            'Step %d: system dimension changed; recycled basis must rebuild.',pc.step);
    end
    info = struct('q',3-arm,'basis_rebuilt',double(build), ...
        'forced_rebuild',double(forced),'ildl_setup_s',f.seconds, ...
        'exact_ldl_s',0,'exact_ldl_builds',0,'basis_setup_s',0, ...
        'transport_s',0,'inverse_applications',0,'inverse_rhs_columns',0);
    if build
        if ~isKey(c,'paired_exact') || c('paired_exact').step~=pc.step
            timer = tic;
            dK = decomposition((K+K')/2,'ldl');
            c('paired_exact') = struct('step',pc.step,'dK',dK,'seconds',toc(timer));
        end
        e = c('paired_exact');
        % Attribute the full factor cost to each algorithm needing it, even
        % when the experiment shares the actual work on its initial step.
        info.exact_ldl_s = e.seconds; info.exact_ldl_builds = 1;
        timer = tic;
        [V,bi] = varvisc_build_gaussian_power_basis(C,c('paired_omega'),3-arm,e.dK);
        U = P.applyCtinv(V);
        info.basis_setup_s = toc(timer);
        info.inverse_applications = bi.inverse_applications;
        info.inverse_rhs_columns = bi.inverse_rhs_columns;
        state = struct('U',U,'step',pc.step,'numerical_rank',bi.numerical_rank);
        c(key) = state; %#ok<NASGU> containers.Map mutates the shared cache handle.
    else
        state = c(key);
        timer = tic;
        [V,ti] = transport_V(state.U,P,C);
        info.transport_s = toc(timer);
        if ti.rank_drop>0
            warning('varvisc:pairedTransportRankLoss','Step %d: transport dropped %d columns.', ...
                pc.step,ti.rank_drop);
        end
    end
    info.basis_built_step = state.step;
    info.basis_numerical_rank = state.numerical_rank;
    info.basis_rank = size(V,2);
    info.rank_drop = o.width-size(V,2);
    [x,flag,rr,it,core] = src.precond.two_level_split_solve(K,b,tol,mit,P,V,o.tau);
    info.coarse_setup_s = core.coarse_setup_s;
    info.minres_s = core.minres_s;
    info.coarse_operator_columns = core.coarse_operator_columns;
    info.minres_operator_columns = core.minres_operator_columns;
    info.algorithm_s = info.ildl_setup_s+info.exact_ldl_s+info.basis_setup_s ...
        +info.transport_s+core.coarse_setup_s+core.minres_s;
    info.details = struct('V',V,'P',P,'core',core);
end

function info = diagnose(pc,info,result,key)
    d = info.details; info = rmfield(info,'details');
    timer = tic;
    V = d.V; AV = d.core.AV;
    info.orthogonality = norm(V'*V-eye(size(V,2)),'fro');
    info.invariance_residual = norm(AV-V*(V'*AV),'fro')/max(norm(AV,'fro'),realmin);
    info.coarse_min_eig = min(d.core.coarse_eigenvalues);
    info.coarse_max_eig = max(d.core.coarse_eigenvalues);
    info.diagnostic_s = toc(timer);
    record = struct('info',info,'resvec',d.core.resvec, ...
        'coarse_eigenvalues',d.core.coarse_eigenvalues,'step',pc.step, ...
        'flag',result.flag,'iters',result.iters,'relres',result.relres, ...
        'true_relres',result.true_relres,'err',result.err);
    if isfield(pc,'output_dir') && ~isempty(pc.output_dir)
        folder = fullfile(pc.output_dir,'diagnostics');
        if ~isfolder(folder), mkdir(folder); end
        save(fullfile(folder,sprintf('%s_step_%03d.mat',key,pc.step)),'record');
    end
end
