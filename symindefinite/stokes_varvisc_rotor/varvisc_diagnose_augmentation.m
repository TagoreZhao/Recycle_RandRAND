function info = varvisc_diagnose_augmentation(K,b,pc,info,result,opts)
%VARVISC_DIAGNOSE_AUGMENTATION Post-solve inspection and paired sweep.
% Called by the engine after solver timing. The sweep uses the identical
% live V, P and RHS, but independent Arnoldi builds and coarse setups.
    timer = tic;
    d = info.details; info = rmfield(info,'details');
    V = d.V; W = d.W; k = size(V,2);
    [health,ritz] = varvisc_arnoldi_metrics(V,W,d.arnoldi);
    names = fieldnames(health);
    for j=1:numel(names), info.(names{j})=health.(names{j}); end
    cachekey = 'aug_gaussian_diagnostic';
    if isKey(pc.cache,cachekey), base = pc.cache(cachekey); else, base = []; end
    if isempty(base) || base.step ~= pc.step
        AV = d.core.AV(:,1:k);
        base = struct('step',pc.step,'residual',AV-V*(V'*AV), ...
            'scale',max(norm(AV,'fro'),realmin), ...
            'orth',norm(V'*V-eye(k),'fro'));
        pc.cache(cachekey) = base;
    end
    info.vv_orth = base.orth;
    info.leakage_before = norm(base.residual,'fro') / base.scale;
    info.leakage_after = norm(base.residual-W*(W'*base.residual),'fro') / base.scale;
    info.diagnostic_s = toc(timer);
    record = struct('step',pc.step,'info',info,'ritz',ritz, ...
        'H',d.arnoldi.H,'coupling',d.arnoldi.B, ...
        'resvec',d.core.resvec,'coarse_eigenvalues',d.core.coarse_eigenvalues, ...
        'flag',result.flag,'iters',result.iters,'relres',result.relres, ...
        'true_relres',result.true_relres,'err',result.err,'sweep',[]);

    if opts.m > 0 && strcmp(pc.case_name,'bar_rotating_nu_orbiting')
        sweep_timer = tic;
        shared = struct('factor_setup_s',info.factor_setup_s, ...
            'base_setup_s',info.base_setup_s,'basis_built_step',info.basis_built_step);
        for m = opts.sweep(:)'
            if m == 0 || m == opts.m, continue; end
            [x,flag,rr,it,si] = varvisc_gaussian_augmented_solve( ...
                K,b,d.tol,d.maxit,d.P,V,m,opts.tau,shared);
            [sh,sr] = varvisc_arnoldi_metrics(V,si.details.W,si.details.arnoldi);
            si = rmfield(si,'details');
            sn = fieldnames(sh);
            for j=1:numel(sn), si.(sn{j})=sh.(sn{j}); end
            entry = struct('info',si,'flag',flag,'iters',it,'relres',rr, ...
                'true_relres',norm(K*x-b)/max(norm(b),eps), ...
                'err',norm(x-result.x_ref)/max(norm(result.x_ref),eps),'ritz',sr);
            record.sweep = [record.sweep; entry];
        end
        info.sweep_s = toc(sweep_timer);
    else
        info.sweep_s = 0;
    end
    record.info = info;
    if isempty(pc.output_dir), return; end
    diagdir = fullfile(pc.output_dir,'diagnostics');
    if ~isfolder(diagdir), mkdir(diagdir); end
    save(fullfile(diagdir,sprintf('m%03d_step_%03d.mat',opts.m,pc.step)), ...
        'record','-v7');
    if opts.m > 0 && ismember(pc.step,opts.snapshots)
        snapdir = fullfile(pc.output_dir,'snapshots');
        if ~isfolder(snapdir), mkdir(snapdir); end
        P = d.P; n = size(K,1);
        C = spdiags(1./P.s,0,n,n) * sparse(P.p,1:n,1,n,n) * P.L * P.Dsqrt;
        snapshot = struct('K',K,'b',b,'C',C,'V',V,'W',W, ...
            'arnoldi',d.arnoldi,'step',pc.step,'case_name',pc.case_name, ...
            'ritz',ritz,'true_relres',result.true_relres);
        save(fullfile(snapdir,sprintf('step_%03d.mat',pc.step)),'snapshot','-v7.3');
    end
end
