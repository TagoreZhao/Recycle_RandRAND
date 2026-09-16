function [results,state] = varvisc_schur_gaussian_refresh_pair(st,Omega,step,state,opts)
%VARVISC_SCHUR_GAUSSIAN_REFRESH_PAIR Current q=1 versus frozen q=2, paired PCG.
% STATE holds only the recycled basis, fixed tau, and reduced-coordinate map.
% Dense S and its current Cholesky are shared work, attributed independently.
    first=isempty(state); n=st.nS; width=size(Omega,2);
    assert(size(Omega,1)==n && width<=n);
    if ~first
        assert(state.n==n && isequal(state.keep,st.keep), ...
            'varviscSchur:coordinateChange','Reduced coordinates changed; cannot recycle.');
        assert(step==state.last_step+1,'Timesteps must be consecutive.');
    else
        assert(step==1,'The paired sequence must start at step 1.');
    end
    timer=tic; S=st.to_dense(); dense_s=toc(timer);
    timer=tic; [R,chol_flag]=chol(S,'lower'); chol_s=toc(timer);
    assert(chol_flag==0,'varviscSchur:notSPD','Current Schur Cholesky failed.');
    tau_s=0;
    if first
        timer=tic;
        eo=struct('tol',1e-10,'maxit',1000,'v0',ones(n,1)/sqrt(n));
        [v,lambda,flag]=eigs(S,1,'largestreal',eo);
        tau=real(lambda(1));
        tau_res=norm(S*v-tau*v)/max(norm(S*v),realmin);
        tau_s=toc(timer);
        assert(flag==0 && tau>0 && tau_res<=1e-8,'Initial tau eigenpair failed.');
        state=struct('n',n,'keep',st.keep,'V',[],'tau',tau, ...
            'tau_eigen_residual',tau_res,'last_step',0);
    end
    % Factor/application checks are diagnostic overhead, outside algorithm time.
    timer=tic; probe=Omega(:,1:min(3,width));
    Z=R'\(R\probe);
    inverse_residual=norm(S*Z-probe,'fro')/max(norm(probe,'fro'),realmin);
    apply_residual=norm(st.apply(probe)-S*probe,'fro')/max(norm(S*probe,'fro'),realmin);
    factor_check_s=toc(timer);
    rhs_scale=norm(st.rhs_S); if rhs_scale==0, rhs_scale=1; end
    rhs=st.rhs_S/rhs_scale;
    results=cell(2,1); op_columns=0;
    for arm=1:2
        rebuild=first || arm==2;
        info=struct('q',3-arm,'basis_rebuilt',double(rebuild), ...
            'basis_built_step',1,'step_operator_s',opts.construction_s, ...
            'materialization_s',0,'exact_chol_s',0,'exact_chol_builds',0, ...
            'materialize_velocity_rhs_columns',0,'tau_setup_s',tau_s, ...
            'basis_setup_s',0,'inverse_applications',0,'inverse_rhs_columns',0, ...
            'tau',state.tau,'tau_eigen_residual',state.tau_eigen_residual);
        if rebuild
            info.materialization_s=dense_s; info.exact_chol_s=chol_s;
            info.exact_chol_builds=1; info.materialize_velocity_rhs_columns=n;
            timer=tic;
            [V,bi]=varvisc_schur_build_paired_inverse_basis(R,Omega,3-arm);
            info.basis_setup_s=toc(timer);
            info.inverse_applications=bi.inverse_applications;
            info.inverse_rhs_columns=bi.inverse_rhs_columns;
            info.basis_built_step=step;
            if arm==1, state.V=V; end
        else
            V=state.V;
        end
        info.basis_rank=size(V,2); info.rank_drop=width-size(V,2);
        op_columns=0; timer=tic;
        [P,E,decE]=src.precond.deflation_P_apply(V,@counted_apply,state.tau,'handle',0);
        info.coarse_setup_s=toc(timer);
        info.coarse_operator_columns=op_columns;
        timer=tic;
        [ys,flag,rr,it,rv]=pcg(@counted_apply,rhs,opts.tol,opts.maxit,P,[],zeros(n,1));
        info.pcg_s=toc(timer);
        info.pcg_operator_columns=op_columns-info.coarse_operator_columns;
        y=ys*rhs_scale;
        timer=tic; x=st.recover(y); info.recovery_s=toc(timer);
        info.algorithm_s=info.step_operator_s+info.materialization_s+info.exact_chol_s ...
            +info.tau_setup_s+info.basis_setup_s+info.coarse_setup_s+info.pcg_s+info.recovery_s;
        timer=tic;
        info.orthogonality=norm(V'*V-eye(size(V,2)),'fro');
        info.coarse_chol_min_diag=min(diag(decE.R));
        info.inverse_probe_residual=inverse_residual;
        info.schur_apply_probe_residual=apply_residual;
        info.schur_relres=norm(st.apply(y)-st.rhs_S)/max(norm(st.rhs_S),eps);
        info.diagnostic_s=factor_check_s+toc(timer);
        info.flag=flag; info.reported_relres=rr; info.iterations=it;
        results{arm}=struct('info',info,'resvec',rv,'y',y,'x',x,'V',V,'E',E);
    end
    state.last_step=step;
    function Y=counted_apply(X)
        op_columns=op_columns+size(X,2); Y=st.apply(X);
    end
end
