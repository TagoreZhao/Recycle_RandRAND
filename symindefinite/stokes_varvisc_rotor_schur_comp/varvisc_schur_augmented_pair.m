function [answers,state] = varvisc_schur_augmented_pair(st,Omega,step,state,opts)
%VARVISC_SCHUR_AUGMENTED_PAIR Frozen q=2 Gaussian basis, with/without Arnoldi.
% Both arms receive full attributed initial setup costs. Only V is cached;
% W is rebuilt from the current Schur operator and RHS and never recycled.
    validateattributes(opts.m,{'numeric'},{'scalar','integer','nonnegative','finite'});
    first=isempty(state); n=st.nS; width=size(Omega,2);
    assert(size(Omega,1)==n && width>0 && width<=n);
    if first
        assert(step==1,'The paired sequence must start at step 1.');
    else
        assert(state.n==n && isequal(state.keep,st.keep), ...
            'varviscSchur:coordinateChange','Reduced coordinates changed; cannot recycle.');
        assert(step==state.last_step+1,'Timesteps must be consecutive.');
        assert(state.width==width,'Sketch width changed during the sequence.');
    end
    shared=struct('q',2,'basis_built_step',1,'basis_rebuilt',double(first), ...
        'step_operator_s',opts.construction_s,'materialization_s',0, ...
        'exact_chol_s',0,'exact_chol_builds',double(first), ...
        'materialize_velocity_rhs_columns',n*double(first),'tau_setup_s',0, ...
        'tau_operator_columns',0,'basis_setup_s',0,'inverse_applications',0, ...
        'inverse_rhs_columns',0);
    initial_diagnostic_s=0; tau_columns=0;
    if first
        timer=tic; S=st.to_dense(); shared.materialization_s=toc(timer);
        timer=tic; [R,flag]=chol(S,'lower'); shared.exact_chol_s=toc(timer);
        assert(flag==0,'varviscSchur:notSPD','Initial Schur Cholesky failed.');
        timer=tic;
        eo=struct('tol',1e-10,'maxit',1000,'v0',ones(n,1)/sqrt(n), ...
            'issym',true,'isreal',true);
        [v,lambda,flag]=eigs(@tau_apply,n,1,'largestreal',eo);
        tau=real(lambda(1));
        tau_res=norm(S*v-tau*v)/max(norm(S*v),realmin);
        shared.tau_setup_s=toc(timer); shared.tau_operator_columns=tau_columns;
        assert(flag==0 && tau>0 && tau_res<=1e-8,'Initial tau eigenpair failed.');
        timer=tic;
        [V,bi]=varvisc_schur_build_paired_inverse_basis(R,Omega,2);
        shared.basis_setup_s=toc(timer);
        shared.inverse_applications=bi.inverse_applications;
        shared.inverse_rhs_columns=bi.inverse_rhs_columns;
        timer=tic; probe=Omega(:,1:min(3,width)); inverse=R'\(R\probe);
        inverse_residual=norm(S*inverse-probe,'fro')/max(norm(probe,'fro'),realmin);
        apply_residual=norm(st.apply(probe)-S*probe,'fro')/max(norm(S*probe,'fro'),realmin);
        state=struct('n',n,'keep',st.keep,'width',width,'V',V,'tau',tau, ...
            'tau_eigen_residual',tau_res,'inverse_probe_residual',inverse_residual, ...
            'schur_apply_probe_residual',apply_residual,'last_step',0);
        initial_diagnostic_s=toc(timer);
        clear S R;
    end
    V=state.V;
    rhs_scale=norm(st.rhs_S); if rhs_scale==0, rhs_scale=1; end
    rhs=st.rhs_S/rhs_scale;
    answers=cell(2,1); op_columns=0;
    for arm=1:2
        info=shared; m=(arm-1)*opts.m;
        info.tau=state.tau; info.tau_eigen_residual=state.tau_eigen_residual;
        info.base_rank=size(V,2); info.rank_drop=width-info.base_rank;
        op_columns=0; timer=tic;
        [W,arn]=varvisc_build_projected_arnoldi(@counted_apply,V,rhs,m);
        info.arnoldi_s=toc(timer); if m==0, info.arnoldi_s=0; end
        info.augment_requested=m; info.augment_rank=size(W,2);
        info.total_rank=info.base_rank+info.augment_rank;
        info.arnoldi_status=arn.status; info.projected_start_norm=arn.start_norm;
        info.arnoldi_operator_columns=op_columns;
        Z=[V,W]; timer=tic;
        [P,E,decE]=src.precond.deflation_P_apply(Z,@counted_apply,state.tau,'handle',0);
        info.coarse_setup_s=toc(timer);
        info.coarse_operator_columns=op_columns-info.arnoldi_operator_columns;
        before_pcg=op_columns; timer=tic;
        [ys,flag,rr,it,rv]=pcg(@counted_apply,rhs,opts.tol,opts.maxit,P,[],zeros(n,1));
        info.pcg_s=toc(timer); info.pcg_operator_columns=op_columns-before_pcg;
        info.forward_operator_columns=op_columns+info.tau_operator_columns;
        y=ys*rhs_scale;
        timer=tic; x=st.recover(y); info.recovery_s=toc(timer);
        info.algorithm_s=info.step_operator_s+info.materialization_s+info.exact_chol_s ...
            +info.tau_setup_s+info.basis_setup_s+info.arnoldi_s+info.coarse_setup_s ...
            +info.pcg_s+info.recovery_s;
        timer=tic;
        health=varvisc_arnoldi_metrics(V,W,arn);
        info.base_orthogonality=norm(V'*V-eye(size(V,2)),'fro');
        info.vw_orth=health.vw_orth; info.ww_orth=health.ww_orth;
        info.arnoldi_relation=health.arnoldi_relation;
        info.coarse_chol_min_diag=min(diag(decE.R));
        info.inverse_probe_residual=state.inverse_probe_residual;
        info.schur_apply_probe_residual=state.schur_apply_probe_residual;
        info.schur_relres=norm(st.apply(y)-st.rhs_S)/max(norm(st.rhs_S),realmin);
        info.diagnostic_s=initial_diagnostic_s+toc(timer);
        info.flag=flag; info.reported_relres=rr; info.iterations=it;
        answers{arm}=struct('info',info,'resvec',rv,'y',y,'x',x, ...
            'V',V,'W',W,'E',E,'arnoldi',arn);
    end
    state.last_step=step;
    function Y=counted_apply(X)
        op_columns=op_columns+size(X,2); Y=st.apply(X);
    end
    function Y=tau_apply(X)
        tau_columns=tau_columns+size(X,2); Y=S*X;
    end
end
