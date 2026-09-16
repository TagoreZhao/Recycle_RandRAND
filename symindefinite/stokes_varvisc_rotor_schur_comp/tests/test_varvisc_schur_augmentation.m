function test_varvisc_schur_augmentation()
%TEST_VARVISC_SCHUR_AUGMENTATION Algebra, cache, costs, breakdown and recovery.
    here=fileparts(mfilename('fullpath')); addpath(fileparts(here));
    add_varvisc_schur_paths(); rng(31);
    n=70; k=8; m=20; [Q,~]=qr(randn(n));
    S=Q*diag(logspace(-2,1,n))*Q'; S=(S+S')/2;
    b=randn(n,1); Omega=randn(n,k);
    opts=struct('tol',1e-11,'maxit',4000,'construction_s',.125,'m',m);
    before=rng; [a,state]=varvisc_schur_augmented_pair(synthetic(S,b),Omega,1,[],opts);
    assert(isequal(before,rng),'Augmentation changed the global random stream.');
    assert(isequal(a{1}.V,a{2}.V) && isequal(state.V,a{1}.V));
    expected=orth(S\(S\Omega));
    assert(norm(state.V-expected*(expected'*state.V),'fro')<1e-9);
    assert(abs(state.tau-max(eig(S)))<1e-9*norm(S));
    assert(a{1}.info.basis_setup_s==a{2}.info.basis_setup_s);
    assert(a{1}.info.exact_chol_s==a{2}.info.exact_chol_s);
    for arm=1:2
        z=a{arm}; Z=[z.V,z.W]; E=Z'*S*Z;
        P=eye(n)-Z*Z'+state.tau*Z*(E\Z');
        assert(norm(P-P','fro')/norm(P,'fro')<1e-12);
        [~,flag]=chol((P+P')/2); assert(flag==0);
        assert(norm(z.E-E,'fro')/norm(E,'fro')<1e-12);
        assert(z.info.flag==0 && z.info.schur_relres<1e-10);
        assert(norm(z.y-S\b)/norm(S\b)<1e-9);
        assert(z.info.inverse_rhs_columns==2*k && z.info.exact_chol_builds==1);
        assert(z.info.coarse_operator_columns==size(Z,2));
        assert(z.info.arnoldi_operator_columns==size(z.W,2));
        assert(z.info.forward_operator_columns==z.info.tau_operator_columns ...
            +z.info.coarse_operator_columns+z.info.arnoldi_operator_columns+z.info.pcg_operator_columns);
    end
    V=state.V; W=a{2}.W; Pi=eye(n)-V*V';
    [We,~]=varvisc_build_projected_arnoldi(@(x)Pi*S*Pi*x,[],Pi*b,m);
    assert(norm(W-We*(We'*W),'fro')<1e-9);
    assert(max([a{2}.info.vw_orth,a{2}.info.ww_orth,a{2}.info.arnoldi_relation])<1e-10);
    assert(norm(W'*Pi*S*Pi*W-a{2}.arnoldi.H,'fro')<1e-10);
    % m=0 follows the identical baseline path, including residual history.
    zero_opts=opts; zero_opts.m=0;
    [z,~]=varvisc_schur_augmented_pair(synthetic(S,b),Omega,1,[],zero_opts);
    assert(isequal(z{1}.y,z{2}.y) && isequal(z{1}.resvec,z{2}.resvec));
    assert(isequal(z{1}.y,a{1}.y) && z{2}.info.arnoldi_s==0);
    % Later operators are never materialized; V and tau are unchanged.
    S2=S+diag(linspace(.01,.3,n)); b2=randn(n,1);
    st=synthetic(S2,b2); st.to_dense=@forbidden_dense;
    [c,state2]=varvisc_schur_augmented_pair(st,Omega,2,state,opts);
    assert(isequal(state2.V,state.V) && state2.tau==state.tau && ~isfield(state2,'W'));
    [expected,~]=varvisc_build_projected_arnoldi(@(x)S2*x,V,b2/norm(b2),m);
    assert(norm(c{2}.W-expected,'fro')<1e-12);
    for arm=1:2
        assert(c{arm}.info.exact_chol_builds==0 && c{arm}.info.inverse_rhs_columns==0);
        assert(c{arm}.info.materialization_s==0 && c{arm}.info.basis_setup_s==0);
    end
    % Static operator + RHS reproduces baseline and augmentation.
    st=synthetic(S,b); st.to_dense=@forbidden_dense;
    [same,~]=varvisc_schur_augmented_pair(st,Omega,2,state,opts);
    assert(isequal(same{2}.W,a{2}.W));
    assert(same{1}.info.iterations==a{1}.info.iterations);
    assert(same{2}.info.iterations==a{2}.info.iterations);
    bad=st; bad.keep=[false;true(n,1)];
    assert_error(@()varvisc_schur_augmented_pair(bad,Omega,2,state,opts),'varviscSchur:coordinateChange');
    bad=synthetic(eye(n+1),ones(n+1,1));
    assert_error(@()varvisc_schur_augmented_pair(bad,ones(n+1,k),2,state,opts),'varviscSchur:coordinateChange');
    % Zero RHS and a projected-zero RHS need no random padding.
    st=synthetic(S,zeros(n,1)); st.to_dense=@forbidden_dense;
    [z,~]=varvisc_schur_augmented_pair(st,Omega,2,state,opts);
    assert(z{2}.info.augment_rank==0 && z{2}.info.arnoldi_status==1);
    assert(z{2}.info.flag==0 && z{2}.info.iterations==0 && norm(z{2}.y)==0);
    st=synthetic(S,V(:,1)); st.to_dense=@forbidden_dense;
    [z,~]=varvisc_schur_augmented_pair(st,Omega,2,state,opts);
    assert(z{2}.info.augment_rank==0 && z{2}.info.arnoldi_status==1);
    [Wi,ai]=varvisc_build_projected_arnoldi(@(x)x,V,b,m);
    assert(size(Wi,2)==1 && ai.status==2);
    [Wc,ac]=varvisc_build_projected_arnoldi(@(x)S*x,Q(:,1:n-3),b,m);
    assert(size(Wc,2)<=3 && ac.operator_columns<=3);
    % A failed iteration limit is returned as data, not hidden or retried.
    limited=opts; limited.maxit=1;
    [z,~]=varvisc_schur_augmented_pair(synthetic(S2,b2),Omega,2,state,limited);
    assert(z{1}.info.flag~=0 && z{1}.info.schur_relres>limited.tol);
    % Real reduced Stokes solve and velocity-boundary/KKT recovery.
    p=struct('h0',.16,'dt',.02,'Tstep',61);
    cfg=varvisc_schur_make_cfg('bar_rotating_nu_orbiting',p,[]);
    ctx=varvisc_schur_context_init(cfg,p);
    st=varvisc_schur_step_operator(ctx,p.dt,zeros(ctx.nU,1));
    [z,~]=varvisc_schur_augmented_pair(st,randn(st.nS,16),1,[],opts);
    [K,rhs]=st.materialize_kkt(); exact=K\rhs;
    for arm=1:2
        assert(z{arm}.info.flag==0 && z{arm}.info.schur_relres<1e-8);
        assert(norm(K*z{arm}.x-rhs)/norm(rhs)<1e-8);
        assert(norm(z{arm}.x-exact)/norm(exact)<1e-8);
        assert(norm(z{arm}.x(cfg.veldofs)-cfg.velvals,inf)<1e-12);
    end
    fprintf('PASS: Schur augmentation algebra, SPD, cache, costs, edge cases and KKT recovery.\n');
end

function st=synthetic(S,b)
    st=struct('nS',size(S,1),'keep',true(size(S,1),1), ...
        'to_dense',@()S,'apply',@(X)S*X,'rhs_S',b,'recover',@(y)y);
end

function S=forbidden_dense()
    S=[]; %#ok<NASGU> Output signature matches the operator's dense handle.
    error('test:unexpectedDense','Later timesteps must not materialize Schur matrices.');
end

function assert_error(action,identifier)
    caught=false;
    try
        action();
    catch exception
        caught=strcmp(exception.identifier,identifier);
    end
    assert(caught,'Expected error %s.',identifier);
end
