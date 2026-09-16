function test_varvisc_schur_gaussian_refresh()
%TEST_VARVISC_SCHUR_GAUSSIAN_REFRESH Powers, current-factor freshness and recovery.
    here=fileparts(mfilename('fullpath')); addpath(fileparts(here));
    add_varvisc_schur_paths(); rng(17);
    n=60; k=8; [U,~]=qr(randn(n));
    S=U*diag(logspace(-2,1,n))*U'; S=(S+S')/2;
    Omega=randn(n,k); R=chol(S,'lower');
    for q=0:2
        [V,info]=varvisc_schur_build_paired_inverse_basis(R,Omega,q);
        expected=orth((S\eye(n))^q*Omega);
        assert(norm(V-expected*(expected'*V),'fro')<1e-9);
        assert(info.inverse_applications==q && info.inverse_rhs_columns==q*k);
        assert(info.basis_rank==k && norm(V'*V-eye(k),'fro')<1e-12);
    end
    opts=struct('tol',1e-11,'maxit',4000,'construction_s',0);
    b=randn(n,1); st=synthetic(S,b); before=rng;
    [a,state]=varvisc_schur_gaussian_refresh_pair(st,Omega,1,[],opts);
    assert(isequal(before,rng),'Paired solver changed the global RNG.');
    assert(a{1}.info.tau==a{2}.info.tau && a{1}.info.tau_eigen_residual<1e-8);
    assert(abs(state.tau-max(eig(S)))<1e-9*norm(S));
    assert(a{1}.info.exact_chol_builds==1 && a{2}.info.exact_chol_builds==1);
    for arm=1:2
        assert(a{arm}.info.flag==0 && norm(a{arm}.y-(S\b))/norm(S\b)<1e-8);
        V=a{arm}.V; E=V'*S*V;
        P=eye(n)-V*V'+state.tau*V*(E\V');
        [~,flag]=chol((P+P')/2); assert(flag==0);
        assert(norm(E-a{arm}.E,'fro')/norm(E,'fro')<1e-12);
    end
    % Repeated static solve reproduces each basis and its iteration count.
    [same,static_state]=varvisc_schur_gaussian_refresh_pair(st,Omega,2,state,opts);
    assert(isequal(state.V,static_state.V) && isequal(a{2}.V,same{2}.V));
    assert(a{1}.info.iterations==same{1}.info.iterations);
    % The changing-system q=1 basis must use S2^-1, with tau still from S1.
    S2=S+diag(linspace(.01,.3,n));
    [c,state2]=varvisc_schur_gaussian_refresh_pair(synthetic(S2,b),Omega,2,state,opts);
    expected=orth(S2\Omega);
    assert(norm(c{2}.V-expected*(expected'*c{2}.V),'fro')<1e-9);
    assert(isequal(c{1}.V,state.V) && state2.tau==state.tau);
    assert(c{1}.info.inverse_applications==0 && c{1}.info.exact_chol_builds==0);
    assert(c{2}.info.inverse_applications==1 && c{2}.info.basis_built_step==2);
    % Reject changed reduced coordinates even if the reduced dimension agrees.
    bad=synthetic(S2,b); bad.keep=[false;true(n,1)]; caught=false;
    try
        varvisc_schur_gaussian_refresh_pair(bad,Omega,2,state,opts);
    catch exception
        caught=strcmp(exception.identifier,'varviscSchur:coordinateChange');
    end
    assert(caught,'Coordinate change was not rejected.');
    % Real reduced Stokes operator, current inverse and full KKT recovery.
    p=struct('h0',.16,'dt',.02,'Tstep',61);
    cfg=varvisc_schur_make_cfg('bar_rotating_nu_orbiting',p,[]);
    ctx=varvisc_schur_context_init(cfg,p);
    st=varvisc_schur_step_operator(ctx,p.dt,zeros(ctx.nU,1));
    Om=randn(st.nS,16);
    [a,~]=varvisc_schur_gaussian_refresh_pair(st,Om,1,[],opts);
    [K,b]=st.materialize_kkt(); exact=K\b;
    for arm=1:2
        assert(a{arm}.info.flag==0);
        assert(a{arm}.info.inverse_probe_residual<1e-9);
        assert(a{arm}.info.schur_apply_probe_residual<1e-12);
        assert(norm(K*a{arm}.x-b)/norm(b)<1e-8);
        assert(norm(a{arm}.x-exact)/norm(exact)<1e-8);
        assert(norm(a{arm}.x(cfg.veldofs)-cfg.velvals,inf)<1e-12);
    end
    fprintf('PASS: Schur powers, current Cholesky, fixed tau, cache, PCG, coordinates and KKT recovery.\n');
end

function st=synthetic(S,b)
    st=struct('nS',size(S,1),'keep',true(size(S,1),1), ...
        'to_dense',@()S,'apply',@(X)S*X,'rhs_S',b,'recover',@(y)y);
end
