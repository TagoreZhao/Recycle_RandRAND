function test_varvisc_gaussian_refresh()
%TEST_VARVISC_GAUSSIAN_REFRESH Independent algebra, cache and transport checks.
    here=fileparts(mfilename('fullpath')); addpath(here,fileparts(fileparts(here)));
    rng(71); n=60; k=8;
    [Q,~]=qr(randn(n));
    K=sparse(Q*diag([-linspace(.2,3,n/2),linspace(.3,4,n/2)])*Q');
    K=(K+K')/2;
    C=diag(linspace(.7,1.3,n)); Om=randn(n,k);
    dec=decomposition(K,'ldl'); B=C'*(K\C);
    for q=1:2
        [V,info]=varvisc_build_gaussian_power_basis(C,Om,q,dec);
        expected=orth(B^q*Om);
        assert(norm(V-expected*(expected'*V),'fro')<1e-10);
        assert(info.inverse_applications==q && info.inverse_rhs_columns==q*k);
        assert(norm(V'*V-eye(k),'fro')<1e-12);
    end
    assert(norm(K*(dec\Om)-Om,'fro')/norm(Om,'fro')<1e-12);
    p=varvisc_default_benchmark_params(); p.SOLVER_PROFILE='gaussian_refresh';
    p.DEFLAT_SM_EIG=k/2; p.PAIRED_SEED=2;
    solvers=varvisc_define_solver_list(p);
    pc=struct('step',1,'cache',containers.Map('KeyType','char','ValueType','any'));
    b=randn(n,1); savedrng=rng;
    [x,flag,~,~,i1]=solvers{1}.solve(K,b,1e-11,n,pc);
    [~,~,~,~,i2]=solvers{2}.solve(K,b,1e-11,n,pc);
    assert(isequal(rng,savedrng),'Solver changed global RNG.');
    assert(flag==0 && norm(K*x-b)/norm(b)<1e-8);
    assert(i1.exact_ldl_s==i2.exact_ldl_s && i1.ildl_setup_s==i2.ildl_setup_s);
    f=pc.cache('paired_factor'); omega=pc.cache('paired_omega');
    % Explicitly match the original generic builder under the same Gaussian RNG.
    rng(p.PAIRED_SEED);
    old=src.precond.build_deflation_V(K,f.P,struct('method','gaussian', ...
        'sm_eig',k/2,'oversample',2,'q',2),dec);
    assert(norm(old-i1.details.V*(i1.details.V'*old),'fro')<1e-9);
    u=pc.cache('paired_basis_1').U;
    pc.step=2; K2=K+spdiags(.05*cos((1:n)'),0,n,n);
    [~,~,~,~,r1]=solvers{1}.solve(K2,b,1e-11,n,pc);
    [~,~,~,~,r2]=solvers{2}.solve(K2,b,1e-11,n,pc);
    physical=orth(r1.details.P.applyCtinv(r1.details.V));
    assert(norm(u-physical*(physical'*u),'fro')/norm(u,'fro')<1e-10);
    assert(r1.basis_built_step==1 && r1.exact_ldl_builds==0 && r1.inverse_applications==0);
    assert(r2.basis_built_step==2 && r2.exact_ldl_builds==1 && r2.inverse_applications==1);
    assert(isequal(pc.cache('paired_omega'),omega));
    f=pc.cache('paired_factor');
    expected=orth(f.C'*(K2\(f.C*omega)));
    assert(norm(r2.details.V-expected*(expected'*r2.details.V),'fro')<1e-9);
    % Reverse order with a clean cache: each arm sees the identical basis.
    pc2=struct('step',1,'cache',containers.Map('KeyType','char','ValueType','any'));
    [~,~,~,~,j2]=solvers{2}.solve(K,b,1e-11,n,pc2);
    [~,~,~,~,j1]=solvers{1}.solve(K,b,1e-11,n,pc2);
    assert(isequal(i1.details.V,j1.details.V) && isequal(i2.details.V,j2.details.V));
    % Fixed K and fixed Omega give the same q=1 subspace on the control.
    pc2.step=2;
    [~,~,~,~,j3]=solvers{2}.solve(K,b,1e-11,n,pc2);
    assert(isequal(j2.details.V,j3.details.V));
    % A changed dimension forces an observable rebuild and new compatible start.
    pc2.step=3; K3=blkdiag(K,2); b3=[b;1];
    [~,~,~,~,j4]=solvers{1}.solve(K3,b3,1e-11,n+1,pc2);
    assert(j4.forced_rebuild==1 && j4.basis_built_step==3);
    assert(size(j4.details.V,1)==n+1 && j4.inverse_applications==2);
    fprintf('PASS: powers, original q=2 equivalence, LDL, paired RNG, refresh, transport, cache order, shape change.\n');
end
