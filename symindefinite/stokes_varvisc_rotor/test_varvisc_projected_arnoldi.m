function test_varvisc_projected_arnoldi()
%TEST_VARVISC_PROJECTED_ARNOLDI Algebra, breakdown, solver and cache contracts.
    here = fileparts(mfilename('fullpath'));
    addpath(here,fileparts(fileparts(here)));
    rng(7);
    n = 90; k = 9; m = 20;
    [Q,~] = qr(randn(n));
    A = Q*diag([-logspace(-1,1,n/2),logspace(-1,1,n/2)])*Q';
    V = orth(randn(n,k)); b = randn(n,1);
    Pi = eye(n)-V*V';
    [W,arn] = varvisc_build_projected_arnoldi(@(x) A*x,V,b,m);
    [health,ritz] = varvisc_arnoldi_metrics(V,W,arn);
    assert(size(W,2)==m && arn.operator_columns==m);
    assert(max([health.vw_orth,health.ww_orth,health.arnoldi_relation])<1e-10);
    assert(norm(W'*Pi*A*Pi*W-arn.H,'fro')<1e-10);
    [We,~] = varvisc_build_projected_arnoldi(@(x) Pi*A*Pi*x,[],Pi*b,m);
    assert(norm(W-We*(We'*W),'fro')<1e-10, 'Explicit projection disagrees.');
    assert(numel(ritz.values)==m && all(isfinite(ritz.relative_residual)));
    [W0,a0] = varvisc_build_projected_arnoldi(@(x) A*x,V,b,0);
    assert(isempty(W0) && a0.operator_columns==0);
    [Wz,az] = varvisc_build_projected_arnoldi(@(x) A*x,V,V(:,1),m);
    assert(isempty(Wz) && az.status==1);
    [Wi,ai] = varvisc_build_projected_arnoldi(@(x) x,V,b,m);
    assert(size(Wi,2)==1 && ai.status==2);
    [Wc,ac] = varvisc_build_projected_arnoldi(@(x) A*x,Q(:,1:n-3),b,m);
    assert(size(Wc,2)<=3 && ac.operator_columns<=3);
    % A complete small complement gives an independent eigenvalue oracle,
    % including a genuine zero in the complement (not a forced V zero).
    diagonal = [-3;0;1;2;4]; vd = [1;0;0;0;0];
    [wd,ad] = varvisc_build_projected_arnoldi(@(x)diagonal.*x,vd,ones(5,1),4);
    [~,rd] = varvisc_arnoldi_metrics(vd,wd,ad);
    assert(norm(rd.values-diagonal(2:end))<1e-10 && all(rd.converged));

    P = struct('applyCinv',@(x)x,'applyCtinv',@(x)x);
    shared = struct('factor_setup_s',0,'base_setup_s',0,'basis_built_step',1);
    [x0,f0,r0,i0] = src.precond.two_level_split_solve(A,b,1e-11,500,P,V,.5);
    [xz,fz,rz,iz,izinfo] = varvisc_gaussian_augmented_solve(A,b,1e-11,500,P,V,0,.5,shared);
    assert(f0==fz && i0==iz && r0==rz && norm(x0-xz)==0);
    [xa,fa,~,~,ia] = varvisc_gaussian_augmented_solve(A,b,1e-11,500,P,V,m,.5,shared);
    assert(fa==0 && norm(A*xa-b)/norm(b)<1e-8);
    assert(norm(xa-(A\b))/norm(A\b)<1e-8);
    assert(ia.total_rank==k+m && izinfo.total_rank==k);

    params = struct('AUGMENT_ENABLED',true,'AUGMENT_DIAGNOSTICS',false, ...
        'AUGMENT_M',m,'DEFLAT_SM_EIG',6,'SKETCH_OVERSAMPLE',2);
    solvers = varvisc_define_solver_list(params);
    keys = cellfun(@(s)s.key,solvers,'UniformOutput',false);
    base = solvers{strcmp(keys,'two_level_gaussian')};
    aug = solvers{strcmp(keys,'two_level_aug_gaussian')};
    K = spdiags([-ones(n,1)*.2,linspace(-3,3,n)',ones(n,1)*.1],[-1 0 1],n,n);
    K = (K+K')/2;
    pc = struct('step',1,'cache',containers.Map('KeyType','char','ValueType','any'));
    [~,~,~,~,ib] = base.solve(K,b,1e-9,500,pc);
    rng_before = rng;
    [~,~,~,~,ig] = aug.solve(K,b,1e-9,500,pc);
    assert(isequal(rng,rng_before),'Augmentation changed the random stream.');
    assert(isequal(ib.details.V,ig.details.V),'Paired bases differ.');
    assert(ib.factor_setup_s==ig.factor_setup_s && ib.base_setup_s==ig.base_setup_s);
    e = pc.cache('V_gaussian'); U = e.U;
    pc.step = 2; K2 = K + spdiags(.02*cos((1:n)'),0,n,n);
    [~,~,~,~,ig2] = aug.solve(K2,b,1e-9,500,pc);
    physical = orth(ig2.details.P.applyCtinv(ig2.details.V));
    assert(norm(U-physical*(physical'*U),'fro')/norm(U,'fro')<1e-10);
    assert(ig2.basis_built_step==1 && ig2.total_rank==12+ig2.augment_rank);
    fprintf('PASS: projection, Arnoldi, breakdown, m=0 equivalence, direct solve, shared cache, transport.\n');
end
