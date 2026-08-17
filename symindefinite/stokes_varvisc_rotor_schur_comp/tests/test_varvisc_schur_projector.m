% TEST_VARVISC_SCHUR_PROJECTOR  Sketch basis and direct-S deflation projector.
clear; clc;
thisDir = fileparts(mfilename('fullpath')); addpath(fileparts(thisDir));
add_varvisc_schur_paths(); import src.precond.*; rng(7);
p = make_varvisc_schur_params(); p.h0=0.1; k=20;
cfg=varvisc_schur_make_cfg('bar_rotating_nu_orbiting',p,[]);
ctx=varvisc_schur_context_init(cfg,p);
st=varvisc_schur_step_operator(ctx,p.dt,zeros(ctx.nU,1)); S=st.S; n=size(S,1);
[U,D]=eig(S,'vector'); [lam,ord]=sort(real(D)); U=real(U(:,ord)); Ve=U(:,1:k);
R=chol(S,'lower'); Sinv=@(X) R'\(R\X);
Vsmall=varvisc_schur_build_sketch_V(Sinv,n,k,p.q);
Vlarge=varvisc_schur_build_sketch_V(@(X) S*X,n,k,p.q);
V=orth([Vlarge,Vsmall]);
assert(norm(Vsmall'*Vsmall-eye(size(Vsmall,2)),'fro')<1e-10, ...
       'Small-tail sketch basis not orthonormal.');
assert(norm(Vlarge'*Vlarge-eye(size(Vlarge,2)),'fro')<1e-10, ...
       'Large-tail sketch basis not orthonormal.');
assert(norm(V'*V-eye(size(V,2)),'fro')<1e-10, ...
       'Combined sketch basis not orthonormal.');
assert(size(Vsmall,2)==k && size(Vlarge,2)==k && size(V,2)==2*k, ...
       'Gaussian sketches did not retain the requested k+k dimensions.');
nv=size(Vsmall,2);
[Pg,Eg,decEg]=deflation_P_apply(Vsmall,S,lam(end),'matrix',0);
assert(isequal(size(Eg),[nv,nv]),'Coarse matrix dropped post-orth columns.');
assert(size(decEg.Z,2)==nv,'Absorbed basis dropped post-orth columns.');
assert(isequal(size(Pg),[n,n]),'Gaussian deflation matrix has the wrong size.');
[~,f]=chol(Eg); assert(f==0,'Coarse matrix not SPD.');

lambdaLower=lam(k+1); lambdaUpper=lam(n-k);
[P1half,E1]=deflation_Psqrt_apply(V,S,lambdaUpper,'handle');
S1apply=@(X) P1half(S*P1half(X));
[P2apply,E2]=deflation_P_apply( ...
    V,S1apply,sqrt(lambdaLower*lambdaUpper),'handle',0);
Ptwo=@(X) P1half(P2apply(P1half(X)));
[~,f1]=chol((E1+E1')/2); [~,f2]=chol((E2+E2')/2);
assert(f1==0 && f2==0,'A two-stage coarse matrix is not SPD.');
u=randn(n,1); v=randn(n,1); Mu=Ptwo(u); Mv=Ptwo(v);
symres=abs(u'*Mv-v'*Mu)/max([abs(u'*Mv),abs(v'*Mu),eps]);
assert(symres<1e-10,'Two-stage apply is not symmetric.');
assert(u'*Mu>0 && v'*Mv>0,'Two-stage apply is not positive definite.');

tau=lam(end); P=deflation_P_apply(Ve,S,tau,'matrix',0);
assert(norm(P-P','fro')<1e-10*norm(P,'fro'),'P is nonsymmetric.');
[~,f]=chol((P+P')/2); assert(f==0,'P is not SPD.');
got=sort(real(eig(P*S))); expected=sort([repmat(tau,k,1);lam(k+1:end)]);
assert(max(abs(got-expected))<1e-8*tau,'Captured modes were not relocated to tau.');
fprintf('test_varvisc_schur_projector: ALL ASSERTIONS PASSED\n');
