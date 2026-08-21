% TEST_VARVISC_SCHUR_PROJECTOR  Sketch, Lanczos, lift, and deflation algebra.
clear; clc;
thisDir = fileparts(mfilename('fullpath')); addpath(fileparts(thisDir));
add_varvisc_schur_paths(); import src.precond.*; rng(7);
p = make_varvisc_schur_params(); p.h0 = 0.1; k = 4;
cfg = varvisc_schur_make_cfg('bar_rotating_nu_orbiting',p,[]);
ctx = varvisc_schur_context_init(cfg,p);
st = varvisc_schur_step_operator(ctx,p.dt,zeros(ctx.nU,1));
S = st.S; n = size(S,1);
[U,D] = eig(S,'vector'); [lam,ord] = sort(real(D)); U = real(U(:,ord));
VexactSmall = U(:,1:k);
R = chol(S,'lower'); Sinv = @(X) R'\(R\X);

[Vsmall,thetaSmall,smallInfo] = gaussian_rayleigh_ritz_basis( ...
    Sinv,@(X) S*X,n,k,2,2,'smallest');
[Vlarge,thetaLarge,largeInfo] = gaussian_rayleigh_ritz_basis( ...
    @(X) S*X,@(X) S*X,n,k,2,2,'largest');
assert(size(Vsmall,2) == k && size(Vlarge,2) == k, ...
       'Rayleigh--Ritz did not compress both sketches to target rank.');
assert(smallInfo.sketchWidth == 2*k && largeInfo.sketchWidth == 2*k, ...
       'Multiplicative oversampling was not applied consistently.');
assert(norm(Vsmall'*Vsmall-eye(k),'fro') < 1e-10, ...
       'Small Gaussian Ritz basis is not orthonormal.');
assert(norm(Vlarge'*Vlarge-eye(k),'fro') < 1e-10, ...
       'Large Gaussian Ritz basis is not orthonormal.');
assert(issorted(thetaSmall,'ascend') && issorted(thetaLarge,'descend'), ...
       'Ritz values are not ordered by the requested tail.');

options = struct('maxSteps',n,'checkEvery',5, ...
    'tolerance',1e-11,'operatorNorm',lam(end));
[Vlanczos,thetaLanczos,lanczosInfo] = ...
    fully_reorthogonalized_lanczos_smallest(S,k+1,options);
assert(size(Vlanczos,2) == k+1, ...
       'Lanczos did not compute the requested extra Ritz pair.');
assert(lanczosInfo.orthogonalityResidual < 1e-10, ...
       'Fully reorthogonalized Lanczos lost orthogonality.');
assert(max(abs(thetaLanczos-lam(1:k+1))) < 1e-9*lam(end), ...
       'Lanczos smallest Ritz values are inaccurate.');

lambdaHat = min(eig(VexactSmall'*(S*VexactSmall)));
liftTau = lambdaHat/(lam(end)-lambdaHat);
[Plift,PliftHalf] = small_eigenvalue_lift(VexactSmall,liftTau);
lifted = PliftHalf(S*PliftHalf(eye(n)));
expectedLifted = [lam(1:k)*(1+1/liftTau); lam(k+1:end)];
gotLifted = sort(real(eig((lifted+lifted')/2)),'ascend');
assert(max(abs(gotLifted-sort(expectedLifted))) < 1e-8*lam(end), ...
       'Small-mode lift does not produce the predicted spectrum.');
assert(abs((1+1/liftTau)*lambdaHat-lam(end)) < 1e-12*lam(end), ...
       'Dynamic lift tau does not map the smallest captured mode to lambdaMax.');
assert(sum(gotLifted > lam(end)*(1+1e-10)) <= k, ...
       'The lift added more than rank(V) eigenvalues above lambdaMax.');
X = randn(n,3);
assert(norm(Plift(X)-PliftHalf(PliftHalf(X)),'fro') < ...
       1e-11*norm(Plift(X),'fro'),'Lift square-root composition is incorrect.');

V = orth([Vlarge,Vsmall]);
lambdaLower = lam(k+1); lambdaUpper = lam(n-k);
tauStar = sqrt(lambdaLower*lambdaUpper);
[P1half,E1] = deflation_Psqrt_apply(V,S,lambdaUpper,'handle');
S1apply = @(X) P1half(S*P1half(X));
[P2apply,E2] = deflation_P_apply(V,S1apply,tauStar,'handle',0);
Ptwo = @(X) P1half(P2apply(P1half(X)));
[~,f1] = chol((E1+E1')/2); [~,f2] = chol((E2+E2')/2);
assert(f1 == 0 && f2 == 0,'A two-stage coarse matrix is not SPD.');
u = randn(n,1); v = randn(n,1); Mu = Ptwo(u); Mv = Ptwo(v);
symres = abs(u'*Mv-v'*Mu)/max([abs(u'*Mv),abs(v'*Mu),eps]);
assert(symres < 1e-10,'Two-stage apply is not symmetric.');
assert(u'*Mu > 0 && v'*Mv > 0,'Two-stage apply is not positive definite.');

tau = lam(end); P = deflation_P_apply(VexactSmall,S,tau,'matrix',0);
got = sort(real(eig(P*S)));
expected = sort([repmat(tau,k,1);lam(k+1:end)]);
assert(max(abs(got-expected)) < 1e-8*tau, ...
       'Captured exact modes were not relocated to tau.');
fprintf('test_varvisc_schur_projector: ALL ASSERTIONS PASSED\n');
