% TEST_VARVISC_SCHUR_REGISTRY  End-to-end four-arm smoke test.
clear; clc;
thisDir = fileparts(mfilename('fullpath')); addpath(fileparts(thisDir));
add_varvisc_schur_paths(); rng(1);
p=make_varvisc_schur_params(); p.h0=.1; p.max_steps=3; p.sm_eig=20;
cfg=varvisc_schur_make_cfg('bar_rotating_nu_orbiting',p,[]);
A=solve_varvisc_schur_sequence(cfg,p,'');
expected={'pcg_unprec','chol','deflate_exact','deflate_gaussian'};
assert(isequal(A.solver_keys(:)',expected),'Unexpected registry ordering.');
for i=1:numel(expected)
    key=expected{i};
    assert(all(A.solver_flag.(key)==0),'%s failed to converge.',key);
    assert(max(A.solver_err.(key))<1e-5,'%s solution error too large.',key);
end
assert(isequal(A.chol_built_step,1),'Frozen Cholesky was not built once.');
for key={'deflate_exact','deflate_gaussian'}
    assert(A.basis_built_step.(key{1})==1,'%s basis was not built once.',key{1});
    assert(A.deflat_dim.(key{1})>=1 && A.deflat_dim.(key{1})<=p.sm_eig, ...
           '%s has invalid realized width.',key{1});
end
assert(A.deflat_dim.deflate_exact==p.sm_eig, ...
       'Exact deflation did not retain the requested width.');
assert(A.deflat_dim.deflate_gaussian==p.sm_eig, ...
       'Gaussian orth unexpectedly reduced the smoke-test width.');
assert(abs(A.tau-A.lambda_max(1))<1e-10*A.lambda_max(1), ...
       'tau is not lambda_max(S_1).');
assert(max(A.vel_recovery_err)<1e-10,'Schur recovery drifted from K\b.');
fprintf('test_varvisc_schur_registry: ALL ASSERTIONS PASSED\n');
