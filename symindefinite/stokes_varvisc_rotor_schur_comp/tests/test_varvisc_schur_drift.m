% TEST_VARVISC_SCHUR_DRIFT  Frozen exact inverse stales only in moving cases.
clear; clc;
thisDir = fileparts(mfilename('fullpath')); addpath(fileparts(thisDir));
add_varvisc_schur_paths(); rng(1);
p=make_varvisc_schur_params(); p.h0=.1; p.max_steps=4;
p.standalone_variants=[]; p.COMPUTE_SPECTRUM=false;
cfg=varvisc_schur_make_cfg('bar_rotating_nu_orbiting',p,[]);
A=solve_varvisc_schur_sequence(cfg,p,'');
assert(A.solver_its.chol(1)<=2,'Frozen chol is not exact at step 1.');
assert(A.InvRelDiff(1)<1e-10,'Frozen inverse inaccurate on S_1.');
assert(max(A.InvRelDiff(2:end))>1e-3,'Moving viscosity did not stale the inverse.');
assert(max(A.solver_its.chol)>A.solver_its.chol(1),'Frozen chol iterations did not grow.');
assert(median(A.solver_its.chol)<median(A.solver_its.pcg_unprec), ...
       'Frozen exact factor did not beat unpreconditioned PCG.');

cfg=varvisc_schur_make_cfg('disk_static_nu_const',p,[]);
C=solve_varvisc_schur_sequence(cfg,p,'');
assert(max(C.InvRelDiff)<1e-10,'Static control made the frozen inverse stale.');
assert(max(C.solver_its.chol)<=2,'Static-control chol needs more than two iterations.');
fprintf('test_varvisc_schur_drift: ALL ASSERTIONS PASSED\n');
