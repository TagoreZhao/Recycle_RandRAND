% TEST_VARVISC_SCHUR_HARD_CASE  Complementary viscosity stales frozen chol.
clear; clc;
thisDir = fileparts(mfilename('fullpath')); addpath(fileparts(thisDir));
add_varvisc_schur_paths(); rng(1);

p = make_varvisc_schur_params();
p.h0 = 0.1;
p.Tstep = 7;                 % six solves: phase 0 -> pi
p.max_steps = 6;
p.standalone_variants = [];
p.COMPUTE_SPECTRUM = false;

cfg = varvisc_schur_make_cfg('disk_static_nu_checkerboard_shift',p,[]);
ctx = varvisc_schur_context_init(cfg,p);
s1 = varvisc_schur_step_operator(ctx,p.dt,zeros(ctx.nU,1));
se = varvisc_schur_step_operator(ctx,(p.Tstep-1)*p.dt,zeros(ctx.nU,1));

nu_lo = 0.02; nu_hi = 2.0;
assert(min(s1.nu_e)>=nu_lo*(1-1e-12) && max(s1.nu_e)<=nu_hi*(1+1e-12), ...
       'Initial viscosity left its prescribed bounds.');
assert(min(se.nu_e)>=nu_lo*(1-1e-12) && max(se.nu_e)<=nu_hi*(1+1e-12), ...
       'Final viscosity left its prescribed bounds.');
assert(max(s1.nu_e)/min(s1.nu_e)>90,'Initial contrast is not approximately 100:1.');
assert(max(se.nu_e)/min(se.nu_e)>90,'Final contrast is not approximately 100:1.');
swap_err = norm(s1.nu_e.*se.nu_e-nu_lo*nu_hi,inf)/(nu_lo*nu_hi);
assert(swap_err<1e-11,'Endpoint high/low viscosity regions were not exchanged.');
assert(isequal(s1.C,se.C),'The hard case changed C; drift is not viscosity-only.');

for st = {s1,se}
    op = st{1};
    S = op.to_dense();
    assert(norm(S-S','fro')<1e-13*norm(S,'fro'),'S is nonsymmetric.');
    [~,flag] = chol(S); assert(flag==0,'S is not SPD.');
    xr = op.K\op.b;
    xs = op.recover(S\op.rhs_S);
    assert(norm(xs-xr)/max(norm(xr),eps)<1e-10,'Schur recovery differs from K\b.');
end

S1 = s1.to_dense(); Se = se.to_dense();
R1 = chol(S1,'lower');
H = (R1\Se)/R1'; H = (H+H')/2;
ev = sort(real(eig(H)),'ascend'); nev = numel(ev);
q01 = ev(max(1,ceil(0.01*nev)));
q99 = ev(min(nev,floor(0.99*nev)));
kappa_frozen = ev(end)/ev(1);
assert(kappa_frozen>1e3,'Frozen-Cholesky generalized condition number is too small.');
assert(q01<0.05 && q99>20, ...
       'Generalized spectrum is not broadly spread beyond isolated outliers.');

A = solve_varvisc_schur_sequence(cfg,p,'');
assert(all(A.solver_flag.pcg_unprec==0) && all(A.solver_flag.chol==0), ...
       'A hard-case PCG arm failed to converge.');
assert(max(A.solver_err.pcg_unprec)<1e-5 && max(A.solver_err.chol)<1e-5, ...
       'A hard-case PCG arm lost reference accuracy.');
assert(max(A.vel_recovery_err)<1e-10,'Hard-case recovery drifted from K\b.');
assert(A.solver_its.chol(end)>100,'Frozen Cholesky did not become difficult.');
assert(A.solver_its.chol(end)>=4*A.solver_its.chol(2), ...
       'Frozen-Cholesky iteration growth is too weak.');

fprintf(['test_varvisc_schur_hard_case: ALL ASSERTIONS PASSED ' ...
         '(kappa=%.3g, q01=%.3g, q99=%.3g, chol %d -> %d)\n'], ...
        kappa_frozen,q01,q99,A.solver_its.chol(2),A.solver_its.chol(end));
