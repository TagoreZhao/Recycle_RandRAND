% TEST_VARVISC_SCHUR_PIN  Pin is the sole negative unreduced Schur direction.
clear; clc;
thisDir = fileparts(mfilename('fullpath')); addpath(fileparts(thisDir));
add_varvisc_schur_paths();
p = make_varvisc_schur_params(); p.h0 = 0.1;
cfg = varvisc_schur_make_cfg('bar_rotating_nu_orbiting',p,[]);
ctx = varvisc_schur_context_init(cfg,p);
st = varvisc_schur_step_operator(ctx,p.dt,zeros(ctx.nU,1));
dA = decomposition(st.A_bc,'chol');
Sfull = full(st.D)+st.G*(dA\full(st.Gt)); Sfull=(Sfull+Sfull')/2;
pin = ctx.pin_node; row = Sfull(pin,:); row(pin)=0;
assert(abs(Sfull(pin,pin)+1)<1e-12,'Pinned diagonal is not -1.');
assert(norm(row,inf)<1e-12,'Pinned direction is coupled.');
ev = eig(Sfull); assert(sum(ev<0)==1,'Expected exactly one negative eigenvalue.');
Smanual = Sfull(st.keep,st.keep);
S = st.to_dense();
assert(norm(S-Smanual,'fro')<1e-13*norm(S,'fro'),'Pin deletion is inconsistent.');
fprintf('test_varvisc_schur_pin: ALL ASSERTIONS PASSED\n');
