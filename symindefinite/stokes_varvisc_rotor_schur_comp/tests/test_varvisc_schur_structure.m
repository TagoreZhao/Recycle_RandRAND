% TEST_VARVISC_SCHUR_STRUCTURE  Viscosity invalidates the border-only shortcut.
clear; clc;
thisDir = fileparts(mfilename('fullpath')); addpath(fileparts(thisDir));
add_varvisc_schur_paths();
p = make_varvisc_schur_params(); p.h0 = 0.1;

cfg = varvisc_schur_make_cfg('bar_rotating_nu_orbiting',p,[]);
ctx = varvisc_schur_context_init(cfg,p); u = zeros(ctx.nU,1); prev = [];
rank_exceeded = false;
for n = 1:4
    st = varvisc_schur_step_operator(ctx,n*p.dt,u);
    if ~isempty(prev)
        assert(norm(st.A_bc-prev.A_bc,'fro')>1e-8,'A_n did not move.');
        assert(norm(st.D-prev.D,'fro')>1e-8,'D_n did not move.');
        dS = st.S-prev.S; dSpp = dS(1:ctx.nP-1,1:ctx.nP-1);
        assert(norm(dSpp,'fro')>1e-8,'Pressure-pressure Schur block did not move.');
        r = rank(dS,1e-10*norm(st.S,'fro'));
        rank_exceeded = rank_exceeded || r>2*st.nC;
    end
    prev = st; xr=st.K\st.b; u=xr(1:ctx.nU);
end
assert(rank_exceeded,'No dS exceeded the old rank-2nC border bound.');

cfg = varvisc_schur_make_cfg('disk_static_nu_const',p,[]);
ctx = varvisc_schur_context_init(cfg,p); u=zeros(ctx.nU,1);
s1 = varvisc_schur_step_operator(ctx,p.dt,u); xr=s1.K\s1.b; u=xr(1:ctx.nU);
s2 = varvisc_schur_step_operator(ctx,2*p.dt,u);
assert(isequal(s1.A_bc,s2.A_bc),'Static-control A changed.');
assert(isequal(s1.D,s2.D),'Static-control D changed.');
assert(norm(s1.S-s2.S,'fro')<1e-14*norm(s1.S,'fro'),'Static-control S changed.');
fprintf('test_varvisc_schur_structure: ALL ASSERTIONS PASSED\n');
