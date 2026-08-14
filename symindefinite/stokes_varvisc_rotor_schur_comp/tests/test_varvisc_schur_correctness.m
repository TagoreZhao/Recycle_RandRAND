% TEST_VARVISC_SCHUR_CORRECTNESS  Reduced solve and recovery equal K\b.
clear; clc;
thisDir = fileparts(mfilename('fullpath')); addpath(fileparts(thisDir));
add_varvisc_schur_paths(); assert_varvisc_schur_helpers(); rng(1);
p = make_varvisc_schur_params(); p.h0 = 0.1;
cases = {'bar_rotating_nu_orbiting','disk_translating_nu_wake','disk_static_nu_const'};
msh = []; npass = 0;
for ci = 1:numel(cases)
    [cfg,msh] = varvisc_schur_make_cfg(cases{ci},p,msh);
    ctx = varvisc_schur_context_init(cfg,p); u = zeros(ctx.nU,1);
    for n = 1:2
        st = varvisc_schur_step_operator(ctx,n*p.dt,u);
        xr = st.K\st.b; xs = st.recover(st.S\st.rhs_S);
        relerr = norm(xs-xr)/max(norm(xr),eps);
        assert(relerr<1e-10,'[%s step %d] recovery error %.3e',cases{ci},n,relerr);
        assert(norm(st.S-st.S','fro')<1e-13*norm(st.S,'fro'),'S is nonsymmetric.');
        [~,flag] = chol(st.S); assert(flag==0,'Reduced S is not SPD.');
        assert(norm(xs(cfg.veldofs)-cfg.velvals,inf)<1e-13,'Velocity BCs changed.');
        npass = npass+4; u = xr(1:ctx.nU);
    end
end
fprintf('test_varvisc_schur_correctness: ALL %d ASSERTIONS PASSED\n',npass);
