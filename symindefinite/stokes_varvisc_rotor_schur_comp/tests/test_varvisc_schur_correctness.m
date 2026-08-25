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
        S = st.to_dense();
        probe = randn(st.nS,3);
        assert(norm(st.apply(probe)-S*probe,'fro') < ...
            1e-12*max(norm(S*probe,'fro'),eps),'Block Schur apply is incorrect.');
        [K,b] = st.materialize_kkt();
        [Kassembled,bassembled] = varvisc_schur_assemble_kkt(ctx,n*p.dt,u);
        assert(norm(K-Kassembled,'fro') < 1e-14*max(norm(K,'fro'),eps), ...
            'Lazy KKT matrix differs from monolithic assembly.');
        assert(norm(b-bassembled) < 1e-14*max(norm(b),eps), ...
            'Lazy KKT RHS differs from monolithic assembly.');
        xr = K\b; xs = st.recover(S\st.rhs_S);
        relerr = norm(xs-xr)/max(norm(xr),eps);
        assert(relerr<1e-10,'[%s step %d] recovery error %.3e',cases{ci},n,relerr);
        assert(norm(S-S','fro')<1e-13*norm(S,'fro'),'S is nonsymmetric.');
        [~,flag] = chol(S); assert(flag==0,'Reduced S is not SPD.');
        assert(norm(xs(cfg.veldofs)-cfg.velvals,inf)<1e-13,'Velocity BCs changed.');
        npass = npass+6; u = xr(1:ctx.nU);
    end
end
fprintf('test_varvisc_schur_correctness: ALL %d ASSERTIONS PASSED\n',npass);
