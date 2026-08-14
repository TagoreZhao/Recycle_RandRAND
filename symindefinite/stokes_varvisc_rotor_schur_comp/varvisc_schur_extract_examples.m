% VARVISC_SCHUR_EXTRACT_EXAMPLES  Save reduced operators at steps 1 and 9.
clearvars; clc;
add_varvisc_schur_paths(); params = make_varvisc_schur_params();
steps = [1 9]; cfg = varvisc_schur_make_cfg('bar_rotating_nu_orbiting',params,[]);
ctx = varvisc_schur_context_init(cfg,params); u = zeros(ctx.nU,1);
for n = 1:max(steps)
    st = varvisc_schur_step_operator(ctx,n*params.dt,u);
    xr = st.K\st.b;
    if ismember(n,steps)
        S = st.S; rhs_S = st.rhs_S; keep = st.keep; %#ok<NASGU>
        y_ref_full = xr(ctx.nU+1:end); y_ref = y_ref_full(keep); %#ok<NASGU>
        meta = struct('case_name',cfg.case_name,'h0',params.h0,'dt',params.dt, ...
            'step',n,'nS',st.nS,'nC',st.nC,'pin_node',ctx.pin_node, ...
            'pin_val',ctx.pin_val,'nu_contrast',max(st.nu_e)/min(st.nu_e)); %#ok<NASGU>
        save(sprintf('varvisc_schur_example_h0p05_step%02d.mat',n), ...
             'S','rhs_S','y_ref','keep','meta','-v7.3');
    end
    u = xr(1:ctx.nU);
end
