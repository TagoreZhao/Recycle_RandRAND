function ctx = varvisc_schur_context_init(cfg, params)
%VARVISC_SCHUR_CONTEXT_INIT  Time-independent data for Schur construction.
%   Variable viscosity makes A_n and D_n time dependent, so no factorization
%   or pressure-pressure Schur block is hoisted into this context.

    import src.stokes.*

    msh = cfg.mesh;
    ctx.msh = msh;
    ctx.N = msh.N;
    ctx.nU = 2 * msh.N;
    ctx.nP = msh.N;
    ctx.dt = params.dt;
    ctx.h0 = cfg.h0;
    ctx.nu_fun = cfg.nu_fun;
    ctx.blk = assemble_stokes_blocks(msh);
    ctx.Mdt = ctx.blk.M2 / ctx.dt;
    ctx.Bdiv = ctx.blk.B;
    ctx.TR = triangulation(msh.t, msh.p);
    ctx.velbc_fun = cfg.velbc_fun;
    ctx.motion_fun = cfg.motion_fun;

    if isfield(cfg, 'bp_mode') && ~isempty(cfg.bp_mode)
        ctx.bp_mode = cfg.bp_mode;
    else
        ctx.bp_mode = 'elementwise';
    end
    if isfield(cfg, 'pin_node') && ~isempty(cfg.pin_node)
        ctx.pin_node = cfg.pin_node;
    else
        [~, ctx.pin_node] = max(msh.p(:, 1));
    end
    if isfield(cfg, 'pin_val') && ~isempty(cfg.pin_val)
        ctx.pin_val = cfg.pin_val;
    else
        ctx.pin_val = 0;
    end
    ctx.has_force = isfield(cfg, 'fnod_fun') && ~isempty(cfg.fnod_fun);
    if ctx.has_force, ctx.fnod_fun = cfg.fnod_fun; end
end
