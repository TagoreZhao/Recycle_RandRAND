function [K, b, C, gvec, nC, nu_e] = varvisc_schur_assemble_kkt(ctx, tcur, u_prev)
%VARVISC_SCHUR_ASSEMBLE_KKT  Assemble one post-BC variable-viscosity KKT pair.
%   This exact-reference helper mirrors src.stokes.solve_stokes_varvisc. Normal
%   Schur solves use varvisc_schur_assemble_blocks and do not call this routine.

    import src.stokes.*

    N = ctx.N; nU = ctx.nU; nP = ctx.nP;
    ZN = sparse(N, N);
    Z = @(a,b) sparse(a,b);

    nu_e = ctx.nu_fun(ctx.msh.cent(:,1), ctx.msh.cent(:,2), tcur);
    K1nu = assemble_visc_stiffness(ctx.msh, nu_e);
    Avel = ctx.Mdt + [K1nu, ZN; ZN, K1nu];
    if strcmp(ctx.bp_mode, 'scalar')
        Lp_eps = (ctx.h0^2 / (12 * min(nu_e))) * ctx.blk.L;
    else
        eps_e = ctx.h0^2 ./ (12 * nu_e);
        Lp_eps = assemble_visc_stiffness(ctx.msh, eps_e);
    end

    mot = ctx.motion_fun(tcur);
    [C, gvec, nC] = assemble_coupling(ctx.TR, N, mot.X, mot.V);
    K = [Avel, ctx.Bdiv', C'; ...
         ctx.Bdiv, -Lp_eps, Z(nP,nC); ...
         C, Z(nC,nP), Z(nC,nC)];

    rhsU = ctx.Mdt * u_prev;
    if ctx.has_force
        rhsU = rhsU + ctx.blk.M2 * ctx.fnod_fun(tcur);
    end
    b = [rhsU; zeros(nP,1); gvec];

    bc = ctx.velbc_fun(tcur);
    [K, b] = apply_dirichlet_sym(K, b, bc.dofs, bc.vals);
    [K, b] = apply_dirichlet_sym(K, b, nU + ctx.pin_node, ctx.pin_val);
end
