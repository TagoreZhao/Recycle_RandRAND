function [K, b, C, gvec, nC] = schur_assemble_kkt(ctx, tcur, u_prev)
%SCHUR_ASSEMBLE_KKT  One immersed-rotor Stokes KKT pair (K, b) at time TCUR.
%   [K, b, C, gvec, nC] = SCHUR_ASSEMBLE_KKT(CTX, TCUR, U_PREV)
%
%   Reproduces the per-step assembly of src.stokes.solve_stokes_immersed
%   EXACTLY -- same block layout, same RHS, and both apply_dirichlet_sym calls
%   in the same order.  The Schur complement is then obtained by SLICING this
%   K rather than re-deriving what the boundary conditions do to each block,
%   which keeps the reduction honest and gives K\b as free ground truth.
%
%       K = [ Avel , B'      , C(t)' ]        Avel = M2/dt + nu*A2  (SPD, const)
%           [ B    , -eps*L  , 0     ]        eps  = h0^2/(12*nu)
%           [ C(t) , 0       , 0     ]        C(t) = the ONLY moving block
%
%   Inputs:
%     ctx    - context from schur_context_init (or the partial context built
%              inside it; only the geometry/operator fields are touched here)
%     tcur   - current time
%     u_prev - nU x 1 velocity from the previous step
%
%   Outputs:
%     K, b   - the symmetric indefinite KKT pair after BC elimination
%     C      - nC x nU coupling block BEFORE BC elimination
%     gvec   - nC x 1 coupling target
%     nC     - number of Lagrange-multiplier rows at this step
%
%   See also: schur_context_init, schur_step_operator,
%             src.stokes.solve_stokes_immersed.

    import src.stokes.*

    nU = ctx.nU;
    nP = ctx.nP;

    % --- Moving coupling C(t_n), g(t_n) ---
    mot = ctx.motion_fun(tcur);
    [C, gvec, nC] = assemble_coupling(ctx.TR, ctx.N, mot.X, mot.V);

    % --- Symmetric indefinite KKT ---
    Z = @(a, b) sparse(a, b);
    K = [ ctx.Avel, ctx.Bdiv',                 C'        ; ...
          ctx.Bdiv, -ctx.eps_stab * ctx.Lp,    Z(nP, nC) ; ...
          C       , Z(nC, nP),                 Z(nC, nC) ];

    % --- Right-hand side ---
    rhsU = (ctx.M2 / ctx.dt) * u_prev;
    if ctx.has_force
        rhsU = rhsU + ctx.M2 * ctx.fnod_fun(tcur);
    end
    b = [rhsU; zeros(nP, 1); gvec];

    % --- Velocity Dirichlet, then pressure pin (order matters) ---
    bc = ctx.velbc_fun(tcur);
    [K, b] = apply_dirichlet_sym(K, b, bc.dofs, bc.vals);
    [K, b] = apply_dirichlet_sym(K, b, nU + ctx.pin_node, ctx.pin_val);
end
