function st = schur_step_operator(ctx, tcur, u_prev)
%SCHUR_STEP_OPERATOR  Explicit Schur complement S(t_n) and its RHS.
%   ST = SCHUR_STEP_OPERATOR(CTX, TCUR, U_PREV)
%
%   Assembles the KKT pair at TCUR, slices it, and forms the (p,lambda) Schur
%   complement EXPLICITLY.  Only nC (20-44) backsolves are needed per step
%   because the (p,p) block was hoisted into CTX by schur_context_init:
%
%       S = [ S_pp (constant) , GtB' * Y_C        ]     Y_C = A_bc^{-1} * GtC
%           [ (GtB' * Y_C)'   , D_cc + GtC' * Y_C ]
%
%   THE PRESSURE PIN.  src.stokes.apply_dirichlet_sym sets K(dofs,dofs) = I, so
%   after the pin K(nU+pin, nU+pin) = +1, i.e. D(pin,pin) = -1.  Since
%   G(pin,:) = 0 the pin sits FULLY DECOUPLED with S(pin,pin) = -1: the raw S is
%   indefinite by exactly one eigenvalue and chol would fail.  Its index is
%   therefore dropped (ST.keep).  Nothing is lost -- S(keep,pin) = 0 exactly and
%   the pin row reduces to y(pin) = pin_val, which ST.recover puts back.
%
%   Returned ST fields:
%     .S        nS x nS DENSE SPD Schur complement (pin dropped)
%     .rhs_S    nS x 1  right-hand side  (G*A^{-1}*b1 - b2), pin dropped
%     .recover  @(y_keep) -> full KKT solution x = [u; y]
%     .K .b .C .gvec .nC .nS .keep .pin_node
%
%   See also: schur_context_init, schur_assemble_kkt.

    nU = ctx.nU;
    nP = ctx.nP;

    [K, b, C, gvec, nC] = schur_assemble_kkt(ctx, tcur, u_prev);

    nS_full = nP + nC;
    b1 = b(1:nU);
    b2 = b(nU+1:end);

    GtC = K(1:nU, nU + nP + (1:nC));            % = C' post-BC (moving block)
    Gt  = [ctx.GtB, GtC];                       % nU x nS_full
    negD = K(nU+1:end, nU+1:end);
    D    = -negD;

    % --- Explicit S: reuse the constant (p,p) block, form only the border ----
    Y_C  = ctx.dA \ full(GtC);                  % nU x nC  -- the per-step cost
    S_pc = ctx.GtB' * Y_C;                      % nP x nC
    S_cc = full(D(nP+1:end, nP+1:end)) + GtC' * Y_C;

    S_full = [ ctx.S_pp , S_pc ; ...
               S_pc'    , (S_cc + S_cc') / 2 ];
    % D is block diagonal in exact arithmetic; fold in any cross term for safety.
    D_pc = full(D(1:nP, nP+1:end));
    if any(D_pc(:))
        S_full(1:nP, nP+1:end) = S_full(1:nP, nP+1:end) + D_pc;
        S_full(nP+1:end, 1:nP) = S_full(nP+1:end, 1:nP) + D_pc';
    end
    S_full = (S_full + S_full') / 2;

    rhs_full = Gt' * (ctx.dA \ b1) - b2;

    % --- Drop the decoupled pin index ---------------------------------------
    keep = true(nS_full, 1);
    keep(ctx.pin_node) = false;

    st.S     = S_full(keep, keep);
    st.rhs_S = rhs_full(keep);
    st.keep  = keep;

    % --- Recovery back to the full KKT solution -----------------------------
    pin_node = ctx.pin_node;
    pin_val  = ctx.pin_val;
    dA       = ctx.dA;
    st.recover = @(y_keep) recover_full(y_keep, keep, pin_node, pin_val, ...
                                        nS_full, b1, Gt, dA);

    st.K        = K;
    st.b        = b;
    st.C        = C;
    st.gvec     = gvec;
    st.nC       = nC;
    st.nS       = nS_full - 1;
    st.pin_node = pin_node;
end

%==========================================================================
function x = recover_full(y_keep, keep, pin_node, pin_val, nS_full, b1, Gt, dA)
%RECOVER_FULL  Scatter the reduced multiplier back and recover the velocity.
    y = zeros(nS_full, 1);
    y(keep)     = y_keep;
    y(pin_node) = pin_val;
    u = dA \ (b1 - Gt * y);
    x = [u; y];
end
