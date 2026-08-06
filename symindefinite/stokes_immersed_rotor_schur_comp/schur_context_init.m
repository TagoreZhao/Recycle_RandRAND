function ctx = schur_context_init(cfg, params)
%SCHUR_CONTEXT_INIT  Time-constant pieces of the Schur-complement sequence.
%   CTX = SCHUR_CONTEXT_INIT(CFG, PARAMS)
%
%   The immersed-rotor KKT system is K = [Avel, G'; G, -D] with G = [B; C(t)]
%   and D = blkdiag(eps*L, 0).  Eliminating the SPD velocity block gives
%
%       S(t_n) = D + G(t_n) * Avel^{-1} * G(t_n)'          (SPD)
%
%   NOTE THE SIGN: from  Avel*u + G'y = b1  and  G*u - D*y = b2,
%       u = Avel^{-1}(b1 - G'y)   =>   (D + G Avel^{-1} G') y = G Avel^{-1} b1 - b2
%   so S = D + G Avel^{-1} G' is SPD.  The negated form is negative definite
%   and pcg refuses it.
%
%   Avel (after Dirichlet elimination) and the divergence block B are BOTH
%   time-constant here, so this routine hoists the expensive part of forming S:
%
%       dA   = decomposition(A_bc, 'chol')      built once
%       Y_B  = dA \ GtB                         nU x nP   -- the one real cost
%       S_pp = D_pp + GtB' * Y_B                CONSTANT for the whole sequence
%
%   Per step only nC (20-44) additional backsolves are needed; see
%   schur_step_operator.
%
%   CFG fields used: .mesh .nu .h0 .velbc_fun .motion_fun [.pin_node] [.pin_val]
%                    [.fnod_fun] [.u0]
%   PARAMS fields used: .dt
%
%   Returned CTX carries the geometry/operator fields consumed by
%   schur_assemble_kkt plus the hoisted Schur pieces:
%     .dA .Y_B .GtB .S_pp .A_bc .D_pp .pin_node .pin_val
%
%   See also: schur_assemble_kkt, schur_step_operator.

    import src.stokes.*

    msh = cfg.mesh;
    N   = msh.N;

    ctx.msh = msh;
    ctx.N   = N;
    ctx.nU  = 2 * N;
    ctx.nP  = N;
    ctx.dt  = params.dt;
    ctx.nu  = cfg.nu;
    ctx.h0  = cfg.h0;

    % --- Time-independent fluid blocks ---
    blk = assemble_stokes_blocks(msh);
    Avel = blk.M2 / ctx.dt + ctx.nu * blk.A2;
    ctx.Avel     = (Avel + Avel') / 2;             % SPD velocity block
    ctx.Bdiv     = blk.B;
    ctx.Lp       = blk.L;
    ctx.Dp       = blk.Dp;
    ctx.M2       = blk.M2;
    ctx.eps_stab = cfg.h0^2 / (12 * ctx.nu);       % Brezzi-Pitkaranta

    ctx.TR = triangulation(msh.t, msh.p);

    % --- Pressure pin ---
    if isfield(cfg, 'pin_node') && ~isempty(cfg.pin_node)
        ctx.pin_node = cfg.pin_node;
    else
        [~, ctx.pin_node] = max(msh.p(:, 1));      % outflow corner
    end
    if isfield(cfg, 'pin_val') && ~isempty(cfg.pin_val)
        ctx.pin_val = cfg.pin_val;
    else
        ctx.pin_val = 0;
    end

    ctx.velbc_fun  = cfg.velbc_fun;
    ctx.motion_fun = cfg.motion_fun;
    ctx.has_force  = isfield(cfg, 'fnod_fun') && ~isempty(cfg.fnod_fun);
    if ctx.has_force
        ctx.fnod_fun = cfg.fnod_fun;
    end

    % --- One reference assembly to extract the CONSTANT post-BC slices -------
    % A_bc and the pressure columns of G' do not depend on t (velbc_fun is
    % steady, B and the pin are fixed), so slicing a single assembled K is both
    % correct and cheaper than re-deriving the BC action block by block.
    u0 = zeros(ctx.nU, 1);
    if isfield(cfg, 'u0') && ~isempty(cfg.u0)
        u0 = cfg.u0;
    end
    Kref = schur_assemble_kkt(ctx, ctx.dt, u0);

    nU = ctx.nU;  nP = ctx.nP;
    ctx.A_bc = Kref(1:nU, 1:nU);                       % SPD, identity on veldofs
    ctx.GtB  = Kref(1:nU, nU + (1:nP));                % = B' post-BC (constant)
    negD_pp  = Kref(nU + (1:nP), nU + (1:nP));         % = -D_pp
    ctx.D_pp = -negD_pp;                               % D_pp; D(pin,pin) = -1

    % --- Hoisted Schur pieces ----------------------------------------------
    ctx.dA   = decomposition(ctx.A_bc, 'chol');
    ctx.Y_B  = ctx.dA \ full(ctx.GtB);                 % nU x nP
    S_pp     = ctx.D_pp + ctx.GtB' * ctx.Y_B;
    ctx.S_pp = (S_pp + S_pp') / 2;                     % exact symmetry
end
