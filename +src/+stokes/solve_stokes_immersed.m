function Astat = solve_stokes_immersed(cfg, params, save_dir)
%SOLVE_STOKES_IMMERSED  Backward-Euler unsteady Stokes with a moving immersed
% rigid solid (distributed Lagrange multipliers) — a simplified MATLAB
% step-70.  Solves a SEQUENCE of SYMMETRIC INDEFINITE saddle-point (KKT)
% systems, one per time step, whose coupling block C(t_n) changes because the
% solid moves.
%
%   ASTAT = SOLVE_STOKES_IMMERSED(CFG, PARAMS, SAVE_DIR)
%
%   Per step n (t_n = n*dt) the system is
%       [ M2/dt + nu*A2 ,  B' ,  C(t_n)' ] [u  ]   [ M2/dt*u^{n-1} + M2*f ]
%       [ B             , -eps*L,   0     ] [p  ] = [ 0                     ]
%       [ C(t_n)        ,  0   ,   0      ] [lam]   [ g(t_n)                ]
%   symmetric (transpose-paired off-diagonals) and indefinite (zero (lam,lam)
%   block + negative -eps*L).  It is solved three ways per step:
%     (1) backslash  (ground truth, also advances the state),
%     (2) MINRES, unpreconditioned,
%     (3) MINRES with an SPD block-diagonal preconditioner
%         P = blkdiag(ichol(Avel), Dp/nu, I_lam).
%   The PCG/ICHOL/AMG/deflation zoo used by solve_deflate_M_P does NOT apply
%   here — the matrix is indefinite.
%
%   Required cfg fields:
%     cfg.mesh        - mesh struct (assemble_fem_struct)
%     cfg.nu          - kinematic viscosity
%     cfg.velbc_fun   - @(t) -> struct('dofs', idx in 1..2N, 'vals', values)
%                       velocity Dirichlet data at time t
%     cfg.motion_fun  - @(t) -> struct('X', K x 2 Lagrange points,
%                                       'V', K x 2 rigid-body velocities)
%     cfg.h0          - mesh size (for default stabilization eps = h0^2/(12*nu))
%   Optional cfg fields:
%     cfg.eps_stab    - override Brezzi-Pitkaranta stabilization coefficient
%     cfg.fnod_fun    - @(t) -> 2N x 1 nodal body force (default 0)
%     cfg.pin_node    - pressure node (1..N) pinned to fix the reference
%                       (default: node of maximum x, i.e. outflow)
%     cfg.pin_val     - pinned pressure value (default 0)
%     cfg.u0          - 2N x 1 initial velocity (default 0)
%     cfg.case_name, cfg.geometry - labels
%
%   params: .dt, .Tstep, .SOLVER_TOL, .SOLVER_MAXIT
%
%   Returns Astat with (Tstep-1)x1 per-step arrays:
%     .minres_unprec_its, .minres_blk_its
%     .minres_unprec_flag, .minres_blk_flag
%     .minres_unprec_relres, .minres_blk_relres
%     .backslash_relres      (||K x - b|| / ||b||)
%     .minres_blk_err        (||x_blk - x_ref|| / ||x_ref||)
%     .constraint_res        (||C u - g|| / ||g||, ground-truth solution)
%     .coupling_change       (||C(t_n) - C(t_{n-1})||_F / ||C(t_{n-1})||_F)
%     .sys_size, .nC
%   plus scalar diagnostics .sym_res_first/mid/last, .mean_nnz_per_row.

    import src.stokes.*

    if nargin < 3, save_dir = ''; end

    msh = cfg.mesh;
    nu  = cfg.nu;
    dt  = params.dt;
    Tstep = params.Tstep;
    tol   = params.SOLVER_TOL;
    maxit = params.SOLVER_MAXIT;

    N = msh.N;
    nU = 2 * N;          % velocity DOFs
    nP = N;              % pressure DOFs

    % --- Stabilization coefficient (Brezzi-Pitkaranta) ---
    if isfield(cfg, 'eps_stab') && ~isempty(cfg.eps_stab)
        eps_stab = cfg.eps_stab;
    else
        eps_stab = cfg.h0^2 / (12 * nu);
    end

    % --- Time-independent fluid blocks (assembled once) ---
    blk  = assemble_stokes_blocks(msh);
    Avel = blk.M2 / dt + nu * blk.A2;     % SPD velocity block (constant)
    Avel = (Avel + Avel') / 2;
    Bdiv = blk.B;
    Lp   = blk.L;
    Dp   = blk.Dp;

    % --- Triangulation for point location (built once) ---
    TR = triangulation(msh.t, msh.p);

    % --- Pressure pin (fix the reference; keeps the pressure block nonsingular) ---
    if isfield(cfg, 'pin_node') && ~isempty(cfg.pin_node)
        pin_node = cfg.pin_node;
    else
        [~, pin_node] = max(msh.p(:, 1));   % outflow corner
    end
    if isfield(cfg, 'pin_val') && ~isempty(cfg.pin_val)
        pin_val = cfg.pin_val;
    else
        pin_val = 0;
    end

    % --- Optional body force ---
    has_force = isfield(cfg, 'fnod_fun') && ~isempty(cfg.fnod_fun);

    % --- Initial velocity ---
    if isfield(cfg, 'u0') && ~isempty(cfg.u0)
        u_prev = cfg.u0;
    else
        u_prev = zeros(nU, 1);
    end

    % --- Preconditioner pieces that do not change across steps ---
    % Velocity block after homogeneous Dirichlet elimination is constant, so
    % its ichol factor is built once.
    bc0 = cfg.velbc_fun(0);
    Au_bc = Avel;
    Au_bc(bc0.dofs, :) = 0;
    Au_bc(:, bc0.dofs) = 0;
    Au_bc(bc0.dofs, bc0.dofs) = speye(numel(bc0.dofs));
    Au_bc = (Au_bc + Au_bc') / 2;
    Lc = ichol(Au_bc, struct('type', 'nofill'));      % SPD
    Rp = chol((Dp + Dp') / 2);                         % pressure mass factor

    nsteps = Tstep - 1;
    Z = @(a, b) sparse(a, b);

    Astat.minres_unprec_its    = zeros(nsteps, 1);
    Astat.minres_blk_its       = zeros(nsteps, 1);
    Astat.minres_unprec_flag   = zeros(nsteps, 1);
    Astat.minres_blk_flag      = zeros(nsteps, 1);
    Astat.minres_unprec_relres = zeros(nsteps, 1);
    Astat.minres_blk_relres    = zeros(nsteps, 1);
    Astat.backslash_relres     = zeros(nsteps, 1);
    Astat.minres_blk_err       = zeros(nsteps, 1);
    Astat.constraint_res       = zeros(nsteps, 1);
    Astat.coupling_change      = nan(nsteps, 1);
    Astat.sys_size             = zeros(nsteps, 1);
    Astat.nC                   = zeros(nsteps, 1);

    C_prev = [];
    sym_log = struct('first', NaN, 'mid', NaN, 'last', NaN);
    mid_step = max(1, round(nsteps / 2));

    for n = 1:nsteps
        tcur = n * dt;

        % --- Moving coupling C(t_n), g(t_n) ---
        mot = cfg.motion_fun(tcur);
        [C, gvec, nC] = assemble_coupling(TR, N, mot.X, mot.V);

        % --- Assemble symmetric indefinite KKT ---
        K = [ Avel ,        Bdiv',      C'        ; ...
              Bdiv ,       -eps_stab*Lp, Z(nP,nC) ; ...
              C    ,        Z(nC,nP),    Z(nC,nC) ];

        % --- Right-hand side ---
        rhsU = (blk.M2 / dt) * u_prev;
        if has_force
            rhsU = rhsU + blk.M2 * cfg.fnod_fun(tcur);
        end
        b = [rhsU; zeros(nP, 1); gvec];

        % --- Velocity Dirichlet BC (symmetric) ---
        bc = cfg.velbc_fun(tcur);
        [K, b] = apply_dirichlet_sym(K, b, bc.dofs, bc.vals);

        % --- Pressure pin (symmetric) ---
        [K, b] = apply_dirichlet_sym(K, b, nU + pin_node, pin_val);

        ntot = size(K, 1);
        Astat.sys_size(n) = ntot;
        Astat.nC(n)       = nC;

        % --- Symmetry diagnostic at first/mid/last ---
        if n == 1 || n == mid_step || n == nsteps
            sr = norm(K - K', 'fro') / max(norm(K, 'fro'), eps);
            if n == 1,        sym_log.first = sr; end
            if n == mid_step, sym_log.mid   = sr; end
            if n == nsteps,   sym_log.last  = sr; end
        end

        % --- (1) Ground truth backslash (advances state) ---
        x_ref = K \ b;
        Astat.backslash_relres(n) = norm(K * x_ref - b) / max(norm(b), eps);

        % --- (2) MINRES unpreconditioned ---
        mit = min(maxit, ntot);
        [~, fl_u, rr_u, it_u] = minres(K, b, tol, mit);
        Astat.minres_unprec_flag(n)   = fl_u;
        Astat.minres_unprec_relres(n) = rr_u;
        Astat.minres_unprec_its(n)    = it_u;

        % --- (3) MINRES + SPD block-diagonal preconditioner ---
        Papply = @(r) block_precond(r, nU, nP, nC, Lc, Rp, nu);
        [x_b, fl_b, rr_b, it_b] = minres(K, b, tol, mit, Papply);
        Astat.minres_blk_flag(n)   = fl_b;
        Astat.minres_blk_relres(n) = rr_b;
        Astat.minres_blk_its(n)    = it_b;
        Astat.minres_blk_err(n)    = norm(x_b - x_ref) / max(norm(x_ref), eps);

        % --- Constraint satisfaction of the ground-truth solution ---
        u_ref = x_ref(1:nU);
        if nC > 0
            Astat.constraint_res(n) = norm(C * u_ref - gvec) / max(norm(gvec), eps);
        end

        % --- Per-step coupling change ---
        if ~isempty(C_prev) && nnz(C_prev) > 0 && size(C,1) == size(C_prev,1)
            Astat.coupling_change(n) = norm(C - C_prev, 'fro') / norm(C_prev, 'fro');
        end
        C_prev = C;

        % --- Advance state with the ground-truth velocity ---
        u_prev = u_ref;

        if mod(n, max(1, round(nsteps/5))) == 0 || n == nsteps
            fprintf('  [%s] step %3d/%d  nC=%4d  MINRES(blk)=%4d its  rr=%.1e  err=%.1e\n', ...
                getfield_default(cfg,'case_name','stokes'), n, nsteps, nC, it_b, rr_b, ...
                Astat.minres_blk_err(n));
        end
    end

    Astat.sym_res_first = sym_log.first;
    Astat.sym_res_mid   = sym_log.mid;
    Astat.sym_res_last  = sym_log.last;
    Astat.mean_nnz_per_row = NaN;   % filled by caller if desired

    if ~isempty(save_dir)
        if ~exist(save_dir, 'dir'), mkdir(save_dir); end
    end
end

% -------------------------------------------------------------------------
function y = block_precond(r, nU, nP, nC, Lc, Rp, nu)
%BLOCK_PRECOND  Apply the SPD block-diagonal preconditioner P^{-1} r.
%   Pu  ~ Avel  (applied via ichol factor Lc)
%   Pp  ~ (1/nu) * pressure mass  (applied via chol factor Rp) -> P^{-1} = nu*M^{-1}
%   Plam = I
    ru = r(1:nU);
    rp = r(nU + (1:nP));
    rl = r(nU + nP + (1:nC));

    yu = Lc' \ (Lc \ ru);
    yp = nu * (Rp \ (Rp' \ rp));
    yl = rl;

    y = [yu; yp; yl];
end

% -------------------------------------------------------------------------
function v = getfield_default(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
