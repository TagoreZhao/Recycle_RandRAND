function Astat = solve_stokes_varvisc(cfg, params, save_dir)
%SOLVE_STOKES_VARVISC  Backward-Euler unsteady VARIABLE-VISCOSITY Stokes with
% a moving immersed rigid solid (distributed Lagrange multipliers).  Sibling
% of solve_stokes_immersed: same symmetric indefinite KKT structure, but the
% viscosity nu(x,t) is a moving high-contrast field, so the velocity
% stiffness A2(nu_e(t)) and the nu-weighted Brezzi-Pitkaranta stabilization
% block change at EVERY nonzero, EVERY step.  The per-step update is dense
% within the sparsity pattern and numerically full-rank — no Woodbury /
% bordered-Schur shortcut exists (contrast: solve_stokes_immersed's update is
% the C(t) border only, rank <= 2*nC).
%
%   ASTAT = SOLVE_STOKES_VARVISC(CFG, PARAMS, SAVE_DIR)
%
%   Per step n (t_n = n*dt) the system is
%       [ M2/dt + A2(nu_e) ,  B'      ,  C(t_n)' ] [u  ]   [ M2/dt*u^{n-1} ]
%       [ B                , -Lp_eps  ,   0      ] [p  ] = [ 0             ]
%       [ C(t_n)           ,  0       ,   0      ] [lam]   [ g(t_n)        ]
%   with A2(nu_e) the grad-grad stiffness scaled by element viscosity and
%   Lp_eps the BP stabilization assembled with eps_e = h0^2/(12*nu_e).
%   Per step it is solved by:
%     (1) backslash (ground truth, also advances the state), and
%     (2) one Krylov solve per entry of the registry params.solvers (cell
%         array from the benchmark's solver list): each entry supplies either
%         .build(pc) -> MINRES 5th-arg preconditioner apply (or [] for
%         unpreconditioned) or a self-contained .solve(K,b,tol,mit,pc).
%
%   Preconditioner context pc handed to build/solve closures:
%     constant:  .nU, .nP, .cache (containers.Map keyed by the closures)
%     per step:  .K, .step, .nC,
%                .Au_bc  - CURRENT-step velocity block after homogeneous
%                          Dirichlet elimination, symmetrized (changes every
%                          step, unlike solve_stokes_immersed),
%                .dP     - nu-weighted lumped pressure-mass diagonal
%                          mrow ./ nu_node  (apply: yp = rp ./ dP),
%                .nu_e   - current element viscosities.
%
%   Required cfg fields:
%     cfg.mesh        - mesh struct (assemble_fem_struct)
%     cfg.nu_fun      - @(xc, yc, t) -> M x 1 element viscosities (centroids)
%     cfg.velbc_fun   - @(t) -> struct('dofs', idx in 1..2N, 'vals', values)
%     cfg.motion_fun  - @(t) -> struct('X', K x 2, 'V', K x 2)
%     cfg.h0          - mesh size (BP stabilization eps_e = h0^2/(12*nu_e))
%   Optional cfg fields:
%     cfg.bp_mode     - 'elementwise' (default) | 'scalar'
%                       (scalar fallback: eps = h0^2/(12*min(nu_e)))
%     cfg.fnod_fun    - @(t) -> 2N x 1 nodal body force (default 0)
%     cfg.pin_node    - pressure node pinned (default: node of maximum x)
%     cfg.pin_val     - pinned pressure value (default 0)
%     cfg.u0          - 2N x 1 initial velocity (default 0)
%     cfg.case_name, cfg.geometry - labels
%
%   params: .dt, .Tstep, .SOLVER_TOL, .SOLVER_MAXIT, optional .solvers
%           (registry cell array; defaults to a minimal built-in list).
%
%   Returns Astat with (Tstep-1)x1 per-step arrays, keyed by solver:
%     .solver_keys, .solver_labels  - ordered ids/labels from the registry
%     .solver_its/.solver_flag/.solver_relres/.solver_err.(key)
%     .backslash_relres, .constraint_res
%     .coupling_change   (||C(t_n) - C(t_{n-1})||_F / ||C(t_{n-1})||_F)
%     .diffK             (||F(t_n) - F(t_{n-1})||_F / ||F(t_{n-1})||_F,
%                         F = [Avel B'; B -Lp_eps] — C excluded, nC may vary)
%     .nu_contrast       (max(nu_e)/min(nu_e))
%     .dK_nnz_frac       (nnz(Avel(t_n)-Avel(t_{n-1})) / nnz(Avel))
%     .sys_size, .nC
%   plus scalars .sym_res_first/mid/last, .mean_nnz_per_row.

    import src.stokes.*

    if nargin < 3, save_dir = ''; end

    msh = cfg.mesh;
    dt  = params.dt;
    Tstep = params.Tstep;
    tol   = params.SOLVER_TOL;
    maxit = params.SOLVER_MAXIT;

    if isfield(cfg, 'bp_mode') && ~isempty(cfg.bp_mode)
        bp_mode = cfg.bp_mode;
    else
        bp_mode = 'elementwise';
    end

    N  = msh.N;
    nU = 2 * N;          % velocity DOFs
    nP = N;              % pressure DOFs

    % --- Time-independent fluid blocks (mass, divergence) ---
    blk  = assemble_stokes_blocks(msh);
    Bdiv = blk.B;
    Dp   = blk.Dp;
    Mdt  = blk.M2 / dt;
    mrow = full(sum(Dp, 2));                       % lumped pressure mass diagonal
    node_cnt = accumarray(msh.t(:), 1, [N, 1]);    % for element->node nu average

    % --- Triangulation for point location (built once) ---
    TR = triangulation(msh.t, msh.p);

    % --- Pressure pin ---
    if isfield(cfg, 'pin_node') && ~isempty(cfg.pin_node)
        pin_node = cfg.pin_node;
    else
        [~, pin_node] = max(msh.p(:, 1));
    end
    if isfield(cfg, 'pin_val') && ~isempty(cfg.pin_val)
        pin_val = cfg.pin_val;
    else
        pin_val = 0;
    end

    has_force = isfield(cfg, 'fnod_fun') && ~isempty(cfg.fnod_fun);

    if isfield(cfg, 'u0') && ~isempty(cfg.u0)
        u_prev = cfg.u0;
    else
        u_prev = zeros(nU, 1);
    end

    % --- Solver registry (extensible; one Krylov solve per entry) ------------
    if isfield(params, 'solvers') && ~isempty(params.solvers)
        solvers = params.solvers;
    else
        solvers = default_solver_list();
    end
    nsolv      = numel(solvers);
    solver_keys   = cellfun(@(s) s.key,   solvers, 'UniformOutput', false);
    solver_labels = cellfun(@(s) s.label, solvers, 'UniformOutput', false);

    % Preconditioner context.  Unlike solve_stokes_immersed there are no
    % constant preconditioner pieces: the fluid block moves every step, so
    % Au_bc/dP/nu_e are refreshed per step and the closures decide (via
    % pc.cache and their refresh cadences) what to rebuild.
    pc = struct('nU', nU, 'nP', nP, 'nC', 0, 'K', [], 'step', 0, ...
                'Au_bc', [], 'dP', [], 'nu_e', []);
    pc.cache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    nsteps = Tstep - 1;
    Z = @(a, b) sparse(a, b);

    Astat.solver_keys   = solver_keys(:);
    Astat.solver_labels = solver_labels(:);
    Astat.solver_its    = struct();
    Astat.solver_flag   = struct();
    Astat.solver_relres = struct();
    Astat.solver_err    = struct();
    for s = 1:nsolv
        k = solver_keys{s};
        Astat.solver_its.(k)    = zeros(nsteps, 1);
        Astat.solver_flag.(k)   = zeros(nsteps, 1);
        Astat.solver_relres.(k) = zeros(nsteps, 1);
        Astat.solver_err.(k)    = zeros(nsteps, 1);
    end
    Astat.backslash_relres     = zeros(nsteps, 1);
    Astat.constraint_res       = zeros(nsteps, 1);
    Astat.coupling_change      = nan(nsteps, 1);
    Astat.diffK                = nan(nsteps, 1);
    Astat.nu_contrast          = zeros(nsteps, 1);
    Astat.dK_nnz_frac          = zeros(nsteps, 1);
    Astat.sys_size             = zeros(nsteps, 1);
    Astat.nC                   = zeros(nsteps, 1);

    C_prev = [];
    F_prev = [];
    Avel_prev = [];
    sym_log = struct('first', NaN, 'mid', NaN, 'last', NaN);
    mid_step = max(1, round(nsteps / 2));

    for n = 1:nsteps
        tcur = n * dt;

        % --- Element viscosities and the nu-scaled fluid blocks ---
        nu_e  = cfg.nu_fun(msh.cent(:, 1), msh.cent(:, 2), tcur);
        K1nu  = assemble_visc_stiffness(msh, nu_e);
        ZN    = sparse(N, N);
        Avel  = Mdt + [K1nu, ZN; ZN, K1nu];
        if strcmp(bp_mode, 'scalar')
            Lp_eps = (cfg.h0^2 / (12 * min(nu_e))) * blk.L;
        else
            eps_e  = cfg.h0^2 ./ (12 * nu_e);
            Lp_eps = assemble_visc_stiffness(msh, eps_e);
        end

        % --- Moving coupling C(t_n), g(t_n) ---
        mot = cfg.motion_fun(tcur);
        [C, gvec, nC] = assemble_coupling(TR, N, mot.X, mot.V);

        % --- Assemble symmetric indefinite KKT ---
        K = [ Avel ,  Bdiv'  ,  C'       ; ...
              Bdiv , -Lp_eps ,  Z(nP,nC) ; ...
              C    ,  Z(nC,nP), Z(nC,nC) ];

        % --- Right-hand side ---
        rhsU = Mdt * u_prev;
        if has_force
            rhsU = rhsU + blk.M2 * cfg.fnod_fun(tcur);
        end
        b = [rhsU; zeros(nP, 1); gvec];

        % --- Velocity Dirichlet BC + pressure pin (symmetric) ---
        bc = cfg.velbc_fun(tcur);
        [K, b] = apply_dirichlet_sym(K, b, bc.dofs, bc.vals);
        [K, b] = apply_dirichlet_sym(K, b, nU + pin_node, pin_val);

        ntot = size(K, 1);
        Astat.sys_size(n) = ntot;
        Astat.nC(n)       = nC;

        % --- Per-step matrix-change / contrast diagnostics ---
        Astat.nu_contrast(n) = max(nu_e) / min(nu_e);
        F = [Avel, Bdiv'; Bdiv, -Lp_eps];
        if ~isempty(F_prev)
            Astat.diffK(n)       = norm(F - F_prev, 'fro') / norm(F_prev, 'fro');
            Astat.dK_nnz_frac(n) = nnz(Avel - Avel_prev) / nnz(Avel);
        end
        F_prev = F;
        Avel_prev = Avel;
        if n == 1
            Astat.mean_nnz_per_row = nnz(Avel) / nU;
        end

        % --- Symmetry diagnostic at first/mid/last ---
        if n == 1 || n == mid_step || n == nsteps
            sr = norm(K - K', 'fro') / max(norm(K, 'fro'), eps);
            if n == 1,        sym_log.first = sr; end
            if n == mid_step, sym_log.mid   = sr; end
            if n == nsteps,   sym_log.last  = sr; end
        end

        % --- Current-step preconditioner context ---
        Au_bc = Avel;
        Au_bc(bc.dofs, :) = 0;
        Au_bc(:, bc.dofs) = 0;
        Au_bc(bc.dofs, bc.dofs) = speye(numel(bc.dofs));
        pc.Au_bc = (Au_bc + Au_bc') / 2;
        nu_node  = accumarray(msh.t(:), repmat(nu_e, 3, 1), [N, 1]) ./ node_cnt;
        pc.dP    = mrow ./ nu_node;      % apply: yp = rp ./ dP (= nu * M_lumped^{-1} rp)
        pc.nu_e  = nu_e;
        pc.nC    = nC;
        pc.K     = K;
        pc.step  = n;

        % --- (1) Ground truth backslash (advances state) ---
        x_ref = K \ b;
        Astat.backslash_relres(n) = norm(K * x_ref - b) / max(norm(b), eps);

        % --- (2) one Krylov solve per registered solver ---
        % A .solve entry owns its own Krylov method (the build path below is
        % MINRES); it must return a SCALAR iteration count, because it is
        % assigned into one element of the per-step array a few lines down.
        mit = min(maxit, ntot);
        it_last = NaN; rr_last = NaN; err_last = NaN;   % for the progress print
        for s = 1:nsolv
            s_entry = solvers{s};
            if isfield(s_entry, 'solve') && ~isempty(s_entry.solve)
                [x_s, fl_s, rr_s, it_s] = s_entry.solve(K, b, tol, mit, pc);
            else
                Papply = s_entry.build(pc);             % [] -> unpreconditioned
                if isempty(Papply)
                    [x_s, fl_s, rr_s, it_s] = minres(K, b, tol, mit);
                else
                    [x_s, fl_s, rr_s, it_s] = minres(K, b, tol, mit, Papply);
                end
            end
            k = solver_keys{s};
            Astat.solver_flag.(k)(n)   = fl_s;
            Astat.solver_relres.(k)(n) = rr_s;
            Astat.solver_its.(k)(n)    = it_s;
            Astat.solver_err.(k)(n)    = norm(x_s - x_ref) / max(norm(x_ref), eps);
            it_last = it_s; rr_last = rr_s; err_last = Astat.solver_err.(k)(n);
        end

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
            fprintf(['  [%s] step %3d/%d  nC=%4d  its(%s)=%4d  rr=%.1e  err=%.1e' ...
                     '  diffK=%.3f  contrast=%.0f\n'], ...
                getfield_default(cfg,'case_name','stokes_varvisc'), n, nsteps, nC, ...
                solver_keys{end}, it_last, rr_last, err_last, ...
                Astat.diffK(n), Astat.nu_contrast(n));
        end
    end

    Astat.sym_res_first = sym_log.first;
    Astat.sym_res_mid   = sym_log.mid;
    Astat.sym_res_last  = sym_log.last;

    if ~isempty(save_dir)
        if ~exist(save_dir, 'dir'), mkdir(save_dir); end
    end
end

% -------------------------------------------------------------------------
function solvers = default_solver_list()
%DEFAULT_SOLVER_LIST  Fallback registry so the engine runs standalone when the
% caller does not supply params.solvers: unpreconditioned MINRES + the
% refreshed nu-weighted block Jacobi (rebuilt from pc.Au_bc/pc.dP each step).
    solvers = { ...
        struct('key', 'minres_unprec', 'label', 'MINRES (unpreconditioned)', ...
               'build', @(pc) []), ...
        struct('key', 'block_jacobi',  'label', 'MINRES (block Jacobi, refreshed)', ...
               'build', @(pc) make_blockjac(pc)) };
    solvers = solvers(:);
end

function Papply = make_blockjac(pc)
    Lc = ichol_robust(pc.Au_bc);
    Papply = @(r) block_precond(r, pc.nU, pc.nP, pc.nC, Lc, pc.dP);
end

% -------------------------------------------------------------------------
function Lc = ichol_robust(A)
%ICHOL_ROBUST  ichol('nofill') with diagonal-compensation escalation; high
% viscosity contrast can break the plain factorization.
    try
        Lc = ichol(A, struct('type', 'nofill'));
        return;
    catch
    end
    alpha = 1e-3;
    for k = 1:8
        try
            Lc = ichol(A, struct('type', 'nofill', 'diagcomp', alpha));
            return;
        catch
            alpha = alpha * 10;
        end
    end
    Lc = ichol(A, struct('type', 'ict', 'droptol', 1e-3, 'diagcomp', 0.1));
end

% -------------------------------------------------------------------------
function y = block_precond(r, nU, nP, nC, Lc, dP)
%BLOCK_PRECOND  Apply the SPD block-diagonal preconditioner P^{-1} r:
%   velocity ~ ichol(Au_bc), pressure ~ nu-weighted lumped mass, lambda = I.
    ru = r(1:nU);
    rp = r(nU + (1:nP));
    rl = r(nU + nP + (1:nC));

    yu = Lc' \ (Lc \ ru);
    yp = rp ./ dP;
    yl = rl;

    y = [yu; yp; yl];
end

% -------------------------------------------------------------------------
function v = getfield_default(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
