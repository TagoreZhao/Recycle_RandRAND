function S = build_stokes_sequence(opts)
%BUILD_STOKES_SEQUENCE  The immersed-rotor KKT time sequence in low-rank form.
%
%   S = BUILD_STOKES_SEQUENCE(opts)
%
%   Reproduces the matrix/RHS sequence that
%   +src/+stokes/solve_stokes_immersed.m generates (same assembly, same
%   Dirichlet/pin elimination, same backslash state advance), but ALSO returns
%   the algebraic decomposition the whole study rests on:
%
%       K_n = K0 + Cblk_n * Sel' + Sel * Cblk_n'
%
%   K0 (fluid blocks, zero coupling) and Sel = [0;0;I_nC] are time-independent,
%   so consecutive systems differ by a symmetric perturbation of rank <= 2*nC
%   (nC = 2 x #in-domain Lagrange points; 20 for bar_rotating, 44 for the disks).
%
%   opts (all optional):
%     .case_name    'bar_rotating' (default) | 'disk_translating' | 'disk_static'
%     .h0           mesh size (default 0.05 = the benchmark; 0.1 = the coarse twin)
%     .dt           time step (default 0.02)
%     .Tstep        benchmark Tstep, sets Tmax = dt*Tstep for the motion (default 61)
%     .nsteps       # steps to build (default Tstep-1)
%     .verify       assert the low-rank identity per step (default true)
%     .use_cache    load/save <kernel>/cache/seq_*.mat (default true)
%     .quiet        suppress progress printing (default false)
%
%   Output struct S:
%     .K0 .Sel .Cblk{n}       the low-rank form (use seq_K(S,n) to materialize)
%     .b{n} .xref{n}          RHS and the backslash ground truth
%     .Ccpl{n} .g{n}          raw coupling block and its RHS at step n
%     .coupling_change(n)     ||C_n - C_{n-1}||_F / ||C_{n-1}||_F
%     .n .nU .nP .nC .N       dimensions
%     .h0 .dt .nu .nsteps .case_name .eps_stab
%     .msh .veldofs .velvals .pin_node
%
%   The benchmark advances the state with the DIRECT solution (not the iterative
%   one), so the RHS sequence here is identical to the benchmark's.
%
%   See also: seq_K, seq_dCblk, src.stokes.solve_stokes_immersed,
%             symindefinite/linear_solves/extract_system.

    import src.discretization.*
    import src.stokes.*

    if nargin < 1 || isempty(opts), opts = struct(); end
    case_name = getdef(opts, 'case_name', 'bar_rotating');
    h0        = getdef(opts, 'h0',        0.05);
    dt        = getdef(opts, 'dt',        0.02);
    Tstep     = getdef(opts, 'Tstep',     61);
    nsteps    = getdef(opts, 'nsteps',    Tstep - 1);
    verify    = getdef(opts, 'verify',    true);
    use_cache = getdef(opts, 'use_cache', true);
    quiet     = getdef(opts, 'quiet',     false);

    % ---- cache lookup ----------------------------------------------------
    p = add_recycle_paths();
    tag = sprintf('seq_%s_h%s_dt%s_n%d', case_name, num2str(h0), num2str(dt), nsteps);
    tag = regexprep(tag, '[^\w]', '_');
    cacheFile = fullfile(p.cacheDir, [tag '.mat']);
    if use_cache && exist(cacheFile, 'file') == 2
        L = load(cacheFile, 'S');
        S = L.S;
        if ~quiet
            fprintf('[seq] loaded cached %s (n=%d, nC=%d, %d steps)\n', ...
                    tag, S.n, S.nC, S.nsteps);
        end
        return;
    end

    % ---- geometry / mesh / time-independent fluid blocks ------------------
    x1 = 0; x2 = 4; y1 = 0; y2 = 1;  Lyc = y2 - y1;  Uin = 1.0;
    msh = build_channel_mesh_pde(h0, x1, x2, y1, y2, {'rect_right'});
    N   = msh.N;  nU = 2*N;  nP = N;

    geo = struct('x1', x1, 'x2', x2, 'y1', y1, 'y2', y2, ...
                 'xc', (x1+x2)/2, 'yc', (y1+y2)/2, 'h0', h0, 'Tmax', dt * Tstep);
    cases = define_motion_list(dt);
    idx   = find(cellfun(@(c) strcmp(c.name, case_name), cases), 1);
    assert(~isempty(idx), 'unknown case_name "%s"', case_name);
    mcase = cases{idx}.factory(geo);
    nu    = mcase.nu;

    blk      = assemble_stokes_blocks(msh);
    Avel     = blk.M2/dt + nu*blk.A2;   Avel = (Avel + Avel')/2;
    eps_stab = h0^2 / (12*nu);
    TR       = triangulation(msh.t, msh.p);

    % ---- boundary conditions (time-independent, as in run_benchmark.m) ----
    left   = find(msh.rect_left);
    walls  = unique([find(msh.rect_top); find(msh.rect_bottom)]);
    bnodes = unique([left; walls]);
    yv     = msh.p(bnodes, 2);
    uxv    = zeros(numel(bnodes), 1);
    isleft = ismember(bnodes, left);
    uxv(isleft) = Uin * 4 .* yv(isleft) .* (Lyc - yv(isleft)) / Lyc^2;
    veldofs = [bnodes; N + bnodes];
    velvals = [uxv; zeros(numel(bnodes), 1)];
    [~, pin_node] = max(msh.p(:, 1));

    % ---- nC from the first step (asserted constant below) -----------------
    mot0 = mcase.motion_fun(dt);
    [~, ~, nC] = assemble_coupling(TR, N, mot0.X, mot0.V);
    assert(nC > 0, 'case "%s" produced an empty coupling block', case_name);
    ntot = nU + nP + nC;

    % ---- K0: the KKT with a ZERO coupling block ---------------------------
    Z   = @(a, b) sparse(a, b);
    K0  = [ Avel  ,  blk.B'          ,  Z(nU, nC) ; ...
            blk.B , -eps_stab*blk.L  ,  Z(nP, nC) ; ...
            Z(nC, nU), Z(nC, nP)     ,  Z(nC, nC) ];
    b0  = zeros(ntot, 1);
    [K0, ~] = apply_dirichlet_sym(K0, b0, veldofs, velvals);
    [K0, ~] = apply_dirichlet_sym(K0, zeros(ntot, 1), nU + pin_node, 0);
    Sel = [sparse(nU + nP, nC); speye(nC)];

    % ---- time loop --------------------------------------------------------
    S = struct();
    S.K0 = K0;  S.Sel = Sel;
    S.Cblk = cell(nsteps, 1);  S.b = cell(nsteps, 1);  S.xref = cell(nsteps, 1);
    S.Ccpl = cell(nsteps, 1);  S.g = cell(nsteps, 1);  S.Xpts = cell(nsteps, 1);
    S.coupling_change = nan(nsteps, 1);
    S.nsteps = nsteps;  S.n = ntot;  S.nU = nU;  S.nP = nP;  S.nC = nC;  S.N = N;
    S.h0 = h0;  S.dt = dt;  S.nu = nu;  S.case_name = case_name;
    S.Tmax = dt * Tstep;
    S.eps_stab = eps_stab;  S.msh = msh;  S.veldofs = veldofs;
    S.velvals = velvals;  S.pin_node = pin_node;

    u_prev = zeros(nU, 1);
    C_prev = [];
    for n = 1:nsteps
        tcur = n * dt;
        mot  = mcase.motion_fun(tcur);
        [C, gvec, nCn] = assemble_coupling(TR, N, mot.X, mot.V);
        assert(nCn == nC, ['nC changed from %d to %d at step %d: the fixed ' ...
               'selector Sel (and the whole low-rank form) assumes a constant ' ...
               'multiplier count.'], nC, nCn, n);

        K = [ Avel  ,  blk.B'         ,  C'        ; ...
              blk.B , -eps_stab*blk.L ,  Z(nP, nC) ; ...
              C     ,  Z(nC, nP)      ,  Z(nC, nC) ];
        b = [ (blk.M2/dt) * u_prev ; zeros(nP, 1) ; gvec ];
        [K, b] = apply_dirichlet_sym(K, b, veldofs, velvals);
        [K, b] = apply_dirichlet_sym(K, b, nU + pin_node, 0);

        % low-rank generator: C' padded into the velocity rows, Dirichlet-masked
        Cu = C';
        Cu(veldofs, :) = 0;
        S.Cblk{n} = [Cu; sparse(nP + nC, nC)];

        if verify
            Kl  = seq_K(S, n);
            rel = norm(K - Kl, 'fro') / max(norm(K, 'fro'), eps);
            assert(rel < 1e-12, ['low-rank identity failed at step %d ' ...
                   '(rel err %.3e): K_n = K0 + Cblk*Sel'' + Sel*Cblk'''], n, rel);
        end

        xref = K \ b;
        S.b{n} = b;  S.xref{n} = xref;  S.Ccpl{n} = C;  S.g{n} = gvec;
        S.Xpts{n} = mot.X;              % Lagrange-point positions (mode localization)
        if ~isempty(C_prev) && nnz(C_prev) > 0
            S.coupling_change(n) = norm(C - C_prev, 'fro') / norm(C_prev, 'fro');
        end
        C_prev = C;
        u_prev = xref(1:nU);

        if ~quiet && (mod(n, max(1, round(nsteps/5))) == 0 || n == nsteps)
            fprintf('  [seq %s] step %3d/%d  nC=%d  ||b||=%.2e  dC=%.3f\n', ...
                    case_name, n, nsteps, nC, norm(b), S.coupling_change(n));
        end
    end

    if use_cache
        save(cacheFile, 'S', '-v7.3');
        if ~quiet, fprintf('[seq] cached %s\n', cacheFile); end
    end
end

%==========================================================================
function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
