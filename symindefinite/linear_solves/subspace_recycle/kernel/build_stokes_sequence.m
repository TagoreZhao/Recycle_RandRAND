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
%     .use_cache    load/save <kernel>/cache/seq_*.mat (default true).  A cache HIT
%                   is validated by CONTENT, not by filename -- see below.
%     .quiet        suppress progress printing (default false)
%     .motion_params  struct forwarded to define_motion_list: .theta0, .Lb_frac,
%                   .bar_pt_frac (see that file).  Defaults reproduce the
%                   previous hardcoded geometry bit-for-bit.
%
%   THE CACHE IS CONTENT-VALIDATED.  The filename tag records
%   (case_name, h0, dt, nsteps) plus a suffix for any non-default motion_params,
%   so it is still blind to much of what defines the problem: Tstep/Tmax, nu, the
%   channel box, Uin, and the geometry literals that remain unparameterised (the
%   disk radius, its 0.95*h0 spacing, the 0.95*rd clip, the max(8,...) floor).
%   Editing one of those and re-running used to return the OLD sequence under the
%   same name, silently.  Every entry therefore stores S.fingerprint and every hit
%   re-derives and compares it; a mismatch, or a legacy entry without one, warns
%   and rebuilds.  The tag suffix is an optimisation -- it keeps a parameter sweep
%   from overwriting one file repeatedly -- not the correctness mechanism.
%
%   Output struct S:
%     .K0 .Sel .Cblk{n}       the low-rank form (use seq_K(S,n) to materialize)
%     .b{n} .xref{n}          RHS and the backslash ground truth
%     .Ccpl{n} .g{n}          raw coupling block and its RHS at step n
%     .coupling_change(n)     ||C_n - C_{n-1}||_F / ||C_{n-1}||_F
%     .n .nU .nP .nC .N       dimensions
%     .h0 .dt .nu .nsteps .case_name .eps_stab
%     .fingerprint            geometry identity of this sequence (see above)
%     .motion_params          the resolved motion knobs (defaults filled in)
%     .motion_meta            what the factory actually built (omega, Lb, gaps...)
%     .msh .veldofs .velvals .pin_node
%
%   The benchmark advances the state with the DIRECT solution (not the iterative
%   one), so the RHS sequence here is identical to the benchmark's.
%
%   See also: seq_K, seq_dCblk, assert_coupling_feasible,
%             src.stokes.solve_stokes_immersed,
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
    motion_params = getdef(opts, 'motion_params', struct());

    p = add_recycle_paths();

    % ---- the cheap half of the setup, needed to fingerprint BEFORE the cache --
    % The mesh is deliberately NOT built here: it is the only expensive call in
    % this prologue, and a clean cache hit must not pay for it.  Everything below
    % is a handful of arithmetic ops and two motion_fun evaluations.
    [geo, Uin] = channel_geometry(h0, dt, Tstep);
    [mcase, mparams, mtag] = motion_case(case_name, dt, geo, motion_params);
    nu         = mcase.nu;
    fp         = sequence_fingerprint(case_name, h0, dt, Tstep, nsteps, ...
                                      geo, Uin, mcase, mparams);

    % ---- cache lookup ----------------------------------------------------
    % mtag is EMPTY whenever the motion knobs are at their defaults, so every
    % pre-existing cache file keeps its name and hits cleanly; only non-default
    % sweep points get a suffix, which stops a sweep from thrashing one filename
    % through staleCache -> rebuild -> overwrite on every point.
    tag = sprintf('seq_%s_h%s_dt%s_n%d%s', case_name, num2str(h0), num2str(dt), ...
                  nsteps, mtag);
    tag = regexprep(tag, '[^\w]', '_');
    cacheFile = fullfile(p.cacheDir, [tag '.mat']);
    if use_cache && exist(cacheFile, 'file') == 2
        L = load(cacheFile, 'S');
        if ~isfield(L.S, 'fingerprint')
            warning('build_stokes_sequence:cacheNoFingerprint', ...
                    ['cached %s predates the geometry fingerprint, so it cannot ' ...
                     'be checked against the current define_motion_list.m.  ' ...
                     'Rebuilding rather than trusting it: at this tag a file built ' ...
                     'from a different point layout is indistinguishable from a ' ...
                     'correct one.  Happens once per cache file.'], tag);
        else
            why = fingerprint_reason(L.S.fingerprint, fp);
            if isempty(why)
                S = L.S;
                % A hit whose fingerprint matches IS the geometry these params
                % generate, so stamping them on is truthful -- and necessary,
                % since a legacy entry carries neither field and every consumer
                % of a cached S would otherwise see them missing.
                S.motion_params = mparams;
                S.motion_meta   = mcase.motion_meta;
                if ~quiet
                    fprintf(['[seq] loaded cached %s (n=%d, nC=%d, %d steps, ' ...
                             'geometry fingerprint OK)\n'], ...
                            tag, S.n, S.nC, S.nsteps);
                end
                return;
            end
            warning('build_stokes_sequence:staleCache', ...
                    ['cached %s was built from a different problem: %s.  The cache ' ...
                     'tag records (case_name, h0, dt, nsteps) plus non-default ' ...
                     'motion_params; Tstep/nu/the channel box and the remaining ' ...
                     'geometry literals inside define_motion_list.m still do not ' ...
                     'appear in it -- which is why this is checked by content ' ...
                     'instead.  Rebuilding and ' ...
                     'overwriting the file.  This is the intended response to ' ...
                     'editing the geometry; it is not a reason to delete the ' ...
                     'fingerprint.'], tag, why);
        end
        clear L
    end

    % ---- mesh / time-independent fluid blocks -----------------------------
    Lyc = geo.y2 - geo.y1;
    msh = build_channel_mesh_pde(h0, geo.x1, geo.x2, geo.y1, geo.y2, {'rect_right'});
    N   = msh.N;  nU = 2*N;  nP = N;

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
    S.fingerprint   = fp;
    S.motion_params = mparams;
    S.motion_meta   = mcase.motion_meta;

    u_prev = zeros(nU, 1);
    C_prev = [];
    for n = 1:nsteps
        tcur = n * dt;
        mot  = mcase.motion_fun(tcur);
        [C, gvec, nCn] = assemble_coupling(TR, N, mot.X, mot.V);
        assert(nCn == nC, ['nC changed from %d to %d at step %d: the fixed ' ...
               'selector Sel (and the whole low-rank form) assumes a constant ' ...
               'multiplier count.'], nC, nCn, n);

        % Cheap (O(nnz(C))) structural gate: more constraint rows than free
        % velocity DOFs to constrain makes K exactly singular.  Checked EVERY
        % step, not once: nC is asserted constant above but the number of DOFs
        % the points touch is not, and a translating body can drift into a
        % region where its points cluster into fewer elements.
        touched = assert_coupling_feasible(C, veldofs, n, case_name);

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

        if ~all(isfinite(b))
            % Reachable only via a non-finite gvec, i.e. a motion_fun returning
            % NaN/Inf velocities (e.g. omega = 2*pi*nrev/Tmax with Tmax == 0):
            % the guard below keeps u_prev finite, so b's velocity block cannot
            % be the culprit.
            error('build_stokes_sequence:nonfiniteRhs', ...
                  ['step %d: the assembled RHS holds %d non-finite entries.  The ' ...
                   'previous step''s solution is checked, so the prescribed ' ...
                   'rigid-body velocity from motion_fun is the remaining source ' ...
                   '-- check Tmax and the motion parameters for "%s".'], ...
                  n, nnz(~isfinite(b)), case_name);
        end

        xref = K \ b;
        if ~all(isfinite(xref))
            error('build_stokes_sequence:nonfiniteSolution', ...
                  ['step %d: K \\ b returned %d non-finite entries out of %d.  ' ...
                   'K_n is singular or numerically so; MATLAB reports that as a ' ...
                   'warning only, and this value would otherwise be stored as ' ...
                   'xref and fed straight into u_prev, poisoning the RHS of all ' ...
                   '%d remaining steps and every quantity derived from them.  ' ...
                   'Failing here rather than %d steps downstream.  A rank-deficient ' ...
                   'coupling block is the usual cause -- assert_coupling_feasible ' ...
                   'names that case directly when it is visible from the row ' ...
                   'count alone.'], ...
                  n, nnz(~isfinite(xref)), numel(xref), nsteps - n, nsteps - n);
        end

        S.b{n} = b;  S.xref{n} = xref;  S.Ccpl{n} = C;  S.g{n} = gvec;
        S.Xpts{n} = mot.X;              % Lagrange-point positions (mode localization)
        if ~isempty(C_prev) && nnz(C_prev) > 0
            S.coupling_change(n) = norm(C - C_prev, 'fro') / norm(C_prev, 'fro');
        end
        C_prev = C;
        u_prev = xref(1:nU);

        if ~quiet && (mod(n, max(1, round(nsteps/5))) == 0 || n == nsteps)
            fprintf(['  [seq %s] step %3d/%d  nC=%d  touched=%d  ||b||=%.2e  ' ...
                     'dC=%.3f\n'], ...
                    case_name, n, nsteps, nC, touched, norm(b), S.coupling_change(n));
        end
    end

    if use_cache
        % Independent of the per-step guard on purpose: a cached NaN sequence
        % outlives the session that produced it and is then loaded silently by
        % every study sharing this kernel, so weakening the guard above must not
        % be able to produce a corrupt file.  Only b and xref are checked --
        % S.coupling_change(1) is NaN by design.
        bad = find(cellfun(@(v) ~all(isfinite(v)), S.xref(:)) | ...
                   cellfun(@(v) ~all(isfinite(v)), S.b(:)), 1);
        if ~isempty(bad)
            error('build_stokes_sequence:refuseToCacheNonfinite', ...
                  ['refusing to write %s: step %d holds a non-finite b or xref.  ' ...
                   'The per-step guard should have stopped this already; this gate ' ...
                   'is deliberately redundant so that a corrupt sequence can never ' ...
                   'reach disk.'], cacheFile, bad);
        end
        save(cacheFile, 'S', '-v7.3');
        if ~quiet, fprintf('[seq] cached %s\n', cacheFile); end
    end
end

%==========================================================================
function [geo, Uin] = channel_geometry(h0, dt, Tstep)
%CHANNEL_GEOMETRY  The fixed channel and the motion's time horizon.
%   Hoisted out of the build path so the fingerprint can see it: the box and Uin
%   are hardcoded, hence invisible to a cache tag built from opts.
    x1 = 0; x2 = 4; y1 = 0; y2 = 1;
    Uin = 1.0;
    geo = struct('x1', x1, 'x2', x2, 'y1', y1, 'y2', y2, ...
                 'xc', (x1+x2)/2, 'yc', (y1+y2)/2, 'h0', h0, 'Tmax', dt * Tstep);
end

%==========================================================================
function [mcase, mparams, mtag] = motion_case(case_name, dt, geo, motion_params)
%MOTION_CASE  Resolve case_name against define_motion_list and instantiate it.
%   Also returns the RESOLVED motion params (defaults filled in) and this case's
%   cache-tag suffix, which is '' unless a knob the case consumes is non-default.
    [cases, mparams] = define_motion_list(dt, motion_params);
    idx   = find(cellfun(@(c) strcmp(c.name, case_name), cases), 1);
    assert(~isempty(idx), 'unknown case_name "%s"', case_name);
    mcase = cases{idx}.factory(geo);
    mtag  = cases{idx}.params_tag;
end

%==========================================================================
function fp = sequence_fingerprint(case_name, h0, dt, Tstep, nsteps, geo, Uin, mcase, mparams)
%SEQUENCE_FINGERPRINT  Content identity of the problem this call would build.
%
%   The Lagrange-point ARRAYS are the load-bearing part.  X is the OUTPUT of the
%   geometry, so storing it catches every knob that shapes the body -- point
%   spacing, bar half-length, revolutions, disk radius, start position, the
%   0.95*rd clip, the max(8,...) floor -- without naming any of them, which
%   matters because they are hardcoded literals inside define_motion_list.m and
%   no parameter list could ever enumerate them.
%
%   Two time samples, not one: an edit that changes only the later trajectory
%   (e.g. the translation speed) leaves t = dt invariant.  V as well as X,
%   because disk_static's velocity could change with X fixed.
%
%   ~30 KB against 16 MB cache files.
    fp = struct();
    fp.case_name = case_name;
    fp.h0        = h0;
    fp.dt        = dt;
    fp.Tstep     = Tstep;
    fp.Tmax      = geo.Tmax;
    fp.nsteps    = nsteps;
    fp.nu        = mcase.nu;
    fp.is_stress = mcase.is_stress;
    fp.box       = [geo.x1, geo.x2, geo.y1, geo.y2];
    fp.Uin       = Uin;
    % Recorded for forensics only.  Deliberately NOT added to fingerprint_reason's
    % compared lists: every one of the pre-existing cache entries lacks the field,
    % so comparing it would declare all of them stale on the first run.  The knobs
    % are already caught through X/V, and now also through the cache tag.
    fp.motion_params = mparams;

    m1 = mcase.motion_fun(dt);
    m2 = mcase.motion_fun(nsteps * dt);
    fp.X_first = m1.X;  fp.V_first = m1.V;
    fp.X_last  = m2.X;  fp.V_last  = m2.V;
end

%==========================================================================
function why = fingerprint_reason(fpc, fpn)
%FINGERPRINT_REASON  '' if the two fingerprints agree, else a printable reason.
%
%   Arrays are compared by SIZE first -- that is the unambiguous signal when the
%   point layout changed -- then by value with a RELATIVE tolerance.  Not isequal:
%   bar_points runs through sin/cos/linspace, and a one-ulp change from a MATLAB
%   upgrade must not invalidate the whole cache.  1e-12 sits ~4 orders above ulp
%   and ~10 orders below any meaningful geometry edit.
    TOL = 1e-12;
    why = '';

    if ~strcmp(fpc.case_name, fpn.case_name)
        why = sprintf('case_name %s -> %s', fpc.case_name, fpn.case_name);
        return;
    end

    scalars = {'h0','dt','Tstep','Tmax','nsteps','nu','is_stress','Uin'};
    for k = 1:numel(scalars)
        f = scalars{k};
        if ~isfield(fpc, f) || ~isequal(fpc.(f), fpn.(f))
            why = sprintf('%s %g -> %g', f, getfield_or_nan(fpc, f), fpn.(f));
            return;
        end
    end

    arrays = {'box','X_first','V_first','X_last','V_last'};
    for k = 1:numel(arrays)
        f = arrays{k};
        if ~isfield(fpc, f)
            why = sprintf('%s missing from the cached fingerprint', f);
            return;
        end
        A = fpc.(f);  B = fpn.(f);
        if ~isequal(size(A), size(B))
            why = sprintf('%s is %s, cached is %s', f, sizestr(B), sizestr(A));
            return;
        end
        d = max(abs(A(:) - B(:)));
        if d > TOL * max(1, max(abs(B(:))))
            why = sprintf('%s differs by %.3e', f, d);
            return;
        end
    end
end

%==========================================================================
function v = getfield_or_nan(s, f)
    if isfield(s, f), v = double(s.(f)); else, v = NaN; end
end

%==========================================================================
function s = sizestr(A)
    s = sprintf('%dx%d', size(A, 1), size(A, 2));
end

%==========================================================================
function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
