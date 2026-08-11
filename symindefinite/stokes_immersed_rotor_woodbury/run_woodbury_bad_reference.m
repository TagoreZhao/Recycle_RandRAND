%RUN_WOODBURY_BAD_REFERENCE  Drive the Woodbury amplifier with a degenerate
%   REFERENCE, and keep a ground truth while doing it.
%
%   Run:  run_woodbury_bad_reference
%         SMOKE = true; run_woodbury_bad_reference     % h0 = 0.1, 8 steps
%
%   WHAT IS BEING MEASURED.  The Woodbury identity has two cancellation sites
%   (README 4.3).  Site 2 -- the final subtraction x = z - Y0 w, amplifier
%
%       rho = ||K_ref^{-1} b|| / ||K_n^{-1} b||
%
%   -- has never been excited on the physical sequence: rho stays in [0.50, 1.01].
%   rho is large exactly when the REFERENCE amplifies b much more than the target
%   does, so the knob is cond(K_ref) with cond(K_n) held down.
%
%   THE MECHANISM, MEASURED RATHER THAN ASSUMED.  This file's predecessor
%   (run_woodbury_near_wall) asserted that the bar tips reaching the Dirichlet
%   walls was the cause, swept Lb_frac 0.35 -> 0.499, and produced ZERO usable
%   points: five rungs died in woodbury_context_init, two in
%   build_stokes_sequence, one in assert_coupling_feasible.  Its own control
%   refuted its story -- at Lb_frac = 0.35 the tips sit 5*h0 from the wall.  The
%   real mechanism is COLLINEARITY:
%
%     * the bar carries nb = 39 points along a straight line;
%     * a line crosses ~35-39 triangles at a generic angle but only 24-29 when it
%       is parallel to a mesh axis;
%     * 39 samples of a function that is piecewise linear on ~28 segments are not
%       independent, so W(:,free) -- and with it C, and with it K -- loses rank.
%
%   Measured at h0 = 0.03: sigma_min/sigma_max of W(:,free) is 2.6e-2 at a generic
%   angle and 2.5e-12 at 0, 90 and 180 degrees, with row 1-norms pinned at 1.0
%   (no wall starvation anywhere) and the null vector spread across all rows.
%   Reducing the point count recovers it monotonically -- 39 -> 1.0e-13,
%   30 -> 2.6e-5, 24 -> 3.3e-1, where nb finally matches the triangles crossed.
%   Wall starvation is real but is a DIFFERENT mechanism: at Lb_frac = 0.499 the
%   minimum row 1-norm falls to 0.038 and 99.7% of the null vector sits on the two
%   tip rows.  The old header conflated the two; this file drives only the first,
%   at the shipped Lb_frac = 0.35, so nC = 78 stays comparable with the benchmark.
%
%   WHY THE OFFSET IS FROM HORIZONTAL.  Off VERTICAL, cond(K) is a step function:
%   1e7 for |delta| >= 0.8 deg and 1e25 below, with nothing between -- no ladder.
%   Off HORIZONTAL it is smooth and spans the whole range at fixed nC = 78:
%
%       delta (deg)   0.3    0.03    0.003   1e-3    1e-4    3e-5    1e-5     0
%       condest(K)   7.4e6   1.2e8   3.1e9   2.5e10  2.5e12  2.7e13  2.4e14  3.8e24
%
%   THE LADDER IS NOT CALIBRATED, IT IS MEASURED.  generateMesh is not stable
%   across MATLAB releases, so delta -> cond is machine specific.  Rather than
%   bisect at run time to hit nominal rungs, the sweep uses a fixed delta ladder
%   and REPORTS the measured condest(K_ref) as the abscissa.  A different mesh
%   slides the points along that axis, which is honest; a calibrated delta would
%   instead hide the shift inside a label.
%
%   THE OFFSET IS NOT delta ITSELF.  build_stokes_sequence evaluates step n at
%   t = n*dt, so step 1 is t = dt, not t = 0, and the phase offset must be
%
%       theta0 = delta - omega*dt = delta - 2*pi*nrev/Tstep      (dt cancels)
%
%   test_motion_params T6 pins that mapping; local_assert_placed re-derives the
%   step-1 angle from the stored points and fails loudly if it moves.
%
%   THE GROUND TRUTH, WHICH IS THE WHOLE POINT.  Stressing the operator degrades
%   the reference too, so "Woodbury is wrong by 1e-12" is worthless if K_n\b is
%   also wrong by 1e-12.  Three independent answers to that:
%
%     1. The headline metrics are REFERENCE-FREE -- relative residual and normwise
%        backward error need no true solution at all.
%     2. Every forward error carries a measured uncertainty (one step of
%        iterative refinement on K_n) and is printed through woodbury_mask_error,
%        which flags with '~' any value not 10x clear of it.
%     3. Arm E manufactures the right-hand side from a KNOWN x_true, so its error
%        is exact by construction and involves no reference solve whatever.
%
%   THE TARGETS ARE SCREENED ON MEASURED CONDITIONING, NOT ON ANGLE.  The angle
%   rule (ANG_G degrees clear of a mesh axis) is only a pre-filter, and it is not
%   sufficient: on the shipped sequence step 10 sits at 118.0 deg -- 28 deg from
%   any axis -- yet condest(K_10) is 1.4e12 against ~1e7 at every other step.  A
%   straight line through an unstructured mesh can lose rank at angles no
%   geometric rule predicts.  Selecting targets by angle alone let that step in
%   and pushed the worst target to 4.3e12, leaving barely a factor 60 between
%   anchor and target; screening on condest(K_n) < CONDMAX restores the intended
%   separation of ~1e6 and is the same lesson as the paragraph above -- measure
%   the conditioning, do not infer it from the geometry.
%
%   A degenerate reference does NOT poison the chain: at condest(K_ref) = 2.7e13,
%   iterative refinement moves the VELOCITY block of K\b -- the block that becomes
%   u_prev -- by 1.3e-15, because C = blkdiag(W,W) puts the near-null direction in
%   the multipliers.
%
%   ARMS
%     B  treatment  ref = 1, the degenerate step, physical RHS
%     C  control    SAME sequence, ref = a healthy step: only the anchor changes
%     E  exact      same sequences, b = K_n*x_true for known x_true (random, and
%                   aligned with the reference's near-null direction)
%     A  baseline   the shipped theta0 = 0 sequence, ref swept over every step
%
%   It writes one figure and no CSVs.
%
%   See also: woodbury_context_init, woodbury_mask_error, woodbury_solve,
%             run_woodbury_stability, define_motion_list.

% NOTE no `clear` -- that would wipe a SMOKE flag set from the base workspace,
% which is how run_woodbury_benchmark's SMOKE_TEST is driven too.
clc;
paths = add_woodbury_paths();
assert_woodbury_helpers();
rng(0);

if ~exist('SMOKE', 'var') || isempty(SMOKE), SMOKE = false; end

params = make_woodbury_params();

CASE   = 'bar_rotating';
NREV   = 2;      % must match make_bar_rotating's literal; local_assert_placed catches drift
LB     = 0.35;   % the SHIPPED half-length: nC = 78, comparable with the cost table
ANG_G  = 6;      % deg from any multiple of 90 that a TARGET must clear.  The
                 % collapse is within ~1 deg (srat 2e-3 at 1 deg vs 2.6e-2 median);
                 % 6 deg is 6x that and still leaves 13 of 14 candidate steps.
CONDMAX = 1e8;   % a target above this is reported but excluded from the headline
NPROBE  = 6;

if SMOKE
    H0 = 0.1;   NSTEPS = 8;   NPROBE = 3;
    % The collapse width scales with h0/Lb -- it is set by how far a point must
    % travel to enter a new triangle -- so the h0 = 0.03 ladder is NOT
    % transferable.  Measured at h0 = 0.1: 10 deg -> 7.0e5, 5 deg -> 2.7e7,
    % 3 deg -> 3.4e30.  There is no smooth band at this resolution, only a cliff
    % between 5 and 3 deg, so SMOKE is a plumbing test and not a ladder.  The
    % 4 deg rung sits on the cliff deliberately: it exercises the infeasible-rung
    % reporting path that the production ladder is built to avoid.
    DELTA = [10 5 4];
else
    H0 = params.h0;  NSTEPS = 15;
    % 15 steps = 14*11.80 = 165.2 deg < 180, so the bar passes an axis exactly
    % once in the window.  At 20 steps it spans 236 deg and a SECOND near-axis
    % step lands at step ~4.75, inside the contamination width -- a trap the
    % predecessor walked into.
    DELTA = [0.3 0.03 0.003 1e-3 1e-4 3e-5 1e-5];
end

fprintf('=== woodbury bad reference: %s, h0 = %g, %d steps%s ===\n', ...
        CASE, H0, NSTEPS, local_tern(SMOKE, '  [SMOKE]', ''));
fprintf(['Lb_frac = %g (shipped, nC fixed), offset from HORIZONTAL, ' ...
         'theta0 = delta - 2*pi*%d/%d\n\n'], LB, NREV, params.Tstep);

nR  = numel(DELTA);
B   = local_alloc(nR);
C   = local_alloc(nR);
E   = local_alloc(nR);
cal = struct('delta', DELTA(:), 'kref', nan(nR,1), 'rcD', nan(nR,1), ...
             'relres', nan(nR,1), 'nC', nan(nR,1), 'margin', nan(nR,1), ...
             'refC', nan(nR,1), 'ktgt', nan(nR,1), 'err_id', {cell(nR,1)}, ...
             'probes', {cell(nR,1)});

for i = 1:nR
    [B, C, E, cal] = local_rung(B, C, E, cal, i, CASE, H0, NSTEPS, LB, ...
                                DELTA(i), NREV, params, ANG_G, CONDMAX, NPROBE);
end

A = local_arm_a(CASE, H0, NSTEPS, params, ANG_G, CONDMAX, NPROBE);

% ---- tables ---------------------------------------------------------------
local_table_cal(cal, CONDMAX);
local_table_arm('B  TREATMENT: ref = 1 (the degenerate step), physical RHS', B, cal);
local_table_arm('C  CONTROL: same sequence, healthy reference, physical RHS', C, cal);
local_table_exact(E, cal);
local_table_a(A);
local_verdict(B, C, E, A, cal, CONDMAX);

% ---- figure ---------------------------------------------------------------
outDir  = local_tern(SMOKE, paths.smokeDir, paths.outDir);
outFile = fullfile(outDir, sprintf('bad_reference_h%s_n%d.png', ...
                   strrep(num2str(H0), '.', 'p'), NSTEPS));
local_figure(B, C, E, A, cal, H0, CASE, outFile);

%==========================================================================
function F = local_alloc(n)
%LOCAL_ALLOC  One row per rung.  The scalar fields hold the WORST probe of that
%   rung -- this is a stability study, so the headline is the number that would
%   bite -- and the cell fields keep every probe for the scatter panels.
    z = nan(n, 1);
    F = struct('rho', z, 'csub', z, 'ccap', z, 'capcond', z, 'fwd', z, ...
               'res', z, 'bwd', z, 'floor', z, 'fresh_res', z, ...
               'fresh_fwd', z, 'unc', z, 'ktgt', z, ...
               'allrho', {cell(n,1)}, 'allfwd', {cell(n,1)}, ...
               'allres', {cell(n,1)}, 'allbwd', {cell(n,1)}, ...
               'allunc', {cell(n,1)}, 'allfresh', {cell(n,1)}, ...
               'allfloor', {cell(n,1)}, 'allcsub', {cell(n,1)});
end

%==========================================================================
function [B, C, E, cal] = local_rung(B, C, E, cal, i, CASE, H0, NSTEPS, LB, ...
                                     delta, NREV, params, ANG_G, CONDMAX, NPROBE)
%LOCAL_RUNG  One ladder rung: build, freeze twice, probe.  Never throws.
%   woodbury_context_init still refuses a genuinely singular reference and
%   build_stokes_sequence can refuse the geometry, so every rung is wrapped and
%   the identifier reported -- where the ladder ends IS one of the results.
    cal.err_id{i} = '';
    ws = warning;                    % FULL state.  Restoring only the first
    warning('off', 'woodbury_solve:singularCapacitance');  % identifier would leave
    warning('off', 'MATLAB:nearlySingularMatrix');         % the other two off for
    warning('off', 'MATLAB:singularMatrix');               % the rest of the session
    try
        theta0 = deg2rad(delta) - 2*pi*NREV/params.Tstep;
        S = build_stokes_sequence(struct('case_name', CASE, 'h0', H0, ...
                'dt', params.dt, 'Tstep', params.Tstep, 'nsteps', NSTEPS, ...
                'verify', false, 'use_cache', true, 'quiet', true, ...
                'motion_params', struct('Lb_frac', LB, 'theta0', theta0)));

        local_assert_placed(S, 1, deg2rad(delta));
        ang       = local_angles(S);
        cal.nC(i) = S.nC;

        tch = inf;
        for n = 1:S.nsteps
            tch = min(tch, assert_coupling_feasible(S.Ccpl{n}, S.veldofs, n, CASE));
        end
        cal.margin(i) = tch - S.nC;

        [probe, refC, kprobe] = local_pick(S, ang, ANG_G, CONDMAX, NPROBE);
        cal.probes{i} = probe;
        cal.refC(i)   = refC;

        ctxB = woodbury_context_init(S, 1);
        ctxC = woodbury_context_init(S, refC);
        cal.kref(i)   = local_condest(seq_K(S, 1));
        cal.rcD(i)    = ctxB.rcond_D;
        cal.relres(i) = ctxB.apply_relres;

        % The reference's near-null direction, for arm E.  C = blkdiag(W,W) after
        % the Dirichlet mask, so K_ref*[0;0;y] = [Cu*y; 0; 0]: the near-null
        % direction of K_ref is the smallest right singular vector of Cu.
        % Aligning x_true with it is what actually drives rho.
        [~, ~, Vn] = svd(full(S.Cblk{1}(1:S.nU, :)), 'econ');
        vnull = [zeros(S.nU + S.nP, 1); Vn(:, end)];

        accB = local_acc(numel(probe));
        accC = local_acc(numel(probe));
        accE = local_acc(numel(probe));
        for j = 1:numel(probe)
            n  = probe(j);
            b  = S.b{n};
            Kn = seq_K(S, n);
            dK = decomposition(Kn, 'ldl');
            kt = kprobe(j);                 % measured during selection, not twice

            % THE reference, and its own error: one step of fixed-precision
            % iterative refinement reusing the same factors.  S.xref{n} is K\b
            % from the build -- the same algorithm on the same matrix, so it is a
            % reproducibility check, not an independent one.
            xref = dK \ b;
            unc  = norm(dK \ (b - Kn*xref)) / max(norm(xref), eps);
            fres = norm(b - Kn*xref) / max(norm(b), eps);

            [xB, iB] = woodbury_solve(ctxB, S, n, b);
            [xC, iC] = woodbury_solve(ctxC, S, n, b);
            accB = local_fill(accB, j, xB, iB, xref, Kn, b, unc, fres, kt);
            accC = local_fill(accC, j, xC, iC, xref, Kn, b, unc, fres, kt);

            accE = local_exact(accE, j, ctxB, S, n, Kn, dK, vnull, kt);
        end
        B = local_reduce(B, i, accB);
        C = local_reduce(C, i, accC);
        E = local_reduce(E, i, accE);
        cal.ktgt(i) = max(accB.ktgt);
    catch ME
        cal.err_id{i} = ME.identifier;
        if isempty(cal.err_id{i}), cal.err_id{i} = '(unidentified)'; end
        fprintf(2, '  delta = %.4g deg -> %s\n', delta, ME.message);
    end
    warning(ws);
end

%==========================================================================
function [probe, refC, kn] = local_pick(S, ang, ANG_G, CONDMAX, NPROBE)
%LOCAL_PICK  Targets screened on MEASURED conditioning, plus the control anchor.
%
%   The angle test is only a cheap pre-filter.  It is not sufficient, and the
%   first full run proved it: on the shipped sequence, step 10 sits at 118.0 deg
%   -- nowhere near a mesh axis, and 28 deg clear of one -- yet condest(K_10) is
%   1.4e12 against ~1e7 everywhere else.  Selecting targets by angle let that
%   step into the probe set and pushed the "worst target" column to 4.3e12,
%   destroying the separation between anchor and target that the whole design
%   rests on.  A straight line through an unstructured mesh can lose rank at
%   angles no geometric rule predicts, which is the same lesson as the header's:
%   measure the conditioning, do not infer it.
%
%   The control anchor is the BEST-CONDITIONED candidate, and it is removed from
%   the probe set -- at n == ref the update is identically zero, so comparing
%   there would be vacuous.
%
%   Returns KN, the measured condest of each returned probe, so the caller does
%   not pay for it twice.
    cand = find(ang > ANG_G);
    cand = cand(cand > 1);
    if isempty(cand)
        error('run_woodbury_bad_reference:noCleanTarget', ...
              'no step clears %g deg from a mesh axis', ANG_G);
    end

    kc = arrayfun(@(n) local_condest(seq_K(S, n)), cand);
    keep = kc < CONDMAX;
    if nnz(keep) < 2
        error('run_woodbury_bad_reference:noHealthyTarget', ...
              ['only %d of %d angle-eligible steps have condest(K_n) < %.0e ' ...
               '(best %.3e): there is no well-posed target to measure against, ' ...
               'so any error reported here would be the target''s.'], ...
              nnz(keep), numel(cand), CONDMAX, min(kc));
    end
    cand = cand(keep);  kc = kc(keep);

    [~, jb]  = min(kc);
    refC     = cand(jb);
    cand(jb) = [];  kc(jb) = [];
    sel   = unique(round(linspace(1, numel(cand), min(NPROBE, numel(cand)))));
    probe = reshape(cand(sel), 1, []);
    kn    = reshape(kc(sel), 1, []);
end

%==========================================================================
function acc = local_acc(np)
    z = nan(np, 1);
    acc = struct('rho', z, 'csub', z, 'ccap', z, 'capcond', z, 'fwd', z, ...
                 'res', z, 'bwd', z, 'floor', z, 'fresh_res', z, ...
                 'fresh_fwd', z, 'unc', z, 'ktgt', z);
end

%==========================================================================
function acc = local_fill(acc, j, x, info, xref, Kn, b, unc, fres, kt)
%LOCAL_FILL  Reference-free metrics first; the reference-dependent one carries
%   the reference's own uncertainty alongside so it can be masked when printed.
    r  = b - Kn * x;
    bn = max(norm(b), eps);
    acc.rho(j)     = info.rho;
    acc.csub(j)    = info.cancel_sub;
    acc.ccap(j)    = info.cancel_cap;
    acc.capcond(j) = info.cap_cond;
    acc.res(j)     = norm(r) / bn;
    acc.bwd(j)     = norm(r) / (norm(Kn, 'fro') * norm(x) + bn);
    acc.floor(j)   = eps * norm(Kn, 1) * norm(x, 1) / max(norm(b, 1), eps);
    acc.fwd(j)     = norm(x - xref) / max(norm(xref), eps);
    acc.unc(j)     = unc;
    acc.fresh_res(j) = fres;
    acc.ktgt(j)      = kt;
end

%==========================================================================
function acc = local_exact(acc, j, ctx, S, n, Kn, dK, vnull, kt)
%LOCAL_EXACT  Arm E: manufacture b from a KNOWN x_true, so the error is exact.
%   Two directions are tried and the WORSE is kept, with its own rho: a random
%   one, and the reference's near-null direction.  Only the second is expected to
%   drive rho, and the random control is what stops "the manufactured RHS is just
%   scaled oddly" -- if a random direction gave the same rho, the amplification
%   would not be a property of the reference.  Only the surviving pair is
%   recorded; which direction won is visible in the rho column.
    cand = {randn(S.n, 1), vnull};
    best = -inf;
    for k = 1:numel(cand)
        xt = cand{k} / norm(cand{k});
        be = Kn * xt;
        [xe, ie] = woodbury_solve(ctx, S, n, be);
        er = norm(xe - xt);                      % ||xt|| == 1
        if er > best
            best = er;
            acc.rho(j)       = ie.rho;
            acc.csub(j)      = ie.cancel_sub;
            acc.ccap(j)      = ie.cancel_cap;
            acc.capcond(j)   = ie.cap_cond;
            acc.fwd(j)       = er;
            acc.res(j)       = norm(be - Kn*xe) / max(norm(be), eps);
            acc.bwd(j)       = norm(be - Kn*xe) / ...
                               (norm(Kn,'fro')*norm(xe) + max(norm(be), eps));
            acc.floor(j)     = eps * norm(Kn,1) * norm(xe,1) / max(norm(be,1), eps);
            acc.fresh_fwd(j) = norm((dK \ be) - xt);
            acc.fresh_res(j) = acc.fresh_fwd(j);
            acc.unc(j)       = 0;                % exact by construction
            acc.ktgt(j)      = kt;
        end
    end
end

%==========================================================================
function F = local_reduce(F, i, acc)
    F.rho(i)       = max(acc.rho);
    F.csub(i)      = max(acc.csub);
    F.ccap(i)      = max(acc.ccap);
    F.capcond(i)   = max(acc.capcond);
    F.fwd(i)       = max(acc.fwd);
    F.res(i)       = max(acc.res);
    F.bwd(i)       = max(acc.bwd);
    F.floor(i)     = max(acc.floor);
    F.fresh_res(i) = max(acc.fresh_res);
    F.fresh_fwd(i) = max(acc.fresh_fwd);
    F.unc(i)       = max(acc.unc);
    F.ktgt(i)      = max(acc.ktgt);
    F.allrho{i}   = acc.rho(:)';
    F.allfwd{i}   = acc.fwd(:)';
    F.allres{i}   = acc.res(:)';
    F.allbwd{i}   = acc.bwd(:)';
    F.allunc{i}   = acc.unc(:)';
    F.allfresh{i} = acc.fresh_res(:)';
    F.allfloor{i} = acc.floor(:)';
    F.allcsub{i}  = acc.csub(:)';
end

%==========================================================================
function A = local_arm_a(CASE, H0, NSTEPS, params, ANG_G, CONDMAX, NPROBE)
%LOCAL_ARM_A  The shipped sequence (theta0 = 0), with the anchor swept over every
%   step.  This is NOT the instrument: README 4.1 measures kappa(K_n) spanning
%   only 2.54e6 to 3.07e6 across the sequence, a 1.2x spread, so no choice of ref
%   on a healthy sequence can produce rho >> 1.  Its jobs are to supply the
%   rho ~ 1 baseline cluster, to exercise the ref != 1 path end to end, and to
%   show that a healthy reference is benign AT EVERY INDEX -- so that the ladder's
%   amplification cannot be attributed to the anchor merely being far from the
%   target in phase.
    A = struct('ref', [], 'kref', [], 'rho', [], 'res', [], 'fwd', [], ...
               'err_id', '', 'probes', []);
    ws = warning;
    warning('off', 'woodbury_solve:singularCapacitance');
    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');
    try
        S = build_stokes_sequence(struct('case_name', CASE, 'h0', H0, ...
                'dt', params.dt, 'Tstep', params.Tstep, 'nsteps', NSTEPS, ...
                'verify', false, 'use_cache', true, 'quiet', true));
        ang = local_angles(S);
        [probe, ~] = local_pick(S, ang, ANG_G, CONDMAX, NPROBE);
        A.probes = probe;

        % Hoisted: none of this depends on the anchor, and paying it inside the
        % ref loop would multiply the cost by nsteps for no information.
        np = numel(probe);
        Kc = cell(np,1);  dc = cell(np,1);  xr = cell(np,1);
        for j = 1:np
            Kc{j} = seq_K(S, probe(j));
            dc{j} = decomposition(Kc{j}, 'ldl');
            xr{j} = dc{j} \ S.b{probe(j)};
        end

        nr = S.nsteps;
        A.ref = (1:nr)';  A.kref = nan(nr,1);
        A.rho = nan(nr,1); A.res = nan(nr,1); A.fwd = nan(nr,1);
        for r = 1:nr
            ctx = woodbury_context_init(S, r);
            A.kref(r) = local_condest(seq_K(S, r));
            rr = -inf; re = -inf; fe = -inf;
            for j = 1:np
                n = probe(j);
                if n == r, continue; end       % dC == 0 there: vacuous
                b = S.b{n};
                [x, info] = woodbury_solve(ctx, S, n, b);
                rr = max(rr, info.rho);
                re = max(re, norm(b - Kc{j}*x) / max(norm(b), eps));
                fe = max(fe, norm(x - xr{j}) / max(norm(xr{j}), eps));
            end
            A.rho(r) = rr;  A.res(r) = re;  A.fwd(r) = fe;
        end
    catch ME
        A.err_id = ME.identifier;
        fprintf(2, '  arm A -> %s\n', ME.message);
    end
    warning(ws);
end

%==========================================================================
function ang = local_angles(S)
%LOCAL_ANGLES  Degrees from the nearest mesh axis, per step, re-derived from the
%   STORED point positions rather than from theta0 -- so it survives any change
%   to how the phase is threaded through define_motion_list.
    ang = nan(S.nsteps, 1);
    for n = 1:S.nsteps
        X  = S.Xpts{n};
        th = mod(rad2deg(atan2(X(end,2) - X(1,2), X(end,1) - X(1,1))), 90);
        ang(n) = min(th, 90 - th);
    end
end

%==========================================================================
function local_assert_placed(S, r, th_target)
%LOCAL_ASSERT_PLACED  Non-vacuity: the offset really put the intended angle on
%   step r.  A bar is a LINE, so its angle is only defined modulo pi.
    X  = S.Xpts{r};
    th = atan2(X(end,2) - X(1,2), X(end,1) - X(1,1));
    d  = mod(th - th_target + pi/2, pi) - pi/2;
    assert(abs(d) < 1e-9, ...
           ['step %d sits at %.9f rad, not the intended %.9f: the phase offset ' ...
            'did not place the bar as asked (theta0 = target - 2*pi*nrev/Tstep, ' ...
            'because step n is evaluated at t = n*dt).'], r, th, th_target);
end

%==========================================================================
function k = local_condest(A)
%LOCAL_CONDEST  condest without leaking its RNG draw or its warnings.
%   condest estimates ||A^{-1}||_1 with normest1, which draws from the GLOBAL
%   random stream; leaving that shift in place would make every later random draw
%   depend on how many condests happened to run.
    st = rng;
    ws = warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');
    k = condest(A);
    warning(ws);
    rng(st);
end

%==========================================================================
function [xx, yy] = local_xy(x, y)
%LOCAL_XY  Only the pairs a log axis can actually show.
%   Replaces the predecessor's `fl = @(v) max(v,1e-18)`: MATLAB's max IGNORES
%   NaN, so that idiom silently plotted every failed rung as a point at 1e-18 --
%   which is why the committed near_wall figure shows flat lines at 1e-18 and one
%   degenerate marker.  Dropping the pair is the only honest option.
    x = x(:);  y = y(:);
    keep = isfinite(x) & isfinite(y) & x > 0 & y > 0;
    xx = x(keep);  yy = y(keep);
end

%==========================================================================
function v = local_cat(c)
%LOCAL_CAT  Flatten the per-probe cell columns into one scatter vector.
    v = [];
    for i = 1:numel(c)
        v = [v, c{i}];  %#ok<AGROW>  -- nR is single digits
    end
    v = v(:);
end

%==========================================================================
function local_table_cal(cal, CONDMAX)
    fprintf('\n--- calibration: what each rung actually built ---\n');
    fprintf('%10s %12s %12s %12s %6s %7s %5s %12s %s\n', 'delta_deg', ...
            'condest(Kref)', '1/condest(D)', 'apply_relres', 'nC', 'margin', ...
            'refC', 'max cond(Kn)', 'status');
    for i = 1:numel(cal.delta)
        if ~isempty(cal.err_id{i})
            fprintf('%10.4g %12s   %s\n', cal.delta(i), '-', cal.err_id{i});
            continue;
        end
        flag = '';
        if cal.ktgt(i) > CONDMAX
            flag = '  <-- SCREEN LEAKED';    % must not happen: local_pick enforces it
        end
        fprintf('%10.4g %12.3e %12.3e %12.3e %6d %7d %5d %12.3e ok%s\n', ...
                cal.delta(i), cal.kref(i), cal.rcD(i), cal.relres(i), ...
                cal.nC(i), cal.margin(i), cal.refC(i), cal.ktgt(i), flag);
    end
    ok = isfinite(cal.kref);
    if any(ok)
        fprintf(['  the ladder spans condest(K_ref) %.2e -> %.2e while every ' ...
                 'TARGET stays below\n  %.2e (screen %.0e): up to a factor %.1e ' ...
                 'between the anchor and what it is used\n  to solve, which is ' ...
                 'the separation the whole design rests on.\n'], ...
                min(cal.kref(ok)), max(cal.kref(ok)), max(cal.ktgt(ok)), ...
                CONDMAX, max(cal.kref(ok)) / max(cal.ktgt(ok)));
    end
end

%==========================================================================
function local_table_arm(name, F, cal)
%LOCAL_TABLE_ARM  Reference-free metrics FIRST -- they are the headline and need
%   no ground truth.  wood_fwd is printed through woodbury_mask_error, so a value
%   the reference was not accurate enough to have measured carries a '~'.
    fprintf('\n--- %s ---\n', name);
    fprintf('%13s %10s %10s %11s %11s %11s %11s %12s %11s\n', ...
            'cond(Kref)', 'rho', 'cancl_sub', 'wood_res', 'wood_bwd', ...
            'res_floor', 'fresh_res', 'wood_fwd', 'ref_unc');
    for i = 1:numel(cal.delta)
        if ~isempty(cal.err_id{i}) || isnan(F.rho(i)), continue; end
        fprintf('%13.3e %10.3e %10.3e %11.3e %11.3e %11.3e %11.3e %12s %11.3e\n', ...
                cal.kref(i), F.rho(i), F.csub(i), F.res(i), F.bwd(i), ...
                F.floor(i), F.fresh_res(i), ...
                woodbury_mask_error(F.fwd(i), F.unc(i)), F.unc(i));
    end
    fprintf('  ("~" on wood_fwd: not 10x clear of ref_unc, so the REFERENCE, not\n');
    fprintf('   Woodbury, is what that number measures -- read wood_res instead.)\n');
end

%==========================================================================
function local_table_exact(E, cal)
    fprintf('\n--- E  EXACT: b = K_n*x_true, so there is no reference at all ---\n');
    fprintf('%13s %10s %10s %12s %12s %12s %10s\n', 'cond(Kref)', 'rho', ...
            'cancl_sub', 'wood_err', 'fresh_err', 'cSub*eps', 'err/cSubEps');
    for i = 1:numel(cal.delta)
        if ~isempty(cal.err_id{i}) || isnan(E.rho(i)), continue; end
        fprintf('%13.3e %10.3e %10.3e %12.3e %12.3e %12.3e %10.2e\n', ...
                cal.kref(i), E.rho(i), E.csub(i), E.fwd(i), E.fresh_fwd(i), ...
                E.csub(i)*eps, E.fwd(i)/max(E.csub(i)*eps, realmin));
    end
    fprintf(['  x_true is known, so wood_err and fresh_err are EXACT forward ' ...
             'errors -- no reference\n  is involved.  The last column falling ' ...
             'towards 1 as the ladder climbs is the crossover:\n  low rungs are ' ...
             'limited by the target''s own conditioning, high rungs by site 2.\n']);
end

%==========================================================================
function local_table_a(A)
    fprintf('\n--- A  BASELINE: shipped theta0 = 0, anchor swept over every step ---\n');
    if ~isempty(A.err_id)
        fprintf('  arm A did not run: %s\n', A.err_id);
        return;
    end
    fprintf('  probes: %s\n', mat2str(A.probes));
    fprintf('%6s %13s %10s %11s %11s\n', 'ref', 'cond(Kref)', 'max rho', ...
            'max wood_res', 'max wood_fwd');
    for r = 1:numel(A.ref)
        fprintf('%6d %13.3e %10.3e %11.3e %11.3e\n', ...
                A.ref(r), A.kref(r), A.rho(r), A.res(r), A.fwd(r));
    end
    ok = isfinite(A.rho);
    fprintf(['  a healthy sequence gives rho in [%.3g, %.3g] at EVERY anchor and ' ...
             'cond(K_ref) in\n  [%.2e, %.2e] -- so moving the anchor is not by ' ...
             'itself what amplifies.\n'], ...
            min(A.rho(ok)), max(A.rho(ok)), min(A.kref), max(A.kref));
end

%==========================================================================
function local_verdict(B, C, E, A, cal, CONDMAX)
%LOCAL_VERDICT  Generated from the numbers, not asserted in advance.
    fprintf('\n--- what the run says ---\n');
    ok  = isfinite(B.rho) & isfinite(cal.kref);
    bad = find(~cellfun(@isempty, cal.err_id));
    fprintf('%d of %d rungs are feasible.\n', nnz(ok), numel(cal.delta));
    for k = bad(:)'
        fprintf('  delta = %.4g deg INFEASIBLE: %s\n', cal.delta(k), cal.err_id{k});
    end
    if ~any(ok)
        fprintf('Nothing to report: no rung produced a probe.\n');
        return;
    end

    % NOT `max(B.rho .* ok)`: NaN*0 is NaN, so that idiom reads as a guard and is
    % none -- it is the predecessor's line 123 bug.  Mask by assignment instead.
    rsel = B.rho;  rsel(~ok) = -inf;
    [rmax, im] = max(rsel);
    fprintf(['\nWorst treatment rung: delta = %.4g deg, condest(K_ref) = %.3e, ' ...
             'rho = %.3g.\n'], cal.delta(im), cal.kref(im), rmax);
    fprintf(['  reference-free: wood_res %.3e vs fresh_res %.3e (factor %.3g), ' ...
             'floor %.3e\n'], B.res(im), B.fresh_res(im), ...
            B.res(im)/max(B.fresh_res(im), realmin), B.floor(im));
    fprintf('  exact arm at the same rung: wood_err %.3e vs fresh_err %.3e\n', ...
            E.fwd(im), E.fresh_fwd(im));

    % Did the ANCHOR do it?  C holds geometry, targets and RHS fixed and moves
    % only the reference, so B/C is the cleanest attribution available.
    if isfinite(C.rho(im))
        fprintf(['\nControl at the same rung (healthy anchor, step %d, same ' ...
                 'targets and RHS):\n  rho %.3g vs %.3g, wood_res %.3e vs %.3e. ' ...
                 'The amplifier is the REFERENCE.\n'], ...
                cal.refC(im), C.rho(im), B.rho(im), C.res(im), B.res(im));
    end

    % WHICH SCALE GOVERNS.  Three different mechanisms produce the same symptom,
    % so the candidates are ranked by how close err/scale sits to 1 -- not tested
    % against a single threshold, which cannot tell "both scales fit" from
    % "neither does".  The exact arm is used because its error needs no reference.
    nm = {'cancel_sub*eps  (site 2, the final subtraction)', ...
          'cond(K_ref)*eps (a degenerate reference)', ...
          'cond(K_n)*eps   (the target''s own conditioning)'};
    sc = [E.csub(im)*eps, cal.kref(im)*eps, B.ktgt(im)*eps];
    rt = E.fwd(im) ./ max(sc, realmin);
    fprintf('\nWhich scale governs the EXACT error %.3e:\n', E.fwd(im));
    for k = 1:3
        fprintf('  %-46s %9.3e   err/scale = %8.2e\n', nm{k}, sc(k), rt(k));
    end
    [~, kb] = min(abs(log10(rt)));
    if abs(log10(rt(kb))) < 2
        fprintf('  => closest is %s.\n', nm{kb});
        if kb == 1
            fprintf(['     Cancellation SITE 2 is excited -- the site README 4.4 ' ...
                     'records as never\n     reached on the shipped sequence.\n']);
        end
    else
        fprintf('  => no candidate is within 2 decades; report all three, claim none.\n');
    end

    fprintf(['\ncancel_cap stays %.3g at that rung against cancel_sub %.3g: ' ...
             'whatever drives\n  this, it is not the capacitance site.\n'], ...
            B.ccap(im), B.csub(im));

    % Was the forward error worth reading at all?
    [~, rep] = woodbury_mask_error(B.fwd(im), B.unc(im));
    fprintf(['\nGround truth at that rung: ref_unc %.3e against wood_fwd %.3e -- ' ...
             'the forward\n  error is %s.  cond(K_n) stayed at %.3e (limit %.0e), ' ...
             'so K_n\\b is sound.\n'], ...
            B.unc(im), B.fwd(im), ...
            local_tern(rep, 'REPORTABLE', 'reference-limited, read wood_res'), ...
            B.ktgt(im), CONDMAX);
    if isempty(A.err_id) && any(isfinite(A.rho))
        fprintf('  arm A baseline: max rho over all %d anchors = %.3g.\n', ...
                numel(A.ref), max(A.rho(isfinite(A.rho))));
    end
end

%==========================================================================
function local_figure(B, C, E, A, cal, H0, CASE, outFile)
    opts = woodbury_fig_defaults();
    fh = figure('Visible', 'off', 'Units', 'inches', ...
                'Position', [1 1 opts.multi_width opts.multi_height + 1]);
    tl = tiledlayout(fh, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('%s: degenerate reference, h_0 = %g, n_C = %d', ...
          strrep(CASE, '_', '\_'), H0, max([cal.nC(isfinite(cal.nC)); NaN])), ...
          'FontSize', opts.titlefontsize, 'Interpreter', 'tex');

    % -- 1  the ladder is real, and the targets do not move ----------------
    ax = nexttile(tl);  hold(ax, 'on');
    local_panel(ax, { ...
        {cal.delta, cal.kref, '-o',  1.6, 5, 'reference K_{ref}'}, ...
        {cal.delta, cal.ktgt, '--s', 1.2, 4, 'worst target K_n'}}, opts);
    set(ax, 'XDir', 'reverse');
    xlabel(ax, 'bar angle off the mesh axis (deg)');
    ylabel(ax, 'condest');
    title(ax, 'the ladder, and that the targets stay put', ...
          'FontSize', opts.subtitlefontsize);

    % -- 2  does a worse reference buy amplification? -----------------------
    kB = repelem(cal.kref, cellfun(@numel, B.allrho));
    kC = repelem(cal.kref, cellfun(@numel, C.allrho));
    kE = repelem(cal.kref, cellfun(@numel, E.allrho));
    ax = nexttile(tl);  hold(ax, 'on');
    local_panel(ax, { ...
        {kB, local_cat(B.allrho), 'o', 1.2, 5, 'B physical'}, ...
        {kE, local_cat(E.allrho), '^', 1.2, 5, 'E manufactured'}, ...
        {kC, local_cat(C.allrho), 's', 1.0, 4, 'C healthy anchor'}, ...
        {A.kref, A.rho,           'x', 1.2, 6, 'A baseline'}}, opts);
    xlabel(ax, 'condest(K_{ref})');  ylabel(ax, '\rho');
    title(ax, 'the amplifier', 'FontSize', opts.subtitlefontsize);

    % -- 3  HEADLINE: no xref appears anywhere in this panel ---------------
    rB = local_cat(B.allrho);
    ax = nexttile(tl);  hold(ax, 'on');
    local_panel(ax, { ...
        {rB, local_cat(B.allres),      'o', 1.2, 5, 'woodbury residual'}, ...
        {rB, local_cat(B.allbwd),      'd', 1.0, 4, 'woodbury backward err'}, ...
        {rB, local_cat(B.allfresh),    's', 1.0, 4, 'fresh K_n\b residual'}, ...
        {rB, eps*local_cat(B.allcsub), '^', 1.0, 4, 'cancel_{sub}\cdot\epsilon'}, ...
        {rB, local_cat(B.allfloor),    '.', 1.0, 8, 'rounding floor'}}, opts);
    xlabel(ax, '\rho');  ylabel(ax, 'relative size');
    title(ax, 'reference-free: residual vs \rho', 'FontSize', opts.subtitlefontsize);

    % -- 4  forward error, with the uncertainty that qualifies it ----------
    fB  = local_cat(B.allfwd);  uB = local_cat(B.allunc);
    rep = fB > 10*uB;
    ax = nexttile(tl);  hold(ax, 'on');
    local_panel(ax, { ...
        {local_cat(E.allrho), local_cat(E.allfwd), '^', 1.4, 6, ...
         'E exact (no reference)'}, ...
        {rB(rep),  fB(rep),  'o', 1.2, 5, 'B reportable'}, ...
        {rB(~rep), fB(~rep), 'o', 0.8, 5, 'B reference-limited', [0.6 0.6 0.6]}, ...
        {rB, uB, '.', 1.0, 9, 'reference uncertainty'}}, opts);
    xlabel(ax, '\rho');  ylabel(ax, 'relative forward error');
    title(ax, 'forward error vs what measured it', 'FontSize', opts.subtitlefontsize);

    save_woodbury_figure(fh, outFile, opts);
end

%==========================================================================
function local_panel(ax, specs, opts)
%LOCAL_PANEL  Draw a panel's series and legend, keeping label and line paired.
%
%   SPECS is a cell of {x, y, marker, linewidth, markersize, label[, color]}.
%   Series with no plottable point are skipped AND their label is dropped with
%   them.  Doing this by index -- plotting everything and then passing
%   labels(1:numel(h)) -- silently shifts every label after an empty series onto
%   the wrong line, and a mislabelled figure is worse than a missing one.
    h = gobjects(1, 0);
    lab = {};
    for k = 1:numel(specs)
        s = specs{k};
        [xx, yy] = local_xy(s{1}, s{2});
        if isempty(xx), continue; end
        args = {'LineWidth', s{4}, 'MarkerSize', s{5}};
        if numel(s) >= 7, args = [args, {'Color', s{7}}]; end %#ok<AGROW>
        h(end+1)   = plot(ax, xx, yy, s{3}, args{:}); %#ok<AGROW>
        lab{end+1} = s{6}; %#ok<AGROW>
    end
    set(ax, 'XScale', 'log', 'YScale', 'log');
    grid(ax, 'on');
    if ~isempty(h)
        legend(ax, h, lab, 'Location', 'northwest', ...
               'FontSize', opts.legendfontsize);
    end
end

%==========================================================================
function out = local_tern(c, a, b)
    if c, out = a; else, out = b; end
end
