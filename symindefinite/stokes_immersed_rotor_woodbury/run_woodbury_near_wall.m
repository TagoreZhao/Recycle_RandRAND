%RUN_WOODBURY_NEAR_WALL  Reach rho >> 1 physically: the close-clearance rotor.
%
%   Run:  run_woodbury_near_wall
%         SMOKE = true; run_woodbury_near_wall     % h0 = 0.1, ~8 steps
%
%   The Woodbury identity has two independent cancellation sites (README 4.3).
%   Site 1, the capacitance, is already excited by the disk point spacing:
%   cond(Cap) reaches 4.8e10.  Site 2 -- the final subtraction x = z - Y0 w, whose
%   amplifier is
%
%       rho = ||K_1^{-1} b|| / ||K_n^{-1} b||
%
%   -- has never been excited on the physical sequence: rho stays in [0.50, 1.01]
%   and README 4.4 item 4 calls it "the one that would break first".  This script
%   excites it, with geometry rather than with an adversarial right-hand side.
%
%   THE MECHANISM.  rho is large when the REFERENCE solve amplifies b much more
%   than the target solve does -- i.e. when K_1 is far closer to singular than
%   K_n.  Two knobs produce exactly that:
%
%     Lb_frac -> 0.499   the bar tips sweep within 0.1% of the channel walls.
%                        Their interpolation weights land mostly on Dirichlet
%                        wall nodes, which apply_dirichlet_sym eliminates, so
%                        those constraint rows go near-zero and K is near
%                        singular AT THOSE ANGLES ONLY.
%     theta0             an initial phase offset placing that configuration on
%                        STEP 1 -- the step woodbury_context_init hard-freezes as
%                        the reference.  Targets stay far from the wall.
%
%   THE OFFSET IS NOT pi/2.  build_stokes_sequence evaluates step n at t = n*dt,
%   so step 1 is t = dt, not t = 0.  The offset must be
%
%       theta0 = pi/2 - omega*dt = pi/2 - 2*pi*nrev/Tstep      (dt cancels)
%
%   A naive pi/2 puts step 1 at 101.8 deg, a gap of 0.39*h0, and moves the true
%   minimum to step 15 -- where a near-singular TARGET depresses rho instead of
%   raising it and re-excites site 1, confounding the whole experiment.
%   test_motion_params T6 pins this.
%
%   ARMS
%     treatment  theta0 = pi/2 - 2*pi*nrev/Tstep : the near-wall bar IS the frozen reference
%     control    theta0 = 0                      : same geometry, benign reference
%     fresh      Kn \ b                          : the operator is fine; only the evaluation is not
%
%   The control is what makes the treatment mean something: the bar still passes
%   the wall at some step in both arms, so a difference between them isolates the
%   REFERENCE as the cause rather than the geometry.
%
%   WHAT THIS BUYS BEYOND rho.  README 4.4's caveat records that cond(Cap) and
%   cancel_cap move together on the shipped sequence and cannot be separated.
%   Here they part company: a near-wall REFERENCE with far-from-wall TARGETS
%   drives cancel_sub while leaving cancel_cap at its structural ~1e2 (C = B is an
%   involution).  That is the first separation of the two sites on the real
%   operator.
%
%   It writes one figure and no CSVs.
%
%   See also: run_woodbury_stability, woodbury_solve, define_motion_list.

% NOTE no `clear` -- that would wipe a SMOKE flag set from the base workspace,
% which is how run_woodbury_benchmark's SMOKE_TEST is driven too.
clc;
paths = add_woodbury_paths();
assert_woodbury_helpers();
rng(0);

if ~exist('SMOKE', 'var') || isempty(SMOKE), SMOKE = false; end

params = make_woodbury_params();

CASE   = 'bar_rotating';
NREV   = 2;      % must match make_bar_rotating's literal; the gap assert catches drift
GAPMIN = 3;      % probe steps must sit > GAPMIN*h0 from the wall, so the TARGETS
                 % are well posed and only the REFERENCE is degenerate
NPROBE = 6;
% Half-length fractions.  Wall gap = (0.5 - Lb_frac)*H, so 0.499 -> 0.1% of the
% channel height.  0.35 is the shipped value and anchors the sweep at rho ~ 1.
% The cliff is steep: at h0 = 0.1 everything from 0.45 up already returns
% singularReference (1/condest(D) ~ 1e-17), so a coarse grid measures nothing.
% Sample finely from the shipped value upward and let the infeasible points
% report themselves -- where the cliff sits IS one of the results.
LB     = [0.35 0.375 0.40 0.42 0.44 0.46 0.48 0.499];
CTRL   = [0.35 0.42];         % control at a feasible pair, not at a dead point

if SMOKE
    H0 = 0.1;  NSTEPS = 8;  NPROBE = 3;
else
    H0 = params.h0;  NSTEPS = 20;   % ~one near-wall recurrence (period ~15 steps)
end

THETA0 = pi/2 - 2*pi*NREV/params.Tstep;

fprintf('=== woodbury near-wall: %s, h0 = %g, %d steps%s ===\n', ...
        CASE, H0, NSTEPS, ternary(SMOKE, '  [SMOKE]', ''));
fprintf('theta0 = %.6f rad (%.2f deg) = pi/2 - 2*pi*%d/%d\n\n', ...
        THETA0, rad2deg(THETA0), NREV, params.Tstep);

nL = numel(LB);
T  = local_alloc(nL);          % treatment
C  = local_alloc(nL);          % control (only CTRL entries filled)

for i = 1:nL
    T = local_run(T, i, CASE, H0, NSTEPS, LB(i), THETA0, params, GAPMIN, NPROBE);
    if ismember(LB(i), CTRL)
        C = local_run(C, i, CASE, H0, NSTEPS, LB(i), 0, params, GAPMIN, NPROBE);
    end
end

% ---- tables --------------------------------------------------------------
fprintf('\n--- treatment: the near-wall bar IS the frozen reference ---\n');
local_table(LB, T, H0);

fprintf('\n--- control: same geometry, theta0 = 0 (benign reference) ---\n');
local_table(LB, C, H0);

% ---- what the numbers say, generated not asserted -------------------------
ok  = isfinite(T.rho);
okc = isfinite(C.rho);
nok = nnz(ok);
fprintf('\n%d of %d treatment points are feasible.\n', nok, numel(LB));

if nok >= 1
    [~, imax] = max(T.fwd .* ok);
    fprintf(['Worst feasible point: Lb_frac = %.4g, step-1 gap %.4g (%.3g*h0), ' ...
             'rho = %.3g,\n  woodbury %.3e vs fresh Kn\\b %.3e -- a factor %.3g, ' ...
             'and it is all Woodbury''s.\n'], ...
            LB(imax), T.gap1(imax), T.gap1(imax)/H0, T.rho(imax), ...
            T.fwd(imax), T.fresh_res(imax), T.fwd(imax)/max(T.fresh_res(imax), eps));

    if any(okc)
        fprintf(['Control (theta0 = 0, same geometry): rho <= %.3g, error <= %.3e. ' ...
                 'The amplifier is\n  the REFERENCE, not the bar length -- the ' ...
                 'geometry is identical in both arms.\n'], ...
                max(C.rho(okc)), max(C.fwd(okc)));
    end

    % Which scale actually governs?  Report it, do not assume it.  These are
    % different mechanisms with the same symptom and the study must not
    % conflate them.
    rs = T.fwd(imax) / max(T.csub(imax)*eps, realmin);
    rk = T.fwd(imax) / max(T.k1(imax)*eps,  realmin);
    fprintf(['\nWhich scale governs: err/(cancel_sub*eps) = %.2e, ' ...
             'err/(cond(K_1)*eps) = %.2e.\n'], rs, rk);
    if rs > 1e2 && rk < 1e2
        fprintf(['  => the error follows cond(K_1)*eps, NOT the final ' ...
                 'subtraction.  This is a\n     DEGENERATE REFERENCE (README ' ...
                 '4.1''s "regime to fear"), not cancellation site 2.\n     ' ...
                 'cancel_sub = %.3g is far too small to explain it.\n'], T.csub(imax));
    elseif rs < 1e2
        fprintf(['  => the error is consistent with cancel_sub*eps: cancellation ' ...
                 'site 2 is\n     excited, which the shipped sequence never ' ...
                 'reaches.\n']);
    else
        fprintf('  => neither scale brackets it; report both and do not claim one.\n');
    end

    fprintf(['\ncancel_cap stays %.3g at the worst point, against cancel_sub %.3g: ' ...
             'whatever\n  drives this, it is not the capacitance site.\n'], ...
            T.ccap(imax), T.csub(imax));
end

bad = find(~cellfun(@isempty, T.err_id));
for k = bad(:)'
    fprintf('  Lb_frac = %.4g INFEASIBLE: %s\n', LB(k), T.err_id{k});
end
if ~isempty(bad)
    fprintf(['The largest feasible Lb_frac is %.4g.  Per the study''s design this ' ...
             'is reported\nas a measured limit, not worked around by widening the ' ...
             'bar point spacing --\nthat would lower nC and break comparability ' ...
             'with the benchmark''s cost table.\n'], max(LB(isfinite(T.rho))));
end

% ---- figure ---------------------------------------------------------------
outFile = fullfile(paths.outDir, sprintf('near_wall_rho_h%s_n%d.png', ...
                   strrep(num2str(H0), '.', 'p'), NSTEPS));
local_figure(T, C, H0, CASE, outFile);

%==========================================================================
function F = local_alloc(n)
%LOCAL_ALLOC  One row per sweep point, one field per reported quantity.
    z = nan(n, 1);
    F = struct('rho', z, 'csub', z, 'ccap', z, 'capcond', z, 'fwd', z, ...
               'res', z, 'bwd', z, 'fresh_res', z, 'k1', z, 'nC', z, ...
               'margin', z, 'gap1', z, 'nwarn', z, 'probes', {cell(n,1)}, ...
               'err_id', {cell(n,1)});
end

%==========================================================================
function F = local_run(F, i, CASE, H0, NSTEPS, Lb_frac, theta0, params, GAPMIN, NPROBE)
%LOCAL_RUN  One sweep point: build, freeze, probe.  Never throws.
%   woodbury_context_init THROWS on a degenerate reference rather than returning,
%   and assert_coupling_feasible can refuse the geometry outright, so every point
%   is wrapped and the identifier reported.
    F.err_id{i} = '';
    % Silence ONLY the known per-step floods, so lastwarn stays meaningful for
    % anything unexpected.  warning('off','all') would also stop lastwarn being
    % set, making the nwarn column a permanent zero.
    ws = warning('off', 'woodbury_solve:singularCapacitance');
    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');
    lastwarn('');
    try
        S = build_stokes_sequence(struct('case_name', CASE, 'h0', H0, ...
                'dt', params.dt, 'Tstep', params.Tstep, 'nsteps', NSTEPS, ...
                'verify', false, 'use_cache', true, 'quiet', true, ...
                'motion_params', struct('Lb_frac', Lb_frac, 'theta0', theta0)));

        % --- wall gap per step, measured from the points and the mesh ------
        y1 = min(S.msh.p(:,2));  y2 = max(S.msh.p(:,2));
        gap = nan(S.nsteps, 1);
        for n = 1:S.nsteps
            Y = S.Xpts{n}(:,2);
            gap(n) = min(min(Y - y1), min(y2 - Y));
        end
        F.gap1(i) = gap(1);

        % Non-vacuity: the offset must actually have landed the intended
        % configuration on step 1.  Fires if nrev or the step/time mapping moves.
        if theta0 ~= 0
            want = (0.5 - Lb_frac) * (y2 - y1);
            assert(abs(gap(1) - want) < 1e-9, ...
                   ['step-1 gap %.4e disagrees with (0.5 - Lb_frac)*H = %.4e: ' ...
                    'the phase offset did not place the bar vertical on step 1'], ...
                   gap(1), want);
        end

        ctx = woodbury_context_init(S);
        K1  = seq_K(S, ctx.ref);
        F.k1(i) = condest(K1);          % once per point: this is an LU
        F.nC(i) = S.nC;

        tch = inf;
        for n = 1:S.nsteps
            tch = min(tch, assert_coupling_feasible(S.Ccpl{n}, S.veldofs, n, CASE));
        end
        F.margin(i) = tch - S.nC;

        % --- probe steps generated from the measured gaps -----------------
        % Step 1 is excluded: dC = 0 there, so the "error" is two factorizations
        % of the same near-singular K_1 differing by cond(K_1)*eps against an
        % xref that is itself inaccurate -- not a Woodbury metric.
        cand = find(gap > GAPMIN * H0);
        cand = cand(cand > 1);
        if isempty(cand)
            error('run_woodbury_near_wall:noCleanProbe', ...
                  'no step sits more than %g*h0 from the wall', GAPMIN);
        end
        sel = unique(round(linspace(1, numel(cand), min(NPROBE, numel(cand)))));
        probe = cand(sel);
        F.probes{i} = probe(:)';

        acc = local_alloc(numel(probe));
        for j = 1:numel(probe)
            n  = probe(j);
            b  = S.b{n};
            Kn = seq_K(S, n);
            [xw, info] = woodbury_solve(ctx, S, n, b);
            xr = S.xref{n};
            [fwd, res, bwd] = local_errs(xw, xr, Kn, b);
            xb = Kn \ b;                       % the fresh direct arm
            acc.rho(j)     = info.rho;
            acc.csub(j)    = info.cancel_sub;
            acc.ccap(j)    = info.cancel_cap;
            acc.capcond(j) = info.cap_cond;
            acc.fwd(j)     = fwd;
            acc.res(j)     = res;
            acc.bwd(j)     = bwd;
            acc.fresh_res(j) = norm(b - Kn*xb) / max(norm(b), eps);
        end
        % Report the WORST probe step: this is a stability study, so the
        % headline number is the one that would bite, not the average.
        F.rho(i)       = max(acc.rho);
        F.csub(i)      = max(acc.csub);
        F.ccap(i)      = max(acc.ccap);
        F.capcond(i)   = max(acc.capcond);
        F.fwd(i)       = max(acc.fwd);
        F.res(i)       = max(acc.res);
        F.bwd(i)       = max(acc.bwd);
        F.fresh_res(i) = max(acc.fresh_res);
    catch ME
        F.err_id{i} = ME.identifier;
        if isempty(F.err_id{i}), F.err_id{i} = '(unidentified)'; end
        fprintf(2, '  Lb_frac = %.4g theta0 = %.4f -> %s\n', ...
                Lb_frac, theta0, ME.message);
    end
    [~, wid] = lastwarn;
    F.nwarn(i) = double(~isempty(wid));
    warning(ws);
end

%==========================================================================
function [fwd, res, bwd] = local_errs(x, xref, Kn, b)
    r   = b - Kn * x;
    bn  = max(norm(b), eps);
    fwd = norm(x - xref) / max(norm(xref), eps);
    res = norm(r) / bn;
    bwd = norm(r) / (norm(Kn, 'fro') * norm(x) + bn);
end

%==========================================================================
function local_table(LB, F, H0)
    % The last two columns are the point of the table: they say WHICH scale the
    % observed error follows.  cancel_sub*eps is site 2 (the final subtraction);
    % cond(K_1)*eps is a degenerate REFERENCE, which is a different mechanism
    % with the same symptom.  Printing both stops one being reported as the other.
    fprintf('%9s %8s %6s %7s %10s %10s %10s %11s %11s %10s %10s %10s\n', 'Lb_frac', ...
            'gap/h0', 'nC', 'margin', 'rho', 'cancl_sub', 'cancl_cap', ...
            'wood_err', 'fresh_res', 'cond(K_1)', 'cSub*eps', 'kK1*eps');
    for i = 1:numel(LB)
        if ~isempty(F.err_id{i})
            fprintf('%9.4g %8s   %s\n', LB(i), '-', F.err_id{i});
        elseif isnan(F.rho(i))
            fprintf('%9.4g %8s   (not run)\n', LB(i), '-');
        else
            fprintf(['%9.4g %8.3f %6d %7d %10.3e %10.3e %10.3e %11.3e %11.3e ' ...
                     '%10.3e %10.3e %10.3e\n'], ...
                    LB(i), F.gap1(i)/H0, F.nC(i), F.margin(i), F.rho(i), ...
                    F.csub(i), F.ccap(i), F.fwd(i), F.fresh_res(i), F.k1(i), ...
                    F.csub(i)*eps, F.k1(i)*eps);
        end
    end
end

%==========================================================================
function local_figure(T, C, H0, CASE, outFile)
    opts = woodbury_fig_defaults();
    fh = figure('Visible', 'off', 'Units', 'inches', ...
                'Position', [1 1 opts.multi_width opts.multi_height + 1]);
    tl = tiledlayout(fh, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('%s: close-clearance rotor, h_0 = %g', ...
          strrep(CASE, '_', '\_'), H0), ...
          'FontSize', opts.titlefontsize, 'Interpreter', 'tex');
    fl = @(v) max(v, 1e-18);              % a log axis cannot show an exact zero
    g  = T.gap1 / H0;

    ax = nexttile(tl);
    semilogy(ax, g, fl(T.rho), '-o', 'LineWidth', 1.6, 'MarkerSize', 4); hold(ax,'on');
    semilogy(ax, g, fl(C.rho), '-s', 'LineWidth', 1.6, 'MarkerSize', 4);
    set(ax, 'XDir', 'reverse');
    xlabel(ax, 'step-1 wall gap / h_0');  ylabel(ax, '\rho');
    title(ax, 'the amplifier', 'FontSize', opts.subtitlefontsize);
    legend(ax, {'reference at the wall', '\theta_0 = 0 control'}, ...
           'Location', 'northwest', 'FontSize', opts.legendfontsize);

    ax = nexttile(tl);
    loglog(ax, fl(T.rho), fl(T.fwd), '-o', 'LineWidth', 1.6, 'MarkerSize', 4); hold(ax,'on');
    loglog(ax, fl(T.rho), fl(T.csub*eps), '--^', 'LineWidth', 1.2, 'MarkerSize', 4);
    loglog(ax, fl(T.rho), fl(T.fresh_res), '-s', 'LineWidth', 1.6, 'MarkerSize', 4);
    xlabel(ax, '\rho');  ylabel(ax, 'relative error');
    title(ax, 'error follows \rho; backslash does not', 'FontSize', opts.subtitlefontsize);
    legend(ax, {'woodbury', 'cancel_{sub}\cdot\epsilon', 'fresh K_n\\b'}, ...
           'Location', 'northwest', 'FontSize', opts.legendfontsize);

    ax = nexttile(tl);
    semilogy(ax, g, fl(T.csub), '-o', 'LineWidth', 1.6, 'MarkerSize', 4); hold(ax,'on');
    semilogy(ax, g, fl(T.ccap), '-s', 'LineWidth', 1.6, 'MarkerSize', 4);
    semilogy(ax, g, fl(T.capcond), '--^', 'LineWidth', 1.2, 'MarkerSize', 4);
    set(ax, 'XDir', 'reverse');
    xlabel(ax, 'step-1 wall gap / h_0');  ylabel(ax, 'factor');
    title(ax, 'the two sites separate', 'FontSize', opts.subtitlefontsize);
    legend(ax, {'cancel_{sub}', 'cancel_{cap}', 'cond(Cap)'}, ...
           'Location', 'northwest', 'FontSize', opts.legendfontsize);

    ax = nexttile(tl);
    semilogy(ax, g, fl(T.k1), '-o', 'LineWidth', 1.6, 'MarkerSize', 4); hold(ax,'on');
    semilogy(ax, g, fl(T.fwd), '-s', 'LineWidth', 1.6, 'MarkerSize', 4);
    set(ax, 'XDir', 'reverse');
    xlabel(ax, 'step-1 wall gap / h_0');  ylabel(ax, 'value');
    title(ax, 'reference conditioning vs error', 'FontSize', opts.subtitlefontsize);
    legend(ax, {'cond(K_1)', 'woodbury err'}, 'Location', 'northwest', ...
           'FontSize', opts.legendfontsize);

    save_woodbury_figure(fh, outFile, opts);
end

%==========================================================================
function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
