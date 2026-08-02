% RUN_EIGENSPACE_MOTION  What exactly does the frozen coarse space miss, and
% does the rank-2nC block supply it?
%
% The coordinates are held FIXED at step 1 throughout, so
%
%     Ahat_n = Ahat_ref + Uhat B Uhat',    Uhat = C^-1 [dC_n, Sel],
%
% is a genuine symmetric rank-<=2nC perturbation and the containment theory
% applies exactly.  (run_drift_factorial covers the coordinate effect; mixing
% the two here would make the perturbation unstructured and the question
% meaningless.)
%
% Measured at each step, against the TRUE smallest-|lambda| eigenspace of Ahat_n:
%   V_ref              the frozen basis                    (the production space)
%   [V_ref, W_raw]     + Uhat itself, no solve             (residual space)
%   [V_ref, W_invref]  + Ahat^-1 Uhat via a frozen factorization  (the proposal)
%   [V_ref, random]    + the same COLUMN COUNT, random     (the control that
%                      separates "right directions" from "more directions")
%
% Also verified: eigenvalue interlacing under a rank-r symmetric perturbation
% displaces each ordered eigenvalue by at most r positions.  With the full dense
% spectrum available on the coarse twin this is checked exactly, which bounds how
% much of a k-dimensional window can turn over in one step.
%
% Fast default ~4 min (dense eig, full spectrum); FULL uses eigs at benchmark
% scale and skips the interlacing check.
%
% See also: run_drift_factorial, lowrank_update_basis, seq_dCblk.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
addpath(fullfile(fileparts(thisFileDir), 'kernel'));
add_recycle_paths();
rng(1);

outDir = fullfile(thisFileDir, 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end

FULL = evalin('base', 'exist(''FULL'',''var'') && logical(FULL)');
if FULL
    H0 = 0.05;  KBASE = 500;  STEPS = [2 3 5 10 20 40];  DENSE = false;
else
    H0 = 0.1;   KBASE = 100;  STEPS = [2 3 5 10];        DENSE = true;
end
CASE = 'bar_rotating';
VARIANTS = {'frozen', 'raw', 'invref', 'random'};

COL = struct('frozen', [0.50 0.35 0.65], 'raw', [0.70 0.55 0.40], ...
             'invref', [0.20 0.45 0.70], 'random', [0.85 0.40 0.32], ...
             'ref', [0.35 0.35 0.35]);

fprintf('=== run_eigenspace_motion (%s, h0=%g, k=%d, dense=%d) ===\n', ...
        CASE, H0, KBASE, DENSE);

S = build_stokes_sequence(struct('case_name', CASE, 'h0', H0, ...
                                 'nsteps', max(STEPS), 'quiet', true));
n = S.n;  m = S.nC;
K1 = seq_K(S, 1);
P1 = src.precond.make_ildl_precond(K1, struct('mode', 'nofill'));
C1 = ildl_coordinate_map(P1);
M1 = C1 * C1';  M1 = (M1 + M1') / 2;

[Uref, ~] = eigs(K1, M1, KBASE, 'smallestabs', 'Tolerance', 1e-11, ...
                 'MaxIterations', 3000);
Vref = transport_V(Uref, P1, C1);
fprintf('  n=%d  nC=%d  2nC=%d  k=%d\n', n, m, 2*m, KBASE);

if DENSE
    Aref_d  = split_dense(C1, K1);
    lam_ref = sort(eig(Aref_d), 'ascend');
    condC   = condest(C1);
    fprintf('  dense Ahat_ref: ||A||_2=%.3e, cond(C)=%.2e\n', norm(Aref_d, 2), condC);
end

rows = struct('case_name', {}, 'step', {}, 'variant', {}, 'ncols_added', {}, ...
    'coarse_dim', {}, 'capture_err2', {}, 'capture_errfro', {}, ...
    'n_ang_below_1pct', {}, 'n_ang_below_0p1pct', {}, 'dK_relnorm', {}, ...
    'interlace_violations', {}, 'interlace_maxviol', {}, 'interlace_rank', {}, ...
    'measured_rank_dAhat', {});
sines = struct();   % sorted sin(theta) per (step, variant) for the money figure
ctx = [];

for st = STEPS
    Kn = seq_K(S, st);

    % --- ground truth: the smallest-|lambda| eigenspace of Ahat_n -------
    if DENSE
        An      = split_dense(C1, Kn);
        [X, Dn] = eig(An);
        dn      = real(diag(Dn));
        [~, ord] = sort(abs(dn), 'ascend');
        Vtrue   = orth_trunc(X(:, ord(1:KBASE)));
        lam_n   = sort(dn, 'ascend');
        [nviol, maxviol] = interlacing_check(lam_ref, lam_n, 2*m);
        % Directly measure the rank of the operator change: this is the claim
        % the whole "missing component is 2nC dimensional" argument rests on,
        % so it is measured rather than assumed.  If it holds but interlacing
        % does not, the discrepancy is conditioning in the dense split, not
        % a failure of the theory.
        dAd    = An - Aref_d;
        sdA    = svd(dAd);
        rk_dA  = sum(sdA > 1e-8 * sdA(1));
    else
        nviol = NaN;  maxviol = NaN;  rk_dA = NaN;
        [Un, ~] = eigs(Kn, M1, KBASE, 'smallestabs', 'Tolerance', 1e-11, ...
                       'MaxIterations', 3000);
        Vtrue   = transport_V(Un, P1, C1);
    end

    % --- candidate augmentations (coordinates frozen at step 1) ---------
    [W_raw, ~, ctx] = lowrank_update_basis(S, st, P1, ctx, ...
                          struct('mode', 'raw',    'ref', 1, 'Cn', C1));
    [W_inv, ~, ctx] = lowrank_update_basis(S, st, P1, ctx, ...
                          struct('mode', 'invref', 'ref', 1, 'Cn', C1));
    W_rnd = randn(n, size(W_inv, 2));

    cand = struct('frozen', zeros(n, 0), 'raw', W_raw, 'invref', W_inv, ...
                  'random', W_rnd);
    dKrel = norm(full(seq_dCblk(S, st, 1)), 'fro') * 2 / norm(K1, 'fro');

    for v = 1:numel(VARIANTS)
        vn = VARIANTS{v};
        Vc = augment_recycle_V(Vref, cand.(vn));
        info = subspace_capture_directed(Vtrue, Vc, [], ...
                   struct('true_is_orth', true, 'comp_is_orth', true));
        rows(end+1) = struct('case_name', CASE, 'step', st, 'variant', vn, ...
            'ncols_added', size(Vc, 2) - size(Vref, 2), ...
            'coarse_dim', size(Vc, 2), ...
            'capture_err2', info.eigspace_err_2, ...
            'capture_errfro', info.eigspace_err_fro, ...
            'n_ang_below_1pct', info.n_angle_below_1pct, ...
            'n_ang_below_0p1pct', info.n_angle_below_0p1pct, ...
            'dK_relnorm', dKrel, 'interlace_violations', nviol, ...
            'interlace_maxviol', maxviol, 'interlace_rank', 2*m, ...
            'measured_rank_dAhat', rk_dA); %#ok<SAGROW>
        sines.(sprintf('s%d_%s', st, vn)) = info.sin_angles_directed;
    end

    fprintf('  step %2d  |dK|/|K|=%.3f  err_fro: frozen %.3f  raw %.3f  invref %.3f  random %.3f', ...
            st, dKrel, ...
            rows(end-3).capture_errfro, rows(end-2).capture_errfro, ...
            rows(end-1).capture_errfro, rows(end).capture_errfro);
    if isfinite(nviol)
        fprintf('  | rank(dAhat)=%d/%d, interlace viol %d (max %.1e)', ...
                rk_dA, 2*m, nviol, maxviol);
    end
    fprintf('\n');
end

T = struct2table(rows);
writetable(T, fullfile(outDir, 'eigenspace_motion.csv'));

%% ===== verdict ==========================================================
gm = @(v, f) mean(T.(f)(strcmp(T.variant, v)));
fprintf('\n==================================================================\n');
fprintf('  Capture of the true step-n eigenspace (k=%d, coordinates FROZEN)\n', KBASE);
fprintf('  mean over steps %s\n\n', mat2str(STEPS));
fprintf('  %-10s %8s %12s %12s %14s\n', ...
        'variant', '+cols', 'err_2', 'err_fro', 'dirs <1%');
for v = 1:numel(VARIANTS)
    vn = VARIANTS{v};
    fprintf('  %-10s %8.0f %12.4f %12.4f %14.0f\n', vn, ...
            gm(vn, 'ncols_added'), gm(vn, 'capture_err2'), ...
            gm(vn, 'capture_errfro'), gm(vn, 'n_ang_below_1pct'));
end
gain_inv = gm('frozen', 'capture_errfro') / gm('invref', 'capture_errfro');
gain_rnd = gm('frozen', 'capture_errfro') / gm('random', 'capture_errfro');
fprintf('\n  rank-2nC block improves err_fro by %.2fx; the same number of RANDOM\n', gain_inv);
fprintf('  columns improves it by %.2fx.\n', gain_rnd);
if gain_inv > 1.5 * gain_rnd
    fprintf('  => the block supplies specific missing directions, not just width.\n');
else
    fprintf('  => the gain is mostly extra width, not targeted directions.\n');
end
if DENSE
    nv = max(T.interlace_violations);
    fprintf('\n  measured rank of Ahat_n - Ahat_ref: %d (predicted 2nC = %d)\n', ...
            max(T.measured_rank_dAhat), 2*m);
    fprintf('  rank-%d interlacing on the spectrum interior: %d violations ', 2*m, nv);
    if nv == 0
        fprintf('(max %.1e).\n', max(T.interlace_maxviol));
        fprintf('  Bound holds, so no eigenvalue moves more than %d places in the\n', 2*m);
        fprintf('  ordering: at most %d of the %d window directions can turn over\n', ...
                min(2*m, KBASE), KBASE);
        fprintf('  per step.  That is the rigorous ceiling on how much of the coarse\n');
        fprintf('  space can go stale from the operator change alone.\n');
    else
        fprintf('(max %.2e).\n  BOUND VIOLATED — investigate.\n', max(T.interlace_maxviol));
    end
end
fprintf('==================================================================\n');

%% ===== money figure: the missing directions =============================
st_show = STEPS(min(2, numel(STEPS)));
fig = figure('Visible', 'off', 'Position', [100 100 1120 460]);
tl = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
for v = 1:numel(VARIANTS)
    vn = VARIANTS{v};
    s  = sines.(sprintf('s%d_%s', st_show, vn));
    plot(ax, 1:numel(s), max(s, 1e-17), '-', 'Color', COL.(vn), 'LineWidth', 1.8);
end
set(ax, 'YScale', 'log');
xlabel(ax, 'direction index (ascending)');
ylabel(ax, 'sin \theta_i  (0 = captured, 1 = missed)');
title(ax, sprintf('Directions of the true eigenspace missed, step %d', st_show));
legend(ax, {'frozen V', '+raw Uhat', '+rank-2nC block', '+random cols'}, ...
       'Location', 'southeast');

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
for v = 1:numel(VARIANTS)
    vn = VARIANTS{v};
    sel = strcmp(T.variant, vn);
    plot(ax, T.step(sel), T.capture_errfro(sel), '-o', 'Color', COL.(vn), ...
         'LineWidth', 1.8, 'MarkerFaceColor', COL.(vn), 'MarkerSize', 5);
end
xlabel(ax, 'time step');  ylabel(ax, '||(I-P)V_{true}||_F / \surdk');
title(ax, 'Average missed energy vs step');
legend(ax, {'frozen V', '+raw Uhat', '+rank-2nC block', '+random cols'}, ...
       'Location', 'east');

sgtitle(fig, sprintf(['What the frozen coarse space misses — %s, n=%d, k=%d, ' ...
                      '2nC=%d (coordinates frozen)'], CASE, n, KBASE, 2*m));
exportgraphics(fig, fullfile(outDir, 'eigenspace_motion.png'), 'Resolution', 180);
close(fig);

fprintf('[saved] %s\n', fullfile(outDir, 'eigenspace_motion.csv'));
fprintf('[saved] %s\n', fullfile(outDir, 'eigenspace_motion.png'));

%==========================================================================
function A = split_dense(C, K)
%SPLIT_DENSE  Dense Ahat = C^-1 K C^-T, using Ahat = C^-1 (C^-1 K)'.
    A = full(C \ ((C \ K)'));
    A = (A + A') / 2;
end

function [nviol, maxviol] = interlacing_check(lam_ref, lam_n, r)
%INTERLACING_CHECK  Verify the rank-r symmetric interlacing bound by VALUE.
%   For a symmetric perturbation of rank r (ascending order),
%       lam_{i-r}(ref) <= lam_i(n) <= lam_{i+r}(ref).
%
%   Checked on values, not on index positions.  A position-based measure
%   ("how many places did eigenvalue i move?") is meaningless here: the split
%   operator has a huge tight cluster near +-1 (the ILDL is nearly exact on most
%   of the spectrum), so sum(lam_ref <= x) jumps by the whole cluster at once
%   and reports enormous displacements that are tie-breaking artifacts rather
%   than real motion.
%
%   Only indices with r < i <= N-r are tested.  Within r of either end the
%   theorem gives no two-sided bound: if E has p positive eigenvalues, the top p
%   eigenvalues of A+E are unbounded above (and symmetrically at the bottom), so
%   clamping the index there would invent a bound and report false violations.
%
%   Returns the number of violations in the interior (should be 0) and the
%   largest violation relative to the spectral radius.
    N   = numel(lam_n);
    scl = max(abs(lam_ref([1 end])));
    idx = (r+1 : N-r)';
    if isempty(idx), nviol = 0;  maxviol = 0;  return; end
    lo  = lam_ref(idx - r);
    hi  = lam_ref(idx + r);
    gap = max(max(lo - lam_n(idx), 0), max(lam_n(idx) - hi, 0));
    nviol   = sum(gap > 1e-8 * scl);
    maxviol = max(gap) / scl;
end
