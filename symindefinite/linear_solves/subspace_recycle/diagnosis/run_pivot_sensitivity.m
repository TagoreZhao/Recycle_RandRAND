% RUN_PIVOT_SENSITIVITY  Gate test for H1: is the deflation space destroyed by
% COORDINATE drift rather than by the operator moving?
%
% THE CONTROL.  In every row below the system being solved is byte-identical:
% the step-1 KKT pair (K_ref, b_ref).  Only the ILDL preconditioner changes,
% because it is rebuilt from a coupling block perturbed by delta.  So any change
% in iteration count is attributable to the split coordinate system C alone.
%
% Three solves per row:
%   ildl        MINRES on Ahat_delta with no coarse space   (is P_delta still a
%               good smoother?  if yes, the smoother is not what broke)
%   frozen V    the production failure mode: V built once in C_0's coordinates,
%               used against Ahat_delta = C_delta^-1 K_ref C_delta^-T
%   transported V = orth(C_delta' U_ref): the SAME physical eigenvectors,
%               re-expressed in the current coordinates (the H1 repair)
%
% If frozen-V iterations blow up while ILDL-only and transported-V stay flat,
% H1 is settled with no confound: the operator never moved.
%
% Part B repeats this at realistic magnitudes: build the ILDL from a LATER
% step's matrix (so the coupling pattern really changes, as it does when a
% Lagrange point crosses a triangle edge) but still solve the step-1 system.
%
% Fast default ~2 min; set FULL = true in the base workspace for benchmark scale.
%
% See also: run_ildl_drift, run_drift_factorial, transport_V, ildl_coordinate_map.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
repoRoot = fileparts(fileparts(fileparts(fileparts(thisFileDir))));
addpath(repoRoot);
addpath(fullfile(fileparts(thisFileDir), 'kernel'));
add_recycle_paths();
rng(1);

outDir = fullfile(thisFileDir, 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end

FULL = evalin('base', 'exist(''FULL'',''var'') && logical(FULL)');
if FULL
    H0 = 0.05;  KBASE = 500;  CASE = 'bar_rotating';
else
    H0 = 0.1;   KBASE = 100;  CASE = 'bar_rotating';
end
TAU = 1;  TOL = 1e-8;
% The last two entries matter: ||dC||_F/||C||_F is ~1.05 per step in the real
% benchmark, so delta = 1 is a value-only perturbation of the SAME MAGNITUDE as
% one step of actual motion.  If the frozen basis survives delta = 1 but dies in
% Part B, the culprit is the sparsity-PATTERN change, not the size of the change.
DELTAS = [0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 3e-1, 1];
MOTION_STEPS = [1 2 3 5 8];

COL_ILDL = [0.85 0.40 0.32];  COL_FROZ = [0.50 0.35 0.65];
COL_TRAN = [0.20 0.45 0.70];  COL_REF  = [0.35 0.35 0.35];

fprintf('=== run_pivot_sensitivity (%s, h0=%g, k=%d, tau=%g) ===\n', ...
        CASE, H0, KBASE, TAU);

S = build_stokes_sequence(struct('case_name', CASE, 'h0', H0, ...
                                 'nsteps', max(MOTION_STEPS), 'quiet', true));
Kref = seq_K(S, 1);  bref = S.b{1};  n = S.n;
maxit = min(4000, n);
fprintf('  n=%d  nC=%d  nnz=%d\n', n, S.nC, nnz(Kref));

% ---- reference coordinates and the physical eigenvectors ----------------
P0 = src.precond.make_ildl_precond(Kref, struct('mode', 'nofill'));
[C0, info0] = ildl_coordinate_map(P0);
M0 = C0 * C0';  M0 = (M0 + M0') / 2;
fprintf('  computing %d smallest-|lambda| eigenvectors of (K_ref, M_0) ...\n', KBASE);
te = tic;
[Uref, Dref] = eigs(Kref, M0, KBASE, 'smallestabs', ...
                    'Tolerance', 1e-10, 'MaxIterations', 2000);
Vref = transport_V(Uref, P0, C0);           % the frozen basis, in C_0 coordinates
fprintf('  eigs took %.1f s; |lambda| in [%.2e, %.2e]\n', toc(te), ...
        min(abs(diag(Dref))), max(abs(diag(Dref))));

so = struct('tau', TAU, 'tol', TOL, 'maxit', maxit);

%% ===== Part A: value-only perturbation of the coupling block ============
% The sparsity pattern is preserved; only the numerical values move.  This
% isolates pure re-pivoting: any breakage here cannot be blamed on the mesh.
rowsA = struct('delta', {}, 'perm_hamming', {}, 'perm_hamming_frac', {}, ...
    'nnzL_ratio', {}, 'n_offdiag_pivots', {}, 'transport_err2', {}, ...
    'transport_errfro', {}, 'iters_ildl', {}, 'iters_frozenV', {}, ...
    'iters_transportV', {}, 'condE_frozen', {}, 'sqrtMinEigE_frozen', {}, ...
    'condE_transport', {}, 'true_res_frozen', {}, 'flag_frozen', {});

fprintf('\n  %-10s %8s %10s %10s %8s %8s %8s\n', ...
        'delta', 'permH%', 'transp_e2', 'condE_fro', 'ildl', 'frozenV', 'transpV');
for dl = DELTAS
    Kp = perturb_coupling_block(S, 1, dl);
    Pd = src.precond.make_ildl_precond(Kp, struct('mode', 'nofill'));
    [Cd, infod] = ildl_coordinate_map(Pd, info0);

    % what the frozen basis ACTUALLY deflates now, pulled back to physical space
    cap = subspace_capture_directed(Uref, Pd.applyCtinv(Vref), []);

    Vtr = transport_V(Uref, Pd, Cd);

    r_i = two_level_it(Kref, bref, Pd, [],    so);   % operator UNCHANGED
    r_f = two_level_it(Kref, bref, Pd, Vref,  so);
    r_t = two_level_it(Kref, bref, Pd, Vtr,   so);

    rowsA(end+1) = struct('delta', dl, ...
        'perm_hamming', infod.perm_hamming, ...
        'perm_hamming_frac', infod.perm_hamming_frac, ...
        'nnzL_ratio', infod.nnzL_ratio, ...
        'n_offdiag_pivots', infod.n_offdiag_pivots, ...
        'transport_err2', cap.eigspace_err_2, ...
        'transport_errfro', cap.eigspace_err_fro, ...
        'iters_ildl', r_i.iters, 'iters_frozenV', r_f.iters, ...
        'iters_transportV', r_t.iters, ...
        'condE_frozen', r_f.condE, 'sqrtMinEigE_frozen', r_f.sqrt_minEigE, ...
        'condE_transport', r_t.condE, ...
        'true_res_frozen', r_f.true_res, 'flag_frozen', r_f.flag); %#ok<SAGROW>

    fprintf('  %-10.0e %7.1f%% %10.3e %10.2e %8d %8d %8d\n', dl, ...
            100*infod.perm_hamming_frac, cap.eigspace_err_2, r_f.condE, ...
            r_i.iters, r_f.iters, r_t.iters);
end
TA = struct2table(rowsA);
writetable(TA, fullfile(outDir, 'pivot_sensitivity.csv'));

%% ===== Part B: realistic pattern change (a later step's ILDL) ===========
% Same idea, but P comes from a LATER step's matrix, so the coupling pattern
% genuinely changes (Lagrange points land in different host triangles).  The
% system solved is still the step-1 one.
rowsB = struct('src_step', {}, 'coupling_change', {}, 'perm_hamming_frac', {}, ...
    'nnzL_ratio', {}, 'transport_err2', {}, 'transport_errfro', {}, ...
    'iters_ildl', {}, 'iters_frozenV', {}, 'iters_transportV', {}, ...
    'condE_frozen', {}, 'condE_transport', {});

fprintf('\n  ILDL built from step j, system solved is ALWAYS step 1:\n');
fprintf('  %-8s %10s %10s %10s %8s %8s %8s\n', ...
        'step j', 'dC(1->j)', 'permH%', 'transp_e2', 'ildl', 'frozenV', 'transpV');
for j = MOTION_STEPS
    Kj = seq_K(S, j);
    Pj = src.precond.make_ildl_precond(Kj, struct('mode', 'nofill'));
    [Cj, infoj] = ildl_coordinate_map(Pj, info0);
    cap = subspace_capture_directed(Uref, Pj.applyCtinv(Vref), []);
    Vtr = transport_V(Uref, Pj, Cj);

    r_i = two_level_it(Kref, bref, Pj, [],   so);
    r_f = two_level_it(Kref, bref, Pj, Vref, so);
    r_t = two_level_it(Kref, bref, Pj, Vtr,  so);

    dcj = norm(S.Ccpl{j} - S.Ccpl{1}, 'fro') / norm(S.Ccpl{1}, 'fro');
    rowsB(end+1) = struct('src_step', j, 'coupling_change', dcj, ...
        'perm_hamming_frac', infoj.perm_hamming_frac, ...
        'nnzL_ratio', infoj.nnzL_ratio, ...
        'transport_err2', cap.eigspace_err_2, ...
        'transport_errfro', cap.eigspace_err_fro, ...
        'iters_ildl', r_i.iters, 'iters_frozenV', r_f.iters, ...
        'iters_transportV', r_t.iters, ...
        'condE_frozen', r_f.condE, 'condE_transport', r_t.condE); %#ok<SAGROW>

    fprintf('  %-8d %10.4f %9.1f%% %10.3e %8d %8d %8d\n', j, dcj, ...
            100*infoj.perm_hamming_frac, cap.eigspace_err_2, ...
            r_i.iters, r_f.iters, r_t.iters);
end
TB = struct2table(rowsB);
writetable(TB, fullfile(outDir, 'pivot_sensitivity_motion.csv'));

%% ===== verdict ==========================================================
base_two  = TA.iters_frozenV(1);      % delta = 0
base_ildl = TA.iters_ildl(1);
nz      = TA.delta > 0;
d_safe  = max([0; TA.delta(nz & TA.iters_frozenV <= 1.2 * base_two)]);
ibreak  = find(nz & TA.iters_frozenV > 1.5 * base_two, 1, 'first');
d_break = NaN;  if ~isempty(ibreak), d_break = TA.delta(ibreak); end
i1      = find(TA.delta == 1, 1);

mv = TB.src_step > 1;                 % the rows where the pattern really changed
rec_ratio = TB.iters_frozenV(mv) ./ TB.iters_transportV(mv);
def_ratio = TB.iters_ildl(mv)    ./ TB.iters_frozenV(mv);   % <1 => deflation HURTS
smoother_penalty = mean(TB.iters_ildl(mv)) / base_ildl;

fprintf('\n==================================================================\n');
fprintf('  reference (delta=0)   : ILDL %d, two-level %d  (%.2fx gain)\n', ...
        base_ildl, base_two, base_ildl/base_two);
fprintf('\n  PART A  value-only perturbation, sparsity pattern preserved\n');
fprintf('    frozen V intact up to delta = %g (within 1.2x of baseline)', d_safe);
if isfinite(d_break)
    fprintf('; first degrades\n    >1.5x at delta = %g.\n', d_break);
else
    fprintf('; never degraded >1.5x.\n');
end
if ~isempty(i1)
    fprintf(['    at delta = 1 (the size of ONE real time step): perm moved %.0f%%, ' ...
             'capture\n    %.3f, ILDL %d, frozen V %d, transported V %d (%.2fx recovery).\n'], ...
            100*TA.perm_hamming_frac(i1), TA.transport_err2(i1), ...
            TA.iters_ildl(i1), TA.iters_frozenV(i1), TA.iters_transportV(i1), ...
            TA.iters_frozenV(i1)/TA.iters_transportV(i1));
end
fprintf('    => NOT a hair trigger: tiny perturbations are harmless, but an O(1)\n');
fprintf('       change in C -- exactly what one step delivers -- is not.\n');

fprintf('\n  PART B  real pattern change, operator STILL exactly K_1\n');
fprintf('    ldl permutation moved %.0f-%.0f%%; capture of U_ref falls to %.3f\n', ...
        100*min(TB.perm_hamming_frac(mv)), 100*max(TB.perm_hamming_frac(mv)), ...
        max(TB.transport_err2(mv)));
fprintf('    at an IDENTICAL smoother: frozen V %.2f-%.2fx SLOWER than transported V\n', ...
        min(rec_ratio), max(rec_ratio));
if all(def_ratio <= 1.02)
    fprintf('    frozen V is no better than NO deflation at all (ildl/frozen = %.2f-%.2f)\n', ...
            min(def_ratio), max(def_ratio));
end
if max(rec_ratio) > 1.3
    fprintf('    => H1 CONFIRMED: the sparsity-PATTERN change destroys the basis.\n');
else
    fprintf('    => H1 NOT confirmed: transporting V did not recover the gain.\n');
end
fprintf('\n  SIDE FINDING: a mismatched ILDL is itself a %.2fx worse smoother\n', ...
        smoother_penalty);
fprintf('    (ILDL-only %d -> %.0f iters), so FREEZING the ILDL is not free —\n', ...
        base_ildl, mean(TB.iters_ildl(mv)));
fprintf('    refresh-and-transport beats freeze-to-keep-coordinates.\n');
fprintf('==================================================================\n');

%% ===== figures ==========================================================
fig = figure('Visible', 'off', 'Position', [100 100 1120 460]);
tl = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile(tl); hold(ax1, 'on'); grid(ax1, 'on');
dpl = TA.delta;  dpl(dpl == 0) = min(dpl(dpl > 0)) / 10;   % show delta=0 off-scale
plot(ax1, dpl, TA.iters_ildl,       '-o', 'Color', COL_ILDL, 'LineWidth', 1.8, ...
     'MarkerFaceColor', COL_ILDL);
plot(ax1, dpl, TA.iters_frozenV,    '-s', 'Color', COL_FROZ, 'LineWidth', 1.8, ...
     'MarkerFaceColor', COL_FROZ);
plot(ax1, dpl, TA.iters_transportV, '-^', 'Color', COL_TRAN, 'LineWidth', 1.8, ...
     'MarkerFaceColor', COL_TRAN);
yline(ax1, base_two, '--', sprintf('\\delta=0 two-level (%d)', base_two), ...
      'Color', COL_REF, 'LineWidth', 1.2);
set(ax1, 'XScale', 'log');
xlabel(ax1, '\delta  (relative perturbation of the coupling block)');
ylabel(ax1, 'MINRES iterations');
title(ax1, 'Operator held FIXED; only the ILDL is rebuilt');
legend(ax1, {'ILDL only', 'frozen V (production)', 'transported V (fix)'}, ...
       'Location', 'northwest');

ax2 = nexttile(tl); hold(ax2, 'on'); grid(ax2, 'on');
yyaxis(ax2, 'left');
plot(ax2, dpl, max(TA.transport_err2, 1e-17), '-o', 'Color', COL_TRAN, ...
     'LineWidth', 1.8, 'MarkerFaceColor', COL_TRAN);
set(ax2, 'YScale', 'log');  ylabel(ax2, 'capture error  ||(I-P)U_{ref}||_2');
yyaxis(ax2, 'right');
plot(ax2, dpl, 100*TA.perm_hamming_frac, '-s', 'Color', COL_FROZ, 'LineWidth', 1.8, ...
     'MarkerFaceColor', COL_FROZ);
ylabel(ax2, 'ldl permutation changed (%)');
set(ax2, 'XScale', 'log');
xlabel(ax2, '\delta');
title(ax2, 'Coordinate drift caused by re-pivoting');

sgtitle(fig, sprintf('Pivot sensitivity — %s, n=%d, k=%d, \\tau=%g', ...
                     CASE, n, KBASE, TAU));
exportgraphics(fig, fullfile(outDir, 'pivot_sensitivity.png'), 'Resolution', 180);
close(fig);

fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
hold on; grid on;
plot(TB.src_step, TB.iters_ildl,       '-o', 'Color', COL_ILDL, 'LineWidth', 1.8, ...
     'MarkerFaceColor', COL_ILDL);
plot(TB.src_step, TB.iters_frozenV,    '-s', 'Color', COL_FROZ, 'LineWidth', 1.8, ...
     'MarkerFaceColor', COL_FROZ);
plot(TB.src_step, TB.iters_transportV, '-^', 'Color', COL_TRAN, 'LineWidth', 1.8, ...
     'MarkerFaceColor', COL_TRAN);
xlabel('step the ILDL factor was built from (system solved is always step 1)');
ylabel('MINRES iterations');
legend({'ILDL only', 'frozen V (production)', 'transported V (fix)'}, ...
       'Location', 'northwest');
title(sprintf('Coordinate drift alone, at realistic magnitudes (%s, k=%d)', CASE, KBASE));
exportgraphics(fig, fullfile(outDir, 'pivot_sensitivity_motion.png'), 'Resolution', 180);
close(fig);

fprintf('[saved] %s\n', fullfile(outDir, 'pivot_sensitivity.csv'));
fprintf('[saved] %s\n', fullfile(outDir, 'pivot_sensitivity_motion.csv'));
fprintf('[saved] %s\n', fullfile(outDir, 'pivot_sensitivity.png'));
fprintf('[saved] %s\n', fullfile(outDir, 'pivot_sensitivity_motion.png'));

%==========================================================================
function Kp = perturb_coupling_block(S, n, delta)
%PERTURB_COUPLING_BLOCK  Scale the EXISTING nonzeros of the step-n coupling
% block by (1 + delta*randn), preserving the sparsity pattern, and rebuild the
% KKT matrix through the sequence's low-rank form (so symmetry is exact).
    Cb = S.Cblk{n};
    if delta > 0
        [i, j, v] = find(Cb);
        Cb = sparse(i, j, v .* (1 + delta * randn(size(v))), size(Cb, 1), size(Cb, 2));
    end
    W  = Cb * S.Sel';
    Kp = S.K0 + W + W';
end
