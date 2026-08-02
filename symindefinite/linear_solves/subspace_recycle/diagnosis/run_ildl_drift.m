% RUN_ILDL_DRIFT  How far the split coordinate system moves per time step, and
% what it costs.
%
% Two questions on the REAL sequence (run_pivot_sensitivity answered them with
% the operator artificially held fixed):
%
%   1. How fast does the coordinate system drift?  Tracked by the ldl
%      permutation Hamming distance and, more meaningfully, by
%          theta_n = angle( U_ref ,  C_n^-T V_ref )
%      i.e. how far what the FROZEN basis actually deflates at step n has moved
%      from the eigenvectors it was built from.  theta_n -> 1 means the
%      production scheme is deflating an unrelated subspace.
%
%   2. Is freezing the ILDL free?  run_pivot_sensitivity found a mismatched
%      factor is itself a ~2x worse smoother, which would make "freeze C to keep
%      the coordinates stable" a bad trade.  Here ILDL-only MINRES is run at
%      every step with the refreshed factor and with the step-1 factor, so the
%      penalty is measured directly.
%
% Fast default ~3 min; set FULL = true for benchmark scale (60 steps, n=5840).
%
% See also: run_pivot_sensitivity, run_drift_factorial, ildl_coordinate_map.

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
    H0 = 0.05;  KBASE = 500;  NS = 60;  CASES = {'bar_rotating','disk_translating','disk_static'};
else
    H0 = 0.1;   KBASE = 100;  NS = 12;  CASES = {'bar_rotating','disk_static'};
end
TAU = 1;  TOL = 1e-8;

COL_ILDL = [0.85 0.40 0.32];  COL_FROZ = [0.50 0.35 0.65];
COL_TRAN = [0.20 0.45 0.70];  COL_REF  = [0.35 0.35 0.35];

rows = struct('case_name', {}, 'n', {}, 'step', {}, 'coupling_change', {}, ...
    'perm_hamming', {}, 'perm_hamming_frac', {}, 'nnzL', {}, 'nnzL_ratio', {}, ...
    'n_offdiag_pivots', {}, 'transport_err2', {}, 'transport_errfro', {}, ...
    'n_ang_below_1pct', {}, 'iters_ildl_refresh', {}, 'iters_ildl_frozen', {}, ...
    'true_res_refresh', {}, 'true_res_frozen', {});

fprintf('=== run_ildl_drift (h0=%g, k=%d, %d steps) ===\n', H0, KBASE, NS);

for cc = 1:numel(CASES)
    cname = CASES{cc};
    S = build_stokes_sequence(struct('case_name', cname, 'h0', H0, ...
                                     'nsteps', NS, 'quiet', true));
    n = S.n;  maxit = min(4000, n);
    so = struct('tau', TAU, 'tol', TOL, 'maxit', maxit);

    K1 = seq_K(S, 1);
    P1 = src.precond.make_ildl_precond(K1, struct('mode', 'nofill'));
    [C1, info1] = ildl_coordinate_map(P1);
    M1 = C1 * C1';  M1 = (M1 + M1') / 2;
    [Uref, ~] = eigs(K1, M1, KBASE, 'smallestabs', 'Tolerance', 1e-10, ...
                     'MaxIterations', 2000);
    Vref = transport_V(Uref, P1, C1);

    fprintf('\n  [%s] n=%d nC=%d\n', cname, n, S.nC);
    fprintf('  %5s %9s %8s %11s %10s %10s\n', ...
            'step', 'dC', 'permH%', 'transport_e2', 'ildl_new', 'ildl_frozen');
    for st = 1:S.nsteps
        Kn = seq_K(S, st);  bn = S.b{st};
        Pn = src.precond.make_ildl_precond(Kn, struct('mode', 'nofill'));
        [~, infon] = ildl_coordinate_map(Pn, info1);

        cap = subspace_capture_directed(Uref, Pn.applyCtinv(Vref), []);

        r_new = two_level_it(Kn, bn, Pn, [], so);
        r_frz = two_level_it(Kn, bn, P1, [], so);

        rows(end+1) = struct('case_name', cname, 'n', n, 'step', st, ...
            'coupling_change', S.coupling_change(st), ...
            'perm_hamming', infon.perm_hamming, ...
            'perm_hamming_frac', infon.perm_hamming_frac, ...
            'nnzL', infon.nnzL, 'nnzL_ratio', infon.nnzL_ratio, ...
            'n_offdiag_pivots', infon.n_offdiag_pivots, ...
            'transport_err2', cap.eigspace_err_2, ...
            'transport_errfro', cap.eigspace_err_fro, ...
            'n_ang_below_1pct', cap.n_angle_below_1pct, ...
            'iters_ildl_refresh', r_new.iters, 'iters_ildl_frozen', r_frz.iters, ...
            'true_res_refresh', r_new.true_res, ...
            'true_res_frozen', r_frz.true_res); %#ok<SAGROW>

        if st <= 5 || mod(st, max(1, round(S.nsteps/6))) == 0
            fprintf('  %5d %9.4f %7.1f%% %11.3e %10d %10d\n', st, ...
                    S.coupling_change(st), 100*infon.perm_hamming_frac, ...
                    cap.eigspace_err_2, r_new.iters, r_frz.iters);
        end
    end
end

T = struct2table(rows);
writetable(T, fullfile(outDir, 'ildl_drift.csv'));

%% ===== verdict ==========================================================
fprintf('\n==================================================================\n');
for cc = 1:numel(CASES)
    cn = CASES{cc};
    Tc = T(strcmp(T.case_name, cn), :);
    mv = Tc.step > 1;
    if ~any(mv), continue; end
    fprintf('  [%s]\n', cn);
    fprintf('    capture of U_ref by the frozen basis: %.3f at step 2, %.3f median\n', ...
            Tc.transport_err2(2), median(Tc.transport_err2(mv)));
    fprintf('    directions still captured to <1%%   : %d of %d at step 2\n', ...
            Tc.n_ang_below_1pct(2), KBASE);
    fprintf('    ldl permutation moved              : %.0f%% median\n', ...
            100*median(Tc.perm_hamming_frac(mv)));
    pen = mean(Tc.iters_ildl_frozen(mv)) / mean(Tc.iters_ildl_refresh(mv));
    fprintf('    ILDL-only refreshed %.0f vs frozen %.0f iters -> freezing costs %.2fx\n', ...
            mean(Tc.iters_ildl_refresh(mv)), mean(Tc.iters_ildl_frozen(mv)), pen);
end
fprintf('==================================================================\n');

%% ===== figures ==========================================================
for cc = 1:numel(CASES)
    cn = CASES{cc};
    Tc = T(strcmp(T.case_name, cn), :);

    fig = figure('Visible', 'off', 'Position', [100 100 1220 420]);
    tl = tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

    ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
    plot(ax, Tc.step, max(Tc.transport_err2, 1e-17), '-o', 'Color', COL_TRAN, ...
         'LineWidth', 1.8, 'MarkerFaceColor', COL_TRAN, 'MarkerSize', 4);
    plot(ax, Tc.step, max(Tc.transport_errfro, 1e-17), '-s', 'Color', COL_FROZ, ...
         'LineWidth', 1.4, 'MarkerSize', 4);
    yline(ax, 1, '--', 'total loss', 'Color', COL_REF, 'LineWidth', 1.2);
    set(ax, 'YScale', 'log');
    xlabel(ax, 'time step');  ylabel(ax, 'capture error of U_{ref}');
    title(ax, 'What the frozen basis actually deflates');
    legend(ax, {'||(I-P)U||_2', '||(I-P)U||_F/\surdk'}, 'Location', 'southeast');

    ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
    yyaxis(ax, 'left');
    plot(ax, Tc.step, 100*Tc.perm_hamming_frac, '-o', 'Color', COL_FROZ, ...
         'LineWidth', 1.8, 'MarkerFaceColor', COL_FROZ, 'MarkerSize', 4);
    ylabel(ax, 'ldl permutation changed (%)');  ylim(ax, [0 100]);
    yyaxis(ax, 'right');
    plot(ax, Tc.step, Tc.coupling_change, '-^', 'Color', COL_ILDL, ...
         'LineWidth', 1.4, 'MarkerSize', 4);
    ylabel(ax, '||\DeltaC||_F / ||C||_F');
    xlabel(ax, 'time step');
    title(ax, 'Coupling motion drives re-pivoting');

    ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
    plot(ax, Tc.step, Tc.iters_ildl_refresh, '-o', 'Color', COL_ILDL, ...
         'LineWidth', 1.8, 'MarkerFaceColor', COL_ILDL, 'MarkerSize', 4);
    plot(ax, Tc.step, Tc.iters_ildl_frozen, '--s', 'Color', COL_ILDL, ...
         'LineWidth', 1.6, 'MarkerSize', 4);
    xlabel(ax, 'time step');  ylabel(ax, 'MINRES iterations (ILDL only)');
    title(ax, 'Is freezing the ILDL free?');
    legend(ax, {'refreshed every step', 'frozen at step 1'}, 'Location', 'northwest');

    sgtitle(fig, sprintf('Coordinate drift — %s, n=%d, k=%d', cn, Tc.n(1), KBASE));
    exportgraphics(fig, fullfile(outDir, sprintf('ildl_drift_%s.png', cn)), ...
                   'Resolution', 180);
    close(fig);
end

fprintf('[saved] %s\n', fullfile(outDir, 'ildl_drift.csv'));
fprintf('[saved] %s\n', fullfile(outDir, 'ildl_drift_<case>.png'));
