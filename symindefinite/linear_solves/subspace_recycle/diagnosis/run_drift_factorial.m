% RUN_DRIFT_FACTORIAL  Separate coordinate drift (H1) from operator drift (H2),
% and score how much of the lost gain each candidate fix recovers.
%
% The production scheme confounds two changes per step: the operator moves
% (K_ref -> K_n) AND the split coordinates move (C_ref -> C_n).  The 2x2
% factorial below varies them independently, then adds the repairs:
%
%   refK_refC   K_ref, C_ref, V_ref     in-sample control (the step-1 result)
%   newK_refC   K_n,   C_ref, V_ref     operator only  -> isolates H2
%   refK_newC   K_ref, C_n,   V_ref     coordinates only -> isolates H1
%   newK_newC   K_n,   C_n,   V_ref     = what the benchmark actually does
%   transport   K_n,   C_n,   orth(C_n' U_ref)              H1 repair
%   frozenC_upd K_n,   C_ref, [V_ref, What_n]               H2 repair
%   both        K_n,   C_n,   [transport, What_n]           H1 + H2
%   rebuild     K_n,   C_n,   fresh eigs                    oracle
%   ildl        K_n,   C_n,   []                            no coarse space
%
% recovery_ratio = (it_production - it_cell) / (it_production - it_oracle):
% 0 = no better than today, 1 = as good as rebuilding V from scratch.
%
% Note the cells with mismatched (K, C) pairs are diagnostic, not deployable:
% refK_newC preconditions K_ref with a factor built from K_n.  That is the point
% — it is the only way to see the coordinate effect on its own.
%
% Fast default ~5 min; set FULL = true for benchmark scale.
%
% See also: run_pivot_sensitivity, run_ildl_drift, run_eigenspace_motion.

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
    H0 = 0.05;  KBASE = 500;  STEPS = [2 3 5 10 20 40 60];
    CASES = {'bar_rotating', 'disk_translating', 'disk_static'};
else
    H0 = 0.1;   KBASE = 100;  STEPS = [2 3 5 10];
    CASES = {'bar_rotating', 'disk_static'};
end
TAU = 1;  TOL = 1e-8;

CELLS = {'ildl', 'refK_refC', 'newK_refC', 'refK_newC', 'newK_newC', ...
         'transport', 'frozenC_upd', 'both', 'rand_aug', 'rebuild'};

rows = struct('case_name', {}, 'n', {}, 'step', {}, 'cell', {}, 'coarse_dim', {}, ...
    'iters', {}, 'flag', {}, 'true_res', {}, 'condE', {}, 'sqrtMinEigE', {}, ...
    'n_backsolves', {}, 'setup_time', {}, 'solve_time', {}, 'recovery_ratio', {});

fprintf('=== run_drift_factorial (h0=%g, k=%d) ===\n', H0, KBASE);

for cc = 1:numel(CASES)
    cname = CASES{cc};
    S = build_stokes_sequence(struct('case_name', cname, 'h0', H0, ...
                                     'nsteps', max(STEPS), 'quiet', true));
    n = S.n;  so = struct('tau', TAU, 'tol', TOL, 'maxit', min(4000, n));

    K1 = seq_K(S, 1);
    P1 = src.precond.make_ildl_precond(K1, struct('mode', 'nofill'));
    [C1, ~] = ildl_coordinate_map(P1);
    M1 = C1 * C1';  M1 = (M1 + M1') / 2;
    [Uref, ~] = eigs(K1, M1, KBASE, 'smallestabs', 'Tolerance', 1e-10, ...
                     'MaxIterations', 2000);
    Vref = transport_V(Uref, P1, C1);
    ctx = [];

    fprintf('\n  [%s] n=%d nC=%d\n', cname, n, S.nC);
    fprintf('  %5s', 'step');
    fprintf(' %11s', CELLS{:});  fprintf('\n');

    for st = STEPS
        Kn = seq_K(S, st);  bn = S.b{st};
        Pn = src.precond.make_ildl_precond(Kn, struct('mode', 'nofill'));
        Cn = ildl_coordinate_map(Pn);

        % repairs
        Vtr = transport_V(Uref, Pn, Cn);
        [W_ref, i_ref, ctx] = lowrank_update_basis(S, st, P1, ctx, ...
                                  struct('mode', 'invref', 'ref', 1, 'Cn', C1));
        [W_new, i_new, ctx] = lowrank_update_basis(S, st, Pn, ctx, ...
                                  struct('mode', 'invref', 'ref', 1, 'Cn', Cn));
        Vupd  = augment_recycle_V(Vref, W_ref);
        Vboth = augment_recycle_V(Vtr,  W_new);
        % control: the SAME number of extra columns, but random.  Without this,
        % "+both" could be winning merely by having a larger coarse space.
        Vrnd  = augment_recycle_V(Vtr, randn(n, size(W_new, 2)));

        % oracle: the coarse space the step would have built for itself
        Mn = Cn * Cn';  Mn = (Mn + Mn') / 2;
        [Un, ~] = eigs(Kn, Mn, KBASE, 'smallestabs', 'Tolerance', 1e-10, ...
                       'MaxIterations', 2000);
        Vor = transport_V(Un, Pn, Cn);

        spec = { 'ildl',        Kn, bn, Pn, [],    0 ; ...
                 'refK_refC',   K1, S.b{1}, P1, Vref,  0 ; ...
                 'newK_refC',   Kn, bn, P1, Vref,  0 ; ...
                 'refK_newC',   K1, S.b{1}, Pn, Vref,  0 ; ...
                 'newK_newC',   Kn, bn, Pn, Vref,  0 ; ...
                 'transport',   Kn, bn, Pn, Vtr,   0 ; ...
                 'frozenC_upd', Kn, bn, P1, Vupd,  i_ref.n_backsolves ; ...
                 'both',        Kn, bn, Pn, Vboth, i_new.n_backsolves ; ...
                 'rand_aug',    Kn, bn, Pn, Vrnd,  0 ; ...
                 'rebuild',     Kn, bn, Pn, Vor,   0 };

        res = struct();
        for q = 1:size(spec, 1)
            r = two_level_it(spec{q,2}, spec{q,3}, spec{q,4}, spec{q,5}, so);
            res.(spec{q,1}) = r;
            rows(end+1) = struct('case_name', cname, 'n', n, 'step', st, ...
                'cell', spec{q,1}, 'coarse_dim', r.coarse_dim, ...
                'iters', r.iters, 'flag', r.flag, 'true_res', r.true_res, ...
                'condE', r.condE, 'sqrtMinEigE', r.sqrt_minEigE, ...
                'n_backsolves', spec{q,6}, 'setup_time', r.setup_time, ...
                'solve_time', r.time, 'recovery_ratio', NaN); %#ok<SAGROW>
        end

        % recovery ratios, relative to production and the oracle
        it_prod = res.newK_newC.iters;  it_or = res.rebuild.iters;
        den = it_prod - it_or;
        for q = 1:size(spec, 1)
            idx = numel(rows) - size(spec,1) + q;
            if den ~= 0
                rows(idx).recovery_ratio = (it_prod - rows(idx).iters) / den;
            else
                % Production already matches the oracle (the disk_static null
                % control): nothing was lost, so nothing is left to recover.
                rows(idx).recovery_ratio = 1;
            end
        end

        fprintf('  %5d', st);
        for q = 1:size(spec, 1), fprintf(' %11d', res.(spec{q,1}).iters); end
        fprintf('\n');
    end
end

T = struct2table(rows);
writetable(T, fullfile(outDir, 'drift_factorial.csv'));

%% ===== verdict ==========================================================
fprintf('\n==================================================================\n');
for cc = 1:numel(CASES)
    cn = CASES{cc};
    Tc = T(strcmp(T.case_name, cn), :);
    if isempty(Tc), continue; end
    g = @(k) mean(Tc.iters(strcmp(Tc.cell, k)));
    rec = @(k) mean(Tc.recovery_ratio(strcmp(Tc.cell, k)));
    fprintf('  [%s]  mean iterations over steps %s\n', cn, mat2str(STEPS));
    fprintf('    ILDL only (no coarse space)      %6.0f\n', g('ildl'));
    fprintf('    step-1 in-sample                 %6.0f   (what deflation is worth)\n', ...
            g('refK_refC'));
    fprintf('    PRODUCTION (frozen V, new C)     %6.0f   = %.2fx ILDL\n', ...
            g('newK_newC'), g('ildl')/g('newK_newC'));
    fprintf('    oracle (rebuild V every step)    %6.0f\n', g('rebuild'));
    fprintf('\n    clean single-cause comparisons (same K, same smoother):\n');
    fprintf('      production -> transport        %6.0f -> %-6.0f  H1 costs %.2fx\n', ...
            g('newK_newC'), g('transport'), g('newK_newC')/g('transport'));
    fprintf('      transport  -> oracle           %6.0f -> %-6.0f  H2 costs %.2fx\n', ...
            g('transport'), g('rebuild'), g('transport')/g('rebuild'));
    fprintf('\n    repairs (recovery: 0 = production, 1 = oracle):\n');
    fprintf('      + transport            (H1)    %6.0f   recovery %.2f\n', ...
            g('transport'), rec('transport'));
    fprintf('      + lowrank, C frozen    (H2)    %6.0f   recovery %.2f\n', ...
            g('frozenC_upd'), rec('frozenC_upd'));
    fprintf('      + BOTH                         %6.0f   recovery %.2f\n', ...
            g('both'), rec('both'));
    fprintf('      + transport & RANDOM cols      %6.0f   recovery %.2f  <- control\n', ...
            g('rand_aug'), rec('rand_aug'));
    fprintf('\n    mismatched-pair diagnostics (not deployable; the smoother is\n');
    fprintf('    built from a different matrix than the one being solved):\n');
    fprintf('      newK_refC %6.0f    refK_newC %6.0f\n', g('newK_refC'), g('refK_newC'));
end
fprintf('==================================================================\n');

%% ===== figure ===========================================================
for cc = 1:numel(CASES)
    cn = CASES{cc};
    Tc = T(strcmp(T.case_name, cn), :);
    if isempty(Tc), continue; end
    show = {'newK_newC', 'transport', 'frozenC_upd', 'both', 'rand_aug', 'rebuild'};
    lab  = {'production (frozen V)', '+transport (H1)', '+lowrank (H2)', ...
            '+both', '+random cols (control)', 'oracle (rebuild)'};
    Y = zeros(numel(STEPS), numel(show));
    for a = 1:numel(STEPS)
        for q = 1:numel(show)
            Y(a, q) = mean(Tc.iters(Tc.step == STEPS(a) & strcmp(Tc.cell, show{q})));
        end
    end

    fig = figure('Visible', 'off', 'Position', [100 100 900 480]);
    b = bar(Y, 'grouped');  hold on; grid on;
    cmap = [0.50 0.35 0.65; 0.20 0.45 0.70; 0.30 0.65 0.35; ...
            0.15 0.55 0.50; 0.70 0.55 0.40; 0.35 0.35 0.35];
    for q = 1:numel(b), b(q).FaceColor = cmap(q, :); end
    yline(mean(Tc.iters(strcmp(Tc.cell, 'ildl'))), '--', 'ILDL only', ...
          'Color', [0.85 0.40 0.32], 'LineWidth', 1.4);
    yline(mean(Tc.iters(strcmp(Tc.cell, 'refK_refC'))), ':', 'step-1 in-sample', ...
          'Color', [0.35 0.35 0.35], 'LineWidth', 1.4);
    set(gca, 'XTickLabel', arrayfun(@(s) sprintf('step %d', s), STEPS, ...
             'UniformOutput', false));
    ylabel('MINRES iterations');
    legend(lab, 'Location', 'eastoutside');
    title(sprintf('Coordinate vs operator drift — %s (n=%d, k=%d, \\tau=%g)', ...
                  cn, Tc.n(1), KBASE, TAU));
    exportgraphics(fig, fullfile(outDir, sprintf('drift_factorial_%s.png', cn)), ...
                   'Resolution', 180);
    close(fig);
end

fprintf('[saved] %s\n', fullfile(outDir, 'drift_factorial.csv'));
fprintf('[saved] %s\n', fullfile(outDir, 'drift_factorial_<case>.png'));
