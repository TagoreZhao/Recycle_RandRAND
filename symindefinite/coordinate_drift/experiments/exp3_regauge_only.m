function [V, D] = exp3_regauge_only(opts)
%EXP3_REGAUGE_ONLY  A perfect preconditioner still breaks a frozen basis.
%Tests Prop 2.4, and measures the gauge/metric split on the real factors.
%
%   [V, D] = EXP3_REGAUGE_ONLY(OPTS)
%
%   Part A -- the controlled proof.  Hold the operator A and the metric
%   M = C*C' BIT-IDENTICAL and change only the factor, C2 = C1*Q with Q a random
%   orthogonal matrix.  Nothing about the preconditioner has changed: same M,
%   same spectrum of Ahat up to an orthogonal similarity, same everything a
%   preconditioner is supposed to be.  Yet a frozen coarse basis is destroyed,
%   because the numbers in V denote a different physical subspace in the new
%   chart.  This isolates the culprit as the FACTOR, not the preconditioner --
%   the sharpest form of the failure, with every competing explanation
%   (the operator moved, the metric moved, the target moved) held at zero.
%
%   Part B -- the same measurement on the real factors.  Step n to n+1 for both
%   families, reporting ||dC||/||C|| against ||dM||/||M|| and the Procrustes
%   split of delta_chart into delta_metric + delta_gauge.  Answers whether the
%   observed drift is mostly a regauge or a genuine metric change.
%
%   Writes figures/gauge_vs_metric.png.
%
%   See also: gauge_split, chart_struct, exp1_chart_and_invariance.

    if nargin < 1 || isempty(opts), opts = struct(); end
    p = add_paths();
    k    = getdef(opts, 'k',    20);
    tol  = getdef(opts, 'tol',  1e-8);
    mit  = getdef(opts, 'mit',  2000);
    nprs = getdef(opts, 'npairs', 4);
    rng(0);

    V = struct([]);
    D = struct();

    %% ---- Part A: pure regauge, operator and metric held identical --------
    for fam = {'ildl', 'ichol'}
        f  = fam{1};
        cs = make_case(f, 1, opts);
        n  = cs.n;
        b  = ones(n, 1);
        U  = pencil_subspace(cs.A, cs.M, k, opts);
        Vb = orth_trunc(cs.C' * U);

        Q   = orth(randn(n));
        C2  = cs.C * Q;
        P2  = chart_struct(C2, cs.defl_kind, cs.tau);
        M2  = C2 * C2';
        dM  = norm(full(M2 - cs.M), 2) / norm(full(cs.M), 2);

        gs  = gauge_split(cs.C, C2, Vb);
        Vtr = orth_trunc(C2' * (cs.applyCtinv(Vb)));      % transported

        [~, ~, ~, it_ref] = two_level_solve_local(cs.A, b, tol, mit, cs, Vb,  cs.tau);
        [~, ~, ~, it_frz] = two_level_solve_local(cs.A, b, tol, mit, P2, Vb,  cs.tau);
        [~, ~, ~, it_trp] = two_level_solve_local(cs.A, b, tol, mit, P2, Vtr, cs.tau);
        [~, ~, ~, it_non] = two_level_solve_local(cs.A, b, tol, mit, P2, [],  cs.tau);

        D.(f).pure = struct('n', n, 'dM', dM, 'delta_chart', gs.delta_chart, ...
                            'delta_metric', gs.delta_metric, 'delta_gauge', gs.delta_gauge, ...
                            'it_ref', it_ref, 'it_frozen', it_frz, ...
                            'it_transported', it_trp, 'it_none', it_non);

        V = [V, vrec(['exp3A/' f], 'Prop 2.4 the metric does not move at all', ...
                     '||M2-M1||/||M1||', dM, '< 1e-12', dM < 1e-12)]; %#ok<AGROW>
        V = [V, vrec(['exp3A/' f], 'Prop 2.4 delta_metric vanishes, delta_gauge does not', ...
                     'delta_metric | delta_gauge', ...
                     sprintf('%.2e | %.3f', gs.delta_metric, gs.delta_gauge), ...
                     'metric ~ 0, gauge ~ 1', ...
                     gs.delta_metric < 1e-8 && gs.delta_gauge > 0.5)]; %#ok<AGROW>
        V = [V, vrec(['exp3A/' f], 'Prop 2.4 transport restores the reference count', ...
                     'its ref | frozen | transported | none', ...
                     sprintf('%d | %d | %d | %d', it_ref, it_frz, it_trp, it_non), ...
                     'transported = ref, frozen > ref', ...
                     abs(it_trp - it_ref) <= 1 && it_frz > it_ref)]; %#ok<AGROW>

        fprintf(['[exp3A/%s] regauge only: dM=%.1e  d_metric=%.1e  d_gauge=%.3f  ' ...
                 'its ref=%d frozen=%d transported=%d none=%d\n'], ...
                f, dM, gs.delta_metric, gs.delta_gauge, it_ref, it_frz, it_trp, it_non);
    end

    %% ---- Part B: the real factors, step n -> n+1 -------------------------
    for fam = {'ildl', 'ichol'}
        f = fam{1};
        rows = [];
        for i = 1:nprs
            c1 = make_case(f, i,     opts);
            c2 = make_case(f, i + 1, opts);
            U  = pencil_subspace(c1.A, c1.M, k, opts);
            Vb = orth_trunc(c1.C' * U);
            gs = gauge_split(c1.C, c2.C, Vb);
            dA = norm(full(c2.A - c1.A), 2) / norm(full(c1.A), 2);
            rows = [rows; i, dA, gs.relC, gs.relC_aligned, gs.relM, ...
                    gs.delta_chart, gs.delta_metric, gs.delta_gauge, ...
                    gs.aligned_fraction]; %#ok<AGROW>
        end
        D.(f).pairs = array2table(rows, 'VariableNames', ...
            {'step', 'relA', 'relC', 'relC_aligned', 'relM', ...
             'delta_chart', 'delta_metric', 'delta_gauge', 'aligned_fraction'});
        disp(['[exp3B/' f ']']); disp(D.(f).pairs);

        tri = all(rows(:, 6) <= rows(:, 7) + rows(:, 8) + 1e-10);
        V = [V, vrec(['exp3B/' f], 'delta_chart <= delta_metric + delta_gauge', ...
                     'triangle over all pairs', tri, 'true', tri)]; %#ok<AGROW>
        V = [V, vrec(['exp3B/' f], 'median relative factor motion ||dC||/||C||', ...
                     'median relC', median(rows(:, 3)), 'ildl >> ichol', NaN)]; %#ok<AGROW>
        V = [V, vrec(['exp3B/' f], 'median relative metric motion ||dM||/||M||', ...
                     'median relM', median(rows(:, 5)), 'ildl >> ichol', NaN)]; %#ok<AGROW>
    end

    %% ---- figure ----------------------------------------------------------
    fh = figure('Visible', 'off');
    tl = tiledlayout(fh, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile(tl); hold on;
    for fam = {'ildl', 'ichol'}
        T = D.(fam{1}).pairs;
        plot(T.step, T.relC, '-o', 'LineWidth', 1.6, 'DisplayName', [fam{1} ': ||dC||/||C||']);
        plot(T.step, T.relM, '--s', 'LineWidth', 1.6, 'DisplayName', [fam{1} ': ||dM||/||M||']);
    end
    set(gca, 'YScale', 'log'); xlabel('step pair n \rightarrow n+1');
    ylabel('relative 2-norm change'); title('Factor vs metric motion');
    legend('Location', 'east'); hold off;

    nexttile(tl); hold on;
    T = D.ildl.pairs;   Ti = D.ichol.pairs;
    bar([T.delta_metric, T.delta_gauge, T.delta_chart]);
    plot(Ti.delta_chart, 'k-o', 'LineWidth', 1.8, 'DisplayName', 'ichol \delta_{chart}');
    ylim([0 1.15]); xlabel('step pair n \rightarrow n+1'); ylabel('gap  d(\cdot,\cdot)');
    title('Chart drift: ILDL bars, ichol line');
    legend({'ILDL \delta_{metric}', 'ILDL \delta_{gauge}', 'ILDL \delta_{chart}', ...
            'ichol \delta_{chart}'}, 'Location', 'southeast');
    hold off;

    save_figure(fh, 'gauge_vs_metric');
    writetable(D.ildl.pairs,  fullfile(p.outDir, 'exp3_pairs_ildl.csv'));
    writetable(D.ichol.pairs, fullfile(p.outDir, 'exp3_pairs_ichol.csv'));
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
