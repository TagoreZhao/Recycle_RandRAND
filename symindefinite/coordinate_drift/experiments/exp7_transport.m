function [V, D] = exp7_transport(opts)
%EXP7_TRANSPORT  The repair: cache the subspace, re-chart on use.
%Tests Thm 5.1 (transport is exact) and Prop 5.2 (the numerical price).
%
%   [V, D] = EXP7_TRANSPORT(OPTS)
%
%   Cache the PHYSICAL subspace U_n = C_n^-T span(V_n) and, at step n+1, use
%   span(V_{n+1}) = C_{n+1}^T U_n.  Then
%
%       C_{n+1}^-T span(V_{n+1}) = U_n     exactly,
%
%   so the chart term of Thm 2.1 is identically zero and only delta_prec and
%   delta_op survive.  Measured three ways per step: the physical round trip,
%   the chart term itself, and MINRES iterations for frozen / transported /
%   no-coarse-space / rebuilt-from-scratch.
%
%   The oracle column matters: transport does not recover it, and is not
%   supposed to.  What it removes is the chart error; what remains is the
%   genuine motion of the target, which no bookkeeping can undo.
%
%   Prop 5.2: C_{n+1}^T is not an isometry, so orthonormality is lost in the
%   re-charting and about log10(kappa_2(C_{n+1})) digits go with it.  Reported
%   alongside the rank drop of the column-pivoted QR.
%
%   Each family solves with its own coarse correction and Krylov method
%   (MINRES + the square-root form for 'ildl', PCG + the direct SPD form for
%   'ichol'), so counts are comparable within a family but not across.
%
%   See also: transport_V, orth_trunc, two_level_solve_local, coarse_correction.

    if nargin < 1 || isempty(opts), opts = struct(); end
    p = add_paths();
    k    = getdef(opts, 'k',      20);
    np   = getdef(opts, 'npairs', 4);
    % tau travels with the case (make_case sets cs.tau from opts.tau, default 0.5)
    tol  = getdef(opts, 'tol',    1e-8);
    mit  = getdef(opts, 'mit',    3000);

    V = struct([]);  D = struct();

    for fam = {'ildl', 'ichol'}
        f = fam{1};  rows = [];
        c1 = make_case(f, 1, opts);
        U1 = pencil_subspace(c1.A, c1.M, k, opts);       % built once, then cached
        V1 = orth_trunc(c1.C' * U1);
        Uc = c1.applyCtinv(V1);                          % the cached physical space

        for i = 2:np+1
            c2 = make_case(f, i, opts);
            b  = ones(c2.n, 1);

            [V2, ti] = transport_V(Uc, [], c2.C);        % orth(C_{n+1}^T U)
            back  = c2.applyCtinv(V2);
            round_trip = gap(back, Uc);                  % Thm 5.1
            [d_frozen, gf] = gap(c2.applyCtinv(V1), Uc); % what freezing denotes

            Uo = pencil_subspace(c2.A, c2.M, k, opts);   % the oracle
            Vo = orth_trunc(c2.C' * Uo);
            [d_or, go] = gap(V2, Vo);

            [~,~,~, it_f] = two_level_solve_local(c2.A, b, tol, mit, c2, V1, c2.tau);
            [~,~,~, it_t] = two_level_solve_local(c2.A, b, tol, mit, c2, V2, c2.tau);
            [~,~,~, it_n] = two_level_solve_local(c2.A, b, tol, mit, c2, [], c2.tau);
            [~,~,~, it_o] = two_level_solve_local(c2.A, b, tol, mit, c2, Vo, c2.tau);

            rows = [rows; i, round_trip, d_frozen, gf.dF, d_or, go.dF, ...
                    it_f, it_t, it_n, it_o, ...
                    cond(full(c2.C)), ti.rank_drop, ...
                    (it_f - it_t) / max(it_f - it_o, 1)]; %#ok<AGROW>
        end

        T = array2table(rows, 'VariableNames', ...
            {'step', 'transport_round_trip', 'frozen_chart_error', 'dF_frozen', ...
             'gap_to_oracle', 'dF_to_oracle', ...
             'it_frozen', 'it_transported', 'it_none', 'it_oracle', ...
             'cond_C', 'rank_drop', 'recovery'});
        D.(f) = T;
        disp(['[exp7/' f ']']);  disp(T);

        exact = max(T.transport_round_trip) < 1e-8;
        beats = all(T.it_transported <= T.it_frozen);
        V = [V, vrec(['exp7/' f], 'Thm 5.1 transport preserves the physical span exactly', ...
                     'max round-trip gap over steps', max(T.transport_round_trip), ...
                     '< 1e-8', exact)]; %#ok<AGROW>
        V = [V, vrec(['exp7/' f], 'Thm 5.1 the frozen chart error it removes', ...
                     'median frozen chart error (2-norm | Frobenius)', ...
                     sprintf('%.4g | %.4g', median(T.frozen_chart_error), ...
                             median(T.dF_frozen)), 'ildl ~ 1, ichol ~ 0', NaN)]; %#ok<AGROW>
        V = [V, vrec(['exp7/' f], 'what transport recovers of the frozen-to-oracle gap', ...
                     'median recovery fraction', median(T.recovery), ...
                     'chart error removed, physics remains', NaN)]; %#ok<AGROW>
        V = [V, vrec(['exp7/' f], 'transported is never worse than frozen', ...
                     'its transported <= frozen, all steps', beats, 'true', beats)]; %#ok<AGROW>
        V = [V, vrec(['exp7/' f], 'iterations frozen | transported | none | oracle', ...
                     'median over steps', ...
                     sprintf('%g | %g | %g | %g', median(T.it_frozen), ...
                             median(T.it_transported), median(T.it_none), ...
                             median(T.it_oracle)), 'transported between', NaN)]; %#ok<AGROW>
        V = [V, vrec(['exp7/' f], 'Prop 5.2 conditioning of the re-charting', ...
                     'median cond_2(C) | total rank drop', ...
                     sprintf('%.3g | %d', median(T.cond_C), sum(T.rank_drop)), ...
                     'round trip holds to ~ eps*cond(C)', NaN)]; %#ok<AGROW>
        writetable(T, fullfile(p.outDir, ['exp7_transport_' f '.csv']));
    end

    %% ---- figure ----------------------------------------------------------
    fh = figure('Visible', 'off');
    tl = tiledlayout(fh, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    for fam = {'ildl', 'ichol'}
        f = fam{1};  T = D.(f);
        nexttile(tl);
        Y = [T.it_frozen, T.it_transported, T.it_none, T.it_oracle];
        bar(T.step, Y);
        xlabel('step');  ylabel('MINRES iterations');
        title(sprintf('%s: what the repair buys', upper(f)));
        legend({'frozen V (numbers reused)', 'transported', 'no coarse space', ...
                'oracle (rebuilt)'}, 'Location', 'northwest');
        ylim([0 max(Y(:)) * 1.40]);
    end
    save_figure(fh, 'transport');
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
