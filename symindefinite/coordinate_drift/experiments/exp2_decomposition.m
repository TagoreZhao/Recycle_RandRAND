function [V, D] = exp2_decomposition(opts)
%EXP2_DECOMPOSITION  Measure the three terms of Thm 2.1 separately.
%Tests Thm 2.1 (the decomposition and its reverse bound) and Thm 2.3 (Davis-Kahan
%on delta_op, and the rank-2nC structure of the operator change).
%
%   [V, D] = EXP2_DECOMPOSITION(OPTS)
%
%   Everything is measured as an M_{n+1}-gap between PHYSICAL subspaces, so no
%   basis and no chart enters a reported number:
%
%     F      = C_{n+1}^-T span(V_n)   what the frozen numbers denote at step n+1
%     U_n    = U_k(A_n,   M_n  )      what they were built to denote
%     U_n'   = U_k(A_n,   M_{n+1})    same operator, refreshed preconditioner
%     U_{n+1}= U_k(A_{n+1},M_{n+1})   the truth at step n+1
%
%     delta_chart = d_M(F, U_n)       pure C effect  (A and the target fixed)
%     delta_prec  = d_M(U_n, U_n')    the metric moved the target
%     delta_op    = d_M(U_n', U_{n+1})pure A effect  (metric fixed)
%
%   The point of the split is that the three terms answer three different
%   questions, and only delta_op is the one a time-stepping code should have to
%   pay.  Writes figures/decomposition_bars.png.
%
%   See also: gap_M, pencil_subspace, exp3_regauge_only.

    if nargin < 1 || isempty(opts), opts = struct(); end
    p = add_paths();
    k     = getdef(opts, 'k', 20);
    npair = getdef(opts, 'npairs', 4);

    V = struct([]);  D = struct();

    for fam = {'ildl', 'ichol'}
        f = fam{1};  rows = [];
        for i = 1:npair
            c1 = make_case(f, i,     opts);
            c2 = make_case(f, i + 1, opts);
            M2 = c2.M;

            U1  = pencil_subspace(c1.A, c1.M, k, opts);
            U1p = pencil_subspace(c1.A, M2,   k, opts);
            [U2, ~, i2] = pencil_subspace(c2.A, M2, k, opts);

            V1 = orth_trunc(c1.C' * U1);                 % frozen chart numbers
            F  = c2.applyCtinv(V1);                      % what they now denote

            [d_ch, g_ch] = gap_M(F,   U1,  M2);
            [d_pr, g_pr] = gap_M(U1,  U1p, M2);
            [d_op, g_op] = gap_M(U1p, U2,  M2);
            [d_tt, g_tt] = gap_M(F,   U2,  M2);

            % Davis-Kahan for delta_op, in the fixed chart n+1 (Thm 2.3).
            % gamma is read off the PERTURBED operator Ahat_{n+1}, so the bound
            % needs Weyl's correction: sin(Theta) <= ||E|| / (gamma - ||E||).
            % Using ||E||/gamma is false -- see the counterexample in the README.
            % When gamma <= ||E|| the separation is not established and the
            % theorem says nothing; that is recorded as NaN, not as a pass.
            dA   = c2.A - c1.A;
            E2   = c2.applyCinv(full(dA * c2.applyCtinv(eye(c2.n))));
            nE   = norm((E2 + E2')/2, 2);
            gam  = i2.gap;
            dk   = NaN;
            if gam > nE
                dk = nE / (gam - nE);
            end

            rows = [rows; i, d_ch, d_pr, d_op, d_tt, d_ch + d_pr + d_op, ...
                    g_ch.dF, g_pr.dF, g_op.dF, g_tt.dF, ...
                    dk, nE, rank(full(dA)), i2.gap]; %#ok<AGROW>
        end
        T = array2table(rows, 'VariableNames', ...
            {'step', 'delta_chart', 'delta_prec', 'delta_op', 'delta_total', ...
             'sum_of_three', 'dF_chart', 'dF_prec', 'dF_op', 'dF_total', ...
             'DK_bound_on_delta_op', 'norm_E', 'rank_dA', 'spectral_gap'});
        D.(f) = T;
        disp(['[exp2/' f ']']);  disp(T);

        tri  = all(T.delta_total <= T.sum_of_three + 1e-9);
        rev  = all(T.delta_total >= T.delta_chart - T.delta_prec - T.delta_op - 1e-9);
        % Thm 2.3 only says something where the gap beats the perturbation.
        appl = isfinite(T.DK_bound_on_delta_op);
        dkok = all(T.delta_op(appl) <= T.DK_bound_on_delta_op(appl) + 1e-9);

        V = [V, vrec(['exp2/' f], 'Thm 2.1 delta_total <= chart + prec + op', ...
                     'triangle inequality over all pairs', tri, 'true', tri)]; %#ok<AGROW>
        V = [V, vrec(['exp2/' f], 'Thm 2.1 reverse bound is not vacuous', ...
                     'total >= chart - prec - op', rev, 'true', rev)]; %#ok<AGROW>
        % Where no pair is applicable the claim is untested, not confirmed:
        % record REPORT rather than a vacuous PASS.
        if any(appl)
            V = [V, vrec(['exp2/' f], 'Thm 2.3 Davis-Kahan bounds delta_op (Weyl-corrected)', ...
                         sprintf('delta_op <= ||E||/(gap-||E||), at %d of %d pairs with gap > ||E||', ...
                                 sum(appl), height(T)), ...
                         dkok, 'true where applicable', dkok)]; %#ok<AGROW>
        else
            V = [V, vrec(['exp2/' f], 'Thm 2.3 Davis-Kahan is inapplicable here', ...
                         'pairs with gap > ||E||', ...
                         sprintf('0 of %d (median ||E||/gap = %.3g)', ...
                                 height(T), median(T.norm_E ./ T.spectral_gap)), ...
                         'the bound says nothing when ||E|| exceeds the gap', NaN)]; %#ok<AGROW>
        end
        V = [V, vrec(['exp2/' f], 'median chart term (2-norm | Frobenius)', ...
                     'median delta_chart | dF_chart', ...
                     sprintf('%.4g | %.4g', median(T.delta_chart), median(T.dF_chart)), ...
                     'ildl ~ 1, ichol ~ 0', NaN)]; %#ok<AGROW>
        V = [V, vrec(['exp2/' f], 'median operator term (2-norm | Frobenius)', ...
                     'median delta_op | dF_op', ...
                     sprintf('%.4g | %.4g', median(T.delta_op), median(T.dF_op)), ...
                     'the only term a stepper should pay', NaN)]; %#ok<AGROW>

        if strcmp(f, 'ildl')
            c1 = make_case(f, 1, opts);
            rk = all(T.rank_dA == 2 * c1.nC);
            V = [V, vrec('exp2/ildl', 'Thm 2.3 the operator change has rank exactly 2*nC', ...
                         sprintf('rank(dA) vs 2*nC=%d', 2*c1.nC), ...
                         mat2str(T.rank_dA'), 'equal', rk)]; %#ok<AGROW>
        end
    end

    %% ---- figure ----------------------------------------------------------
    fh = figure('Visible', 'off');
    tl = tiledlayout(fh, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    for fam = {'ildl', 'ichol'}
        T = D.(fam{1});
        nexttile(tl);
        bar(T.step, [T.dF_chart, T.dF_prec, T.dF_op], 'grouped');
        set(gca, 'YScale', 'log');  ylim([1e-6 4]);
        xlabel('step pair n \rightarrow n+1');
        ylabel('RMS-angle M_{n+1}-gap  d_F');
        title(sprintf('%s: the three terms', upper(fam{1})));
        % ILDL bars sit near 1 and ichol bars near 1e-3, so the free corner of
        % the (shared) axis is at opposite ends in the two panels.
        loc = 'northwest';
        if strcmp(fam{1}, 'ildl'), loc = 'southwest'; end
        legend({'\delta_{chart}  (C_n vs C_{n+1})', ...
                '\delta_{prec}  (M_n vs M_{n+1})', ...
                '\delta_{op}  (A_n vs A_{n+1})'}, 'Location', loc);
    end
    save_figure(fh, 'decomposition_bars');

    writetable(D.ildl,  fullfile(p.outDir, 'exp2_decomposition_ildl.csv'));
    writetable(D.ichol, fullfile(p.outDir, 'exp2_decomposition_ichol.csv'));
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
