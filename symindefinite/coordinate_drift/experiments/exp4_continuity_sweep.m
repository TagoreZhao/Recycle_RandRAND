function [V, D] = exp4_continuity_sweep(opts)
%EXP4_CONTINUITY_SWEEP  Is the chart an analytic function of the operator?
%Tests Thm 3.1 (ichol is real-analytic) against Thm 3.2 (ILDL only piecewise).
%
%   [V, D] = EXP4_CONTINUITY_SWEEP(OPTS)
%
%   Interpolate toward the next step,
%
%       A(s) = A_1 + s * (A_2 - A_1),      s in [1e-8, 1],
%
%   rebuild the factor at each s, and measure the pure chart term
%
%       delta_chart(s) = d( span V_1 , span C(s)' C_1^-T V_1 ),
%
%   the physical subspace being held fixed throughout.  The slope of
%   log delta_chart against log ||dA||/||A|| is the whole question:
%
%     slope ~ 1  the map A -> C is differentiable, drift vanishes with the step
%               (Thm 3.1: fixed pattern, no pivoting, L_ii > 0);
%     slope ~ 0  the map jumps, and no time step is small enough to be safe
%               (Thm 3.2: AMD ordering, MC64 scaling and the Bunch-Kaufman
%                threshold are all piecewise constant in A).
%
%   Also runs the 'ict' variant of ichol, whose drop pattern IS value-dependent,
%   to check that a value-dependent pattern is not automatically as bad as
%   pivoting: dropping an entry below droptol perturbs C by O(droptol), whereas
%   reordering permutes it.
%
%   Writes figures/continuity_sweep.png -- the headline figure of the document.
%
%   See also: exp5_bk_counterexample, gap.

    if nargin < 1 || isempty(opts), opts = struct(); end
    p = add_paths();
    k  = getdef(opts, 'k', 20);
    ss = getdef(opts, 'svals', logspace(-8, 0, 17));

    runs = { 'ildl',  opts,                                            'ILDL (nofill)'
             'ichol', opts,                                            'ichol (nofill)'
             'ichol', setfield(opts, 'ichol_type', 'ict'),             'ichol (ict, droptol 1e-3)' }; %#ok<SFLD>

    D = struct('runs', {cell(size(runs, 1), 1)});
    V = struct([]);

    for r = 1:size(runs, 1)
        fam = runs{r, 1};  o = runs{r, 2};  lab = runs{r, 3};
        c1  = make_case(fam, 1, o);
        c2  = make_case(fam, 2, o);
        U1  = pencil_subspace(c1.A, c1.M, k, o);
        V1  = orth_trunc(c1.C' * U1);
        U1p = c1.applyCtinv(V1);                       % the physical subspace
        dA  = c2.A - c1.A;
        nA  = normest(c1.A, 1e-3);
        nd  = normest(dA,   1e-3);

        rows = zeros(numel(ss), 4);
        for t = 1:numel(ss)
            s  = ss(t);
            As = c1.A + s * dA;
            [C, ph] = factor_of(fam, As, o, c1);
            rows(t, :) = [s, s * nd / nA, gap(V1, C' * U1p), ph];
        end
        T = array2table(rows, 'VariableNames', {'s', 'relA', 'delta_chart', 'perm_frac'});
        D.runs{r} = struct('label', lab, 'family', fam, 'table', T);
        disp(['[exp4] ' lab]);  disp(T);

        good  = T.delta_chart > 1e-14 & T.relA < 1e-2;   % below saturation
        slope = NaN;
        if sum(good) >= 3
            c = polyfit(log10(T.relA(good)), log10(T.delta_chart(good)), 1);
            slope = c(1);
        end
        D.runs{r}.slope = slope;

        if strcmp(fam, 'ildl')
            ok = slope < 0.2 && min(T.delta_chart) > 0.5;
            V = [V, vrec('exp4/ildl', 'Thm 3.2 delta_chart does not vanish with ||dA||', ...
                         'log-log slope | min delta_chart', ...
                         sprintf('%.3f | %.3f', slope, min(T.delta_chart)), ...
                         'slope ~ 0, delta ~ 1', ok)]; %#ok<AGROW>
        else
            ok = abs(slope - 1) < 0.25;
            V = [V, vrec(['exp4/' lab], 'Thm 3.1 delta_chart is first order in ||dA||', ...
                         'log-log slope', slope, '~ 1', ok)]; %#ok<AGROW>
        end
    end

    % The separation, quoted at a MATCHED perturbation size.  The two families
    % sweep disjoint relA ranges (the KKT step is ~1e-1, the kernel step ~1e-5),
    % so comparing row 1 against row 1 would compare 1e-9 against 1e-13.  Take
    % the largest relA the ichol run reaches and read the ILDL run at its
    % nearest relA instead.
    Ti = D.runs{1}.table;  Th = D.runs{2}.table;
    [relA_h, jh]  = max(Th.relA);
    [~, ji]       = min(abs(log10(Ti.relA) - log10(relA_h)));
    relA_i        = Ti.relA(ji);
    ratio = Ti.delta_chart(ji) / max(Th.delta_chart(jh), realmin);
    V = [V, vrec('exp4', ...
                 sprintf('ILDL vs ichol chart drift at a matched ||dA||/||A|| (%.2g vs %.2g)', ...
                         relA_i, relA_h), ...
                 'delta_chart ratio ILDL / ichol', ratio, '>> 1', ratio > 1e3)];

    %% ---- figure ----------------------------------------------------------
    fh = figure('Visible', 'off');
    ax = axes(fh);  hold(ax, 'on');
    mk = {'-o', '-s', '-^'};
    for r = 1:numel(D.runs)
        T = D.runs{r}.table;
        loglog(ax, T.relA, max(T.delta_chart, 1e-16), mk{r}, 'LineWidth', 1.8, ...
               'MarkerSize', 6, 'DisplayName', ...
               sprintf('%s   (slope %.2f)', D.runs{r}.label, D.runs{r}.slope));
    end
    xr = [min(Th.relA) max(Th.relA)];
    loglog(ax, xr, xr / xr(1) * max(Th.delta_chart(1), 1e-16), 'k:', ...
           'LineWidth', 1.2, 'DisplayName', 'slope 1 (analytic)');
    set(ax, 'XScale', 'log', 'YScale', 'log');
    xlabel(ax, '||A(s) - A_1||_2 / ||A_1||_2');
    ylabel(ax, '\delta_{chart} = d(span V_1, span C(s)^T C_1^{-T} V_1)');
    title(ax, 'Chart drift under a shrinking operator perturbation');
    legend(ax, 'Location', 'southeast');  ylim(ax, [1e-16 3]);  hold(ax, 'off');
    save_figure(fh, 'continuity_sweep');

    for r = 1:numel(D.runs)
        writetable(D.runs{r}.table, ...
            fullfile(p.outDir, sprintf('exp4_sweep_%d.csv', r)));
    end
end

%==========================================================================
function [C, perm_frac] = factor_of(fam, As, o, cref)
%FACTOR_OF  Rebuild the chart factor of the perturbed operator.
    perm_frac = NaN;
    if strcmp(fam, 'ildl')
        P = src.precond.make_ildl_precond(As, struct('mode', 'nofill'));
        [~, iref] = ildl_coordinate_map(cref.P);
        [C, inf1]  = ildl_coordinate_map(P, iref);
        perm_frac = inf1.perm_hamming_frac;
    elseif strcmp(getdef(o, 'ichol_type', 'nofill'), 'nofill')
        C = ichol(As, struct('type', 'nofill'));
    else
        C = ichol(As, struct('type', 'ict', ...
                             'droptol', getdef(o, 'droptol', 1e-3), 'michol', 'on'));
    end
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
