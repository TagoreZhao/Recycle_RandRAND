function T = run_all(opts)
%RUN_ALL  Run every experiment, write output/verdicts.csv and figures/*.png.
%
%   T = RUN_ALL()            fast mode (the default)
%   T = RUN_ALL(struct('FULL', true))
%
%   Section 6 of README.md is built from the table this returns.  A claim with
%   no row here is a claim with no evidence, and a FAIL row is reported, not
%   quietly dropped -- if an experiment contradicts a theorem, the theorem is
%   what gets fixed.
%
%   Fast mode: Stokes h0 = 0.15 (n = 760), SPD n = 600, k = 50, 4 step pairs.
%   FULL mode: Stokes h0 = 0.05 (n = 5840), SPD n = 2000, k = 100, 8 pairs --
%   minutes to an hour, and the conclusions do not change; the fast numbers are
%   what the document quotes.
%
%   See also: exp1_chart_and_invariance ... exp8_M_independence, test_kernel.

    if nargin < 1 || isempty(opts), opts = struct(); end
    p = add_paths();

    if getdef(opts, 'FULL', false)
        o = struct('h0', 0.05, 'nsteps', 10, 'n_gp', 2000, 'density', 0.02, ...
                   'k', 100, 'npairs', 8, 'tau', 0.5);
    else
        o = struct('h0', 0.15, 'nsteps', 8, 'n_gp', 600, 'density', 0.02, ...
                   'k', 50, 'npairs', 4, 'tau', 0.5);
    end
    fn = fieldnames(opts);
    for j = 1:numel(fn), o.(fn{j}) = opts.(fn{j}); end

    steps = { @() exp1_chart_and_invariance(o), 'exp1 chart and invariance'
              @() exp2_decomposition(o),        'exp2 three-term decomposition'
              @() exp3_regauge_only(o),         'exp3 regauge control'
              @() exp4_continuity_sweep(o),     'exp4 continuity sweep'
              @() exp5_bk_counterexample(o),    'exp5 discontinuity counterexample'
              @() exp6_cost_vs_angle(o),        'exp6 cost vs angle'
              @() exp7_transport(o),            'exp7 transport'
              @() exp8_M_independence(o),       'exp8 M-independence' };

    all_v = struct([]);
    for j = 1:size(steps, 1)
        fprintf('\n========== %s ==========\n', steps{j, 2});
        t = tic;
        v = steps{j, 1}();
        fprintf('---------- %s: %.1f s ----------\n', steps{j, 2}, toc(t));
        all_v = [all_v, v]; %#ok<AGROW>
    end

    T = struct2table(all_v);
    outFile = fullfile(p.outDir, 'verdicts.csv');
    writetable(T, outFile);

    nP = sum(T.verdict == "PASS");
    nF = sum(T.verdict == "FAIL");
    nR = sum(T.verdict == "REPORT");
    fprintf('\n===== verdicts: %d PASS, %d FAIL, %d REPORT =====\n', nP, nF, nR);
    if nF > 0
        disp(T(T.verdict == "FAIL", :));
    end
    fprintf('wrote %s\n', outFile);
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
