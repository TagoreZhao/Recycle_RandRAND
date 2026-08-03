function [V, D] = exp5_bk_counterexample(opts)
%EXP5_BK_COUNTEREXAMPLE  The ILDL chart is discontinuous, three ways.
%Tests Prop 3.3 (the minimal Bunch-Kaufman flip, proved), Obs 3.4 (the pattern
%flip on the production matrix) and Obs 3.5 (|D|^{1/2}).  Obs 3.4/3.5 are
%measurements, not theorems -- nothing about them is proved.
%
%   [V, D] = EXP5_BK_COUNTEREXAMPLE(OPTS)
%
%   Part A -- the production discontinuity, on the real KKT matrix.  Two
%   perturbations of the SAME size eta are applied to A:
%
%     (i)  value-only: scale an entry that already exists, so pattern(A) is
%          unchanged.  AMD returns the same ordering and C moves smoothly.
%     (ii) pattern-changing: place eta at a position that was structurally
%          zero.  AMD reorders, and C jumps.
%
%   Sweeping eta down to 1e-12 separates the two completely: (i) has
%   delta_chart proportional to eta, (ii) has delta_chart independent of eta.
%   That is Thm 3.2 in one table -- the chart drift does not vanish with the
%   perturbation, so no time step is small enough to make a frozen basis safe.
%   This is the mechanism that fires every step in the immersed-rotor problem,
%   where Lagrange points crossing triangle edges change the coupling pattern.
%
%   Part B -- the minimal analytic example.  On A(e) = [e 1; 1 0] the textbook
%   Bunch-Kaufman rule takes a 1x1 pivot when |a11| >= alpha*max_j|a_j1|, i.e.
%   e >= alpha = (1+sqrt(17))/8 = 0.640388..., and a 2x2 pivot otherwise, so the
%   factor has two different one-sided limits at e = alpha:
%
%       C_+ = [ sqrt(alpha)  0 ; 1/sqrt(alpha)  1/sqrt(alpha) ],
%       C_- = |A(alpha)|^{1/2},
%
%   both exact.  Verified against MATLAB's unscaled ldl (the 3-output form).
%   The production path adds MC64 scaling before pivoting, which MOVES the
%   switching set but does not remove it -- Part A is the production-path
%   evidence that a switching set is crossed in practice.
%
%   Part C -- one extra off-diagonal entry reorders AMD, in isolation.
%
%   Part D -- |D|^{1/2}.  For A(d) = diag(1,-d), C = diag(1, sqrt d), so
%   kappa_2(C) = |d|^{-1/2} -> infinity as a pivot approaches zero, inflating
%   Lemma 2.2's bound by exactly that factor.  Structurally absent for ichol.
%
%   See also: ildl_coordinate_map, src.precond.make_ildl_precond, gap.

    if nargin < 1 || isempty(opts), opts = struct(); end
    add_paths();
    V = struct([]);  D = struct();
    k     = getdef(opts, 'k', 20);
    alpha = (1 + sqrt(17)) / 8;

    %% ---- Part A: production path, value-only vs pattern-changing ---------
    cs = make_case('ildl', 1, opts);
    A0 = cs.A;
    U0 = pencil_subspace(A0, cs.M, k, opts);
    V0 = orth_trunc(cs.C' * U0);
    [~, i0] = ildl_coordinate_map(cs.P);

    [iz, jz] = first_structural_zero(A0);          % a structurally zero position
    rng(1);
    R  = sprandsym(A0);                            % random values ON pattern(A0)
    R  = R / normest_q(R);

    etas = logspace(-2, -12, 6);
    rows = zeros(numel(etas), 5);
    nA   = normest_q(A0);
    for t = 1:numel(etas)
        eta = etas(t);
        Av  = A0 + (eta * nA) * R;                 % ||dA||/||A|| = eta, pattern kept
        Ap  = A0;  Ap(iz, jz) = eta;  Ap(jz, iz) = eta;   % one new position
        rows(t, :) = [eta, chart_probe(Av, V0, cs, i0), chart_probe(Ap, V0, cs, i0)];
    end
    D.pattern = array2table(rows, 'VariableNames', ...
        {'eta', 'value_only_delta', 'value_only_permfrac', ...
         'pattern_delta', 'pattern_permfrac'});
    disp('[exp5A] value-only vs pattern-changing perturbation of the same size');
    disp(D.pattern);

    sv = polyfit(log10(rows(:,1)), log10(max(rows(:,2), 1e-16)), 1);
    V = [V, vrec('exp5A', 'Thm 3.2 value-only perturbation: chart drift is smooth', ...
                 'log-log slope of delta_chart vs eta', sv(1), '~ 1', sv(1) > 0.5)];
    V = [V, vrec('exp5A', 'Thm 3.2 pattern change: chart drift does not vanish with eta', ...
                 'min delta_chart over eta down to 1e-12', min(rows(:, 4)), ...
                 '> 0.5 (flat)', min(rows(:, 4)) > 0.5)];
    V = [V, vrec('exp5A', 'Obs 3.4 an eta=1e-12 entry is enough', ...
                 'delta_chart at eta=1e-12 | permutation moved', ...
                 sprintf('%.3f | %.0f%%', rows(end, 4), 100*rows(end, 5)), ...
                 'delta ~ 1', rows(end, 4) > 0.5)];
    fprintf('[exp5A] eta=1e-12: value-only delta=%.2e, pattern-change delta=%.4f\n', ...
            rows(end, 2), rows(end, 4));

    %% ---- Part B: the minimal analytic Bunch-Kaufman example --------------
    ee = alpha + [-1 1] * 1e-6;
    Cs = cell(1, 2);  n2 = zeros(1, 2);
    for j = 1:2
        [L, Dd] = ldl(full([ee(j) 1; 1 0]));       % unscaled, textbook BK
        [Dh, ~] = abs_half(Dd);
        Cs{j}   = L * Dh;
        n2(j)   = nnz(tril(Dd, -1));
    end
    Cp = [sqrt(alpha) 0; 1/sqrt(alpha) 1/sqrt(alpha)];
    Aa = [alpha 1; 1 0];  [Qa, La] = eig(Aa);
    Cm = Qa * diag(sqrt(abs(diag(La)))) * Qa';

    err = max(norm(Cs{2} - Cp, 'fro') / norm(Cp, 'fro'), ...
              norm(Cs{1}*Cs{1}' - Cm*Cm', 'fro') / norm(Cm*Cm', 'fro'));
    Tb  = Cs{2}' / Cs{1}';
    db  = gap([1; 0], Tb * [1; 0]);
    D.bk = struct('alpha', alpha, 'e', ee, 'n2x2', n2, 'C_minus', Cs{1}, ...
                  'C_plus', Cs{2}, 'C_plus_closed', Cp, 'C_minus_closed', Cm, ...
                  'delta_chart', db, 'relC', norm(Cs{2}-Cs{1},2)/norm(Cs{1},2));

    V = [V, vrec('exp5B', 'Prop 3.3 dense ldl flips 2x2 -> 1x1 exactly at alpha', ...
                 '#2x2 pivots below | above alpha', sprintf('%d | %d', n2(1), n2(2)), ...
                 '1 | 0', n2(1) == 1 && n2(2) == 0)];
    % the probes sit 1e-6 either side of alpha, so agreement is expected to O(1e-6)
    V = [V, vrec('exp5B', 'Prop 3.3 the closed-form branches match dense ldl', ...
                 'rel. error vs C_+ and M_-', err, '< 1e-5 (probe offset 1e-6)', ...
                 err < 1e-5)];
    V = [V, vrec('exp5B', 'Prop 3.3 O(1e-6) perturbation, O(1) chart jump', ...
                 'delta_chart across the switching point', db, '> 0.1', db > 0.1)];
    fprintf('[exp5B] BK flip at alpha=%.6f: pivots %d|%d, delta_chart=%.4f\n', ...
            alpha, n2(1), n2(2), db);

    %% ---- Part C: the AMD pattern flip in isolation -----------------------
    B0 = spdiags(repmat([1 4 1], 9, 1), -1:1, 9, 9);
    B0(1, 6) = 1;  B0(6, 1) = 1;
    B1 = B0;  B1(2, 9) = 1;  B1(9, 2) = 1;
    p0 = amd(B0);  p1 = amd(B1);
    ham = sum(p0(:) ~= p1(:)) / numel(p0);
    D.amd = struct('p0', p0, 'p1', p1, 'hamming_frac', ham);
    V = [V, vrec('exp5C', 'Obs 3.4 one extra entry reorders AMD', ...
                 'fraction of the permutation that moves', ham, '> 0', ham > 0)];

    %% ---- Part D: |D|^{1/2} and the conditioning of C ---------------------
    dd = logspace(-1, -10, 10);  kap = zeros(size(dd));
    for j = 1:numel(dd)
        P = src.precond.make_ildl_precond(sparse([1 0; 0 -dd(j)]), struct('mode','nofill'));
        kap(j) = cond(full(ildl_coordinate_map(P)));
    end
    sl = polyfit(log10(dd), log10(kap), 1);
    D.pivot = struct('d', dd, 'kappa', kap, 'slope', sl(1));
    V = [V, vrec('exp5D', 'Obs 3.5 kappa_2(C) ~ |d_min|^{-1/2}', ...
                 'log-log slope', sl(1), '-0.5', abs(sl(1) + 0.5) < 0.02)];
    fprintf('[exp5D] kappa_2(C) vs vanishing pivot: slope %.4f (theory -0.5)\n', sl(1));
end

%==========================================================================
function out = chart_probe(A, V0, cs0, i0)
%CHART_PROBE  [delta_chart, permutation drift] for a perturbed operator.
%   The physical subspace is held fixed at C_0^-T span(V0); only the factor is
%   rebuilt.  delta_chart is therefore the pure chart term of Thm 2.1.
    P = src.precond.make_ildl_precond(A, struct('mode', 'nofill'));
    [C, i1] = ildl_coordinate_map(P, i0);
    out = [gap(V0, C' * cs0.applyCtinv(V0)), i1.perm_hamming_frac];
end

function v = normest_q(A)
%NORMEST_Q  normest with a loose tolerance -- these are scale factors for a
%   perturbation, not quantities anyone reports, so 1e-3 is ample and avoids
%   normest's non-convergence warning on the KKT matrix.
    v = normest(A, 1e-3);
end

function [i, j] = first_structural_zero(A)
%FIRST_STRUCTURAL_ZERO  A structurally zero off-diagonal position, well inside
%   the matrix so the added entry genuinely perturbs the elimination tree.
    n = size(A, 1);
    for i = round(n/3):n
        r = A(i, :);
        for j = 1:i-1
            if r(j) == 0, return; end
        end
    end
    error('exp5:noZero', 'no structural zero found');
end

function [Dh, Di] = abs_half(Dd)
%ABS_HALF  |D|^{1/2} and |D|^{-1/2} for a 1x1/2x2 block-diagonal D.
    [Q, L] = eig(full(Dd));
    a  = max(abs(diag(L)), 1e-14);
    Dh = Q * diag(sqrt(a))   * Q';
    Di = Q * diag(1./sqrt(a)) * Q';
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
