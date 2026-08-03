function V = exp1_chart_and_invariance(opts)
%EXP1_CHART_AND_INVARIANCE  The chart is an inner product, and the method is a
%function on the Grassmannian.  Tests Thm 1.1, 1.2, 1.3, 1.4.
%
%   V = EXP1_CHART_AND_INVARIANCE(OPTS)   returns verdict rows.
%
%   Thm 1.1  A u = lambda M u  <=>  Ahat (C'u) = lambda (C'u).
%   Thm 1.2  Phi_C is an isometry from the M-geometry to the Euclidean one:
%            d(C'X, C'Y) = d_M(X, Y).  Checked with gap_M and gap, which share
%            no code (gap_M never forms C, gap never forms M).
%   Thm 1.3  The deflation operator, and the MINRES iteration count, depend on
%            the coarse basis only through its span: V and V*Q agree exactly.
%   Thm 1.4  A CONSISTENT regauge (C -> C*Q, V -> Q'*V) leaves the iteration
%            count unchanged; freezing V while C is regauged does not.  This is
%            the whole bug in two lines of experiment.
%
%   Run for both families, so nothing here is an artifact of the indefinite case.
%
%   See also: gap, gap_M, pencil_subspace, chart_struct, run_all.

    if nargin < 1 || isempty(opts), opts = struct(); end
    add_paths();
    k    = getdef(opts, 'k',   20);
    tol  = getdef(opts, 'tol', 1e-8);
    mit  = getdef(opts, 'mit', 2000);
    rng(0);

    V = struct([]);
    for fam = {'ildl', 'ichol'}
        f  = fam{1};
        cs = make_case(f, 1, opts);
        n  = cs.n;
        Ahat = @(y) cs.applyCinv(cs.A * cs.applyCtinv(y));

        % ---- Thm 1.1: the chart maps pencil modes to split-operator modes ----
        [U, lam] = pencil_subspace(cs.A, cs.M, k, opts);
        Vh = cs.C' * U;
        AV = zeros(n, size(Vh, 2));
        for j = 1:size(Vh, 2), AV(:, j) = Ahat(Vh(:, j)); end
        inv_err = gap(Vh, AV);
        Q  = orth_trunc(Vh);
        R  = zeros(size(Q,2));
        for j = 1:size(Q,2), R(:, j) = Q' * Ahat(Q(:, j)); end
        lam_hat  = sort(real(eig((R + R')/2)));
        lam_err  = norm(sort(lam) - lam_hat, inf) / max(abs(lam));

        V = [V, vrec(['exp1/' f], 'Thm 1.1 chart maps pencil -> Ahat modes', ...
                     'd(span C''U, span Ahat C''U)', inv_err, '< 1e-8', inv_err < 1e-8)]; %#ok<AGROW>
        V = [V, vrec(['exp1/' f], 'Thm 1.1 eigenvalues are preserved', ...
                     'rel. max eigenvalue error', lam_err, '< 1e-6', lam_err < 1e-6)]; %#ok<AGROW>

        % ---- Thm 1.2: isometry, by two disjoint code paths -------------------
        X = randn(n, k);  Y = randn(n, k);
        iso = abs(gap_M(X, Y, cs.M) - gap(cs.C'*X, cs.C'*Y));
        V = [V, vrec(['exp1/' f], 'Thm 1.2 Phi_C is an M-to-Euclidean isometry', ...
                     '|d_M(X,Y) - d(C''X,C''Y)|', iso, '< 1e-8', iso < 1e-8)]; %#ok<AGROW>

        % ---- Thm 1.3: basis invariance of the method -------------------------
        % Tested at the OPERATOR level.  An integer iteration count is a
        % discontinuous function of a residual norm, so it can move by one when
        % the operator moves by roundoff; the claim is about the operator.
        b   = ones(n, 1);
        Vb  = orth_trunc(Vh);
        Qk  = orth(randn(size(Vb, 2)));
        % Each family gets ITS OWN coarse correction: the SPD form P built on
        % Ahat for 'ichol', the square-root form built on Ahat^2 for 'ildl'.
        [Pd1, E1] = coarse_correction(Vb,      Ahat, cs.tau, cs.defl_kind, 'handle');
        Pd2       = coarse_correction(Vb * Qk, Ahat, cs.tau, cs.defl_kind, 'handle');
        Z   = randn(n, 8);
        opdiff = norm(Pd1(Z) - Pd2(Z), 2) / norm(Z, 2);
        % The identity is exact; what is achievable numerically is limited by the
        % coarse matrix E, whose inverse (spd) or inverse square root (indef) is
        % formed by chol / eig.  For 'indef' E = V'Ahat^2 V is badly conditioned
        % BY CONSTRUCTION -- squaring squares the condition number -- and the
        % tolerance must say so.  For 'spd' E = V'Ahat V is not squared, so the
        % same tolerance rule buys far more room; the two cond(E) values below
        % are the price of the squaring, measured.
        cE  = cond(E1);
        tolI = 1e-12 * max(1, cE);
        V = [V, vrec(['exp1/' f], ...
                     sprintf('Thm 1.3 %s coarse correction depends on span(V) only', ...
                             cs.defl_kind), ...
                     '||P(V)Z - P(VQ)Z||/||Z||  (cond E)', ...
                     sprintf('%.3g  (%.2g)', opdiff, cE), ...
                     '< 1e-12 cond(E)', opdiff < tolI)]; %#ok<AGROW>

        [~, ~, ~, it_V]  = two_level_solve_local(cs.A, b, tol, mit, cs, Vb,      cs.tau);
        [~, ~, ~, it_VQ] = two_level_solve_local(cs.A, b, tol, mit, cs, Vb * Qk, cs.tau);
        V = [V, vrec(['exp1/' f], 'Thm 1.3 iteration count follows', ...
                     'its(V) vs its(V*Q)', sprintf('%d vs %d', it_V, it_VQ), ...
                     'within 1', abs(it_V - it_VQ) <= 1)]; %#ok<AGROW>

        % ---- Thm 1.4: gauge covariance vs the frozen-basis bug ---------------
        Qn  = orth(randn(n));
        Pg  = chart_struct(cs.C * Qn, cs.defl_kind, cs.tau);   % same M, other factor
        [~, ~, ~, it_cons] = two_level_solve_local( ...
                                 cs.A, b, tol, mit, Pg, orth_trunc(Qn' * Vb), cs.tau);
        [~, ~, ~, it_froz] = two_level_solve_local( ...
                                 cs.A, b, tol, mit, Pg, Vb, cs.tau);
        V = [V, vrec(['exp1/' f], 'Thm 1.4 consistent regauge changes nothing', ...
                     'its(C,V) vs its(CQ, Q''V)', sprintf('%d vs %d', it_V, it_cons), ...
                     'within 1', abs(it_V - it_cons) <= 1)]; %#ok<AGROW>
        V = [V, vrec(['exp1/' f], 'Thm 1.4 frozen V under the SAME regauge fails', ...
                     'its(CQ, V frozen)', sprintf('%d (vs %d)', it_froz, it_V), ...
                     '> consistent', it_froz > it_cons)]; %#ok<AGROW>

        fprintf(['[exp1/%s] n=%d  invariance %.1e  isometry %.1e  ' ...
                 'its V=%d VQ=%d  regauge consistent=%d frozen=%d\n'], ...
                f, n, inv_err, iso, it_V, it_VQ, it_cons, it_froz);
    end
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
