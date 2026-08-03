function [V, D] = exp8_M_independence(opts)
%EXP8_M_INDEPENDENCE  How much does the deflation target depend on the metric?
%Tests Prop 5.3, a bound that is uniform in M -- and measures how weak it is.
%
%   [V, D] = EXP8_M_INDEPENDENCE(OPTS)
%
%   Hold A fixed and vary M as violently as the code plausibly could -- the
%   ILDL metric of this step, of a much later step, a block-Jacobi metric, and
%   the identity -- then measure how far the deflation target
%   U_k(A,M) moves from A's OWN smallest-|lambda| eigenspace N_k(A).
%
%   Prop 5.3.  If A u = lambda M u with ||u||_M = 1 then ||A u||_{M^-1} <= |lambda|,
%   so every unit vector of U_k(A,M) sits in the eta-approximate null space of A
%   alone with eta = |lambda_k| * ||M||_2.  Splitting u in A's eigenbasis at
%   index k gives |lambda_{k+1}(A)| * ||u_high|| <= ||A u||, hence
%
%       d( U_k(A,M) , N_k(A) )  <=  eta / |lambda_{k+1}(A)| ,
%
%   a bound that does not depend on WHICH M was used.
%
%   The experiment is what stops that from being oversold.  The bound is
%   uniform in M but WEAK: eta carries a factor ||M||_2, and for the ILDL
%   metric it is orders of magnitude above 1, i.e. vacuous.  The measurement
%   agrees -- the ILDL deflation target really does sit far from N_k(A).  So
%   the case for caching the physical subspace is NOT that the target is
%   metric-independent.  It is narrower and still decisive: the chart term is
%   the one part of the error that is Theta(1) at every step size and that
%   bookkeeping alone can remove exactly (Thm 5.1), while delta_prec and
%   delta_op are properties of the problem that no representation can undo.
%
%   Reports numbers only.  An earlier draft plotted this; the plot mostly
%   displayed how vacuous the bound is, which a table says more honestly and in
%   less space.  The figure budget is spent on exp7 instead, where the repair
%   is actually demonstrated.
%
%   See also: pencil_subspace, gap, exp7_transport.

    if nargin < 1 || isempty(opts), opts = struct(); end
    p = add_paths();
    k = getdef(opts, 'k', 20);

    V = struct([]);  D = struct();

    for fam = {'ildl', 'ichol'}
        f  = fam{1};
        c1 = make_case(f, 1, opts);
        cL = make_case(f, getdef(opts, 'npairs', 4) + 1, opts);
        n  = c1.n;
        A  = c1.A;

        dj = abs(diag(A));  dj(dj < 1e-8 * max(dj)) = 1;   % KKT has zero diagonal rows
        Ms = { 'identity',            speye(n)
               'block-Jacobi |diag|',  spdiags(dj, 0, n, n)
               'ILDL/ichol, this step', c1.M
               'ILDL/ichol, last step', cL.M };

        % A's own smallest-|lambda| eigenspace, and the gap that bounds everything
        [N, ~, iA] = pencil_subspace(A, speye(n), k, opts);
        lam_kp1 = abs(iA.lam_ext(min(k + 1, numel(iA.lam_ext))));

        rows = [];  labs = {};
        for j = 1:size(Ms, 1)
            M = (Ms{j, 2} + Ms{j, 2}') / 2;
            [U, lamP] = pencil_subspace(A, M, k, opts);
            [dg, gi]  = gap(U, N);
            eta   = max(abs(lamP)) * normest(M, 1e-3);
            bound = eta / lam_kp1;
            rows  = [rows; j, dg, gi.dF, eta, bound, dg <= bound]; %#ok<AGROW>
            labs{end+1} = Ms{j, 1}; %#ok<AGROW>
        end
        T = array2table(rows, 'VariableNames', ...
            {'variant', 'gap_to_Nk', 'dF_to_Nk', 'eta', 'bound', 'bound_holds'});
        T.metric = string(labs(:));
        D.(f) = T;
        disp(['[exp8/' f '] A fixed, M varied; N_k(A) is the reference']);  disp(T);

        okb = all(T.bound_holds > 0);
        tight = max(T.bound);
        V = [V, vrec(['exp8/' f], 'Prop 5.3 the uniform bound holds for every metric', ...
                     'd(U_k(A,M), N_k(A)) <= |lambda_k| ||M|| / |lambda_{k+1}(A)|', ...
                     okb, 'true for all four', okb)]; %#ok<AGROW>
        V = [V, vrec(['exp8/' f], 'Prop 5.3 is uniform in M but weak', ...
                     'largest bound value over the four metrics', tight, ...
                     'vacuous (>1) for the ILDL metric', NaN)]; %#ok<AGROW>
        V = [V, vrec(['exp8/' f], 'how far the target itself moves with the metric', ...
                     'd_F(U_k(A,M), N_k(A)) over four metrics', ...
                     sprintf('%.3g .. %.3g', min(T.dF_to_Nk), max(T.dF_to_Nk)), ...
                     'not negligible for ILDL', NaN)]; %#ok<AGROW>

        % The comparison that decides the algorithm: with A held fixed and only
        % the metric refreshed, how much of the damage is the CHART (removable
        % exactly, Thm 5.1) and how much is the TARGET moving (not removable)?
        U1c = pencil_subspace(A, c1.M, k, opts);
        U2c = pencil_subspace(A, cL.M, k, opts);
        Vc  = orth_trunc(c1.C' * U1c);
        [dchart, gc] = gap(Vc, cL.C' * c1.applyCtinv(Vc));
        [dphys,  gp] = gap(U1c, U2c);
        V = [V, vrec(['exp8/' f], 'A fixed, metric refreshed: chart vs target motion', ...
                     'd_F chart | d_F target', ...
                     sprintf('%.4g | %.4g', gc.dF, gp.dF), ...
                     'both real; only the chart one is removable', NaN)]; %#ok<AGROW>
        D.([f '_contrast']) = struct('delta_chart', dchart, 'physical_gap', dphys, ...
                                     'dF_chart', gc.dF, 'dF_phys', gp.dF);
        fprintf('[exp8/%s] A fixed, metric refreshed: d_F chart %.4g, d_F target %.4g\n', ...
                f, gc.dF, gp.dF);
    end

    writetable(D.ildl,  fullfile(p.outDir, 'exp8_M_independence_ildl.csv'));
    writetable(D.ichol, fullfile(p.outDir, 'exp8_M_independence_ichol.csv'));
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
