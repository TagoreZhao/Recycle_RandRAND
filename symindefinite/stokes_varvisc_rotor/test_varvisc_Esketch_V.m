%TEST_VARVISC_ESKETCH_V  Unit tests for the symmetric E-sketch deflation arm.
%
%   Run:  cd symindefinite/stokes_varvisc_rotor; test_varvisc_Esketch_V
%
%   THE METHOD UNDER TEST.  With A_1 = K_ref factored ONCE (exact split factor
%   C_ref, C_ref C_ref' = |A_1|) and A_2 = K_n the current step,
%
%       E = C_ref^{-1} (A_2 - A_1) C_ref^{-T}     (SYMMETRIC, ref-hat coords)
%       Y = orth-reorthed E^{2q+1} Omega,  Omega = randn(n, k)
%       U = C_ref^{-T} Y                          (physical subspace)
%       V = orth(C_n' U)                          (current-step hat coords)
%
%   and V is handed to src.precond.two_level_split_solve.  E is the operator
%   that actually perturbs the split system MINRES runs on: with the exact
%   reference factor, Ahat = C_ref^{-1} A_2 C_ref^{-T} = sign(D) + E.  Its
%   dominant directions are EIGENVECTORS (E is symmetric), and eigenspaces --
%   unlike the singular spaces of the old physical-coordinate D-sketch -- ARE
%   transported exactly by the similarity map C_n^T C_ref^{-T}.  T1/T1b pin
%   that claim; everything is stated over SPANS (projector residuals), never
%   over a chosen basis.
%
%   FIXTURE.  A synthetic saddle-point pair, NOT the benchmark assembly: the
%   update A_2 - A_1 = W diag(g) W' has exactly known range and an eigenvalue
%   cliff (g drops by 1e8 after the first 10 modes), so the dominant eigenspace
%   of E is unambiguous and the transported reference spaces are computable
%   explicitly at this size.  Indefiniteness of A_1 exercises the 2x2 pivots in
%   |D|^{1/2}.
%
%   T3 is the falsification control: dK exactly zero must return an empty V and
%   the registry arm must degrade to plain ILDL with IDENTICAL counts.
%
%   See also: varvisc_build_Esketch_V, varvisc_esketch_ref_context,
%             varvisc_define_solver_list, src.precond.two_level_split_solve.

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(here)));          % repo root, for src.*
addpath(here);
addpath(fullfile(fileparts(here), 'linear_solves', 'subspace_recycle', 'kernel'));
rng(0);

np = 0; nf = 0;
fprintf('=== test_varvisc_Esketch_V ===\n');

%% ----------------------------------------------------------- fixture ----
nU  = 300;  nP = 100;  n = nU + nP;
Auu = sprandsym(nU, 0.02, 0.1, 1);            % SPD velocity-like block
Bp  = sprandn(nP, nU, 0.05);
A1  = [Auu, Bp'; Bp, sparse(nP, nP)];
A1  = (A1 + A1') / 2;                         % exactly symmetric, indefinite

r     = 30;  kcliff = 10;
Wg    = sprandn(n, r, 0.08);
g     = [linspace(2, 1, kcliff), 1e-8 * ones(1, r - kcliff)];   % 1e8 cliff
dKt   = Wg * spdiags(g(:), 0, r, r) * Wg';
dKt   = (dKt + dKt') / 2;
dKt   = dKt * (0.25 * norm(A1, 'fro') / norm(dKt, 'fro'));
A2    = A1 + dKt;

Pref = src.precond.make_ildl_precond(A1, struct('mode', 'exact'));
Pn   = src.precond.make_ildl_precond(A2, struct('mode', 'nofill'));
Cref = ildl_coordinate_map(Pref);
Cn   = ildl_coordinate_map(Pn);
fprintf('  fixture: n=%d, rank(dK)=%d (cliff after %d), ||dK||_F/||A1||_F=%.2f\n', ...
        n, r, kcliff, norm(dKt, 'fro') / norm(A1, 'fro'));

ctx = varvisc_esketch_ref_context(A1);

%% -------------------------------------------- explicit E, small-n truth ----
Ce = full(Cref);
E  = Ce \ full(A2 - A1) / Ce';                % E = C_ref^{-1} dK C_ref^{-T}

% T0  the target really is symmetric (this is what makes an eigen-sketch valid).
[np, nf] = chk(np, nf, sprintf('T0  explicit E is symmetric (asym %.2e)', ...
                               norm(E - E', 'fro') / norm(E, 'fro')), ...
    norm(E - E', 'fro') / norm(E, 'fro') < 1e-12);

% T0b the context factors really give C_ref C_ref' = |A_1| (batched handles are
%     load-bearing downstream, so they are checked, not trusted).
Rb = randn(n, 5);
Xb = ctx.P.applyMinv(Rb);                     % M^{-1} R, batched
[np, nf] = chk(np, nf, 'T0b context handles invert M = C_ref C_ref'' (batched)', ...
    norm(Ce * (Ce' * Xb) - Rb, 'fro') / norm(Rb, 'fro') < 1e-8);

%% ------------------------------ T1: dominant eigenspace, correct transport ----
% Reference: top-kcliff eigenvectors of the explicit E (ref-hat coords), pushed
% through the similarity transport C_n^T C_ref^{-T} into current hat coords.
[Wev, Lam] = eig((E + E') / 2);
[~, ord]   = sort(abs(diag(Lam)), 'descend');
Wtop       = Wev(:, ord(1:kcliff));
Vref       = orth_trunc(Cn' * (Ce' \ Wtop));

o1 = struct('k', kcliff, 'q', 2, 'reorth', true, 'Cn', Cn);
[V1, i1, Y1] = varvisc_build_Esketch_V(ctx, A2, Pn, o1);
gap1 = max(subspace_residual(Vref, V1), subspace_residual(V1, Vref));
[np, nf] = chk(np, nf, sprintf(['T1  V matches the transported dominant eigenspace ' ...
                                'of E (gap %.2e)'], gap1), gap1 < 1e-6);

% T1b transport preserves the PHYSICAL span exactly: the hat-coordinate V must
%     denote the same physical subspace as U = C_ref^{-T} Y.
Uphys = Pref.applyCtinv(Y1);
Vphys = Pn.applyCtinv(V1);
gap1b = max(subspace_residual(orth_trunc(Uphys), Vphys), ...
            subspace_residual(orth_trunc(Vphys), Uphys));
[np, nf] = chk(np, nf, sprintf('T1b V and C_ref^{-T}Y span the same physical space (gap %.2e)', ...
                               gap1b), gap1b < 1e-8);

%% --------------------------------------- T2: contract of the returned V ----
[np, nf] = chk(np, nf, 'T2  V real, orthonormal, no wider than k', ...
    isreal(V1) && size(V1, 2) <= o1.k && ...
    norm(V1' * V1 - eye(size(V1, 2)), 'fro') < 1e-12);

% T2b operation counts follow the (2q+1)*k contract.
[np, nf] = chk(np, nf, sprintf('T2b (2q+1)*k = %d E-applies, %d dK matvecs', ...
                               i1.n_E_applies, i1.n_dK_matvecs), ...
    i1.n_E_applies == (2 * o1.q + 1) * o1.k && i1.n_dK_matvecs == i1.n_E_applies);

% T2c deterministic under a fixed rng seed (a function of the RNG state only).
rng(7);  Va = varvisc_build_Esketch_V(ctx, A2, Pn, o1);
rng(7);  Vb = varvisc_build_Esketch_V(ctx, A2, Pn, o1);
[np, nf] = chk(np, nf, 'T2c deterministic under a fixed rng seed', isequal(Va, Vb));

%% ----------------------------------- T3: falsification control, dK = 0 ----
[V0, i0] = varvisc_build_Esketch_V(ctx, A1, Pn, o1);
[np, nf] = chk(np, nf, sprintf('T3  dK = 0 -> empty V (%d cols, nnz(dK) %d, %d applies)', ...
                               i0.ncols, i0.dK_nnz, i0.n_E_applies), ...
    isempty(V0) && i0.ncols == 0 && i0.dK_nnz == 0 && i0.n_E_applies == 0);

%% --------------------------- T4: full-range recovery and rank truncation ----
% rank(E) = rank(dK) = r exactly; k above it must recover the WHOLE range and
% report the truncation.  range(E) transported = C_n^T |A_1|^{-1} range(W).
kfull = r + 5;
[V4, i4] = varvisc_build_Esketch_V(ctx, A2, Pn, ...
               struct('k', kfull, 'q', 2, 'reorth', true, 'Cn', Cn));
Qref = orth_trunc(Cn' * Pref.applyMinv(full(Wg)));
gap4 = max(subspace_residual(Qref, V4), subspace_residual(V4, Qref));
[np, nf] = chk(np, nf, sprintf('T4  k=%d >= rank recovers the full transported range (gap %.2e)', ...
                               kfull, gap4), gap4 < 1e-6);
[np, nf] = chk(np, nf, sprintf('T4b rank truncation reported (%d raw -> %d cols, drop %d)', ...
                               i4.ncols_raw, i4.ncols, i4.rank_drop), ...
    i4.ncols == r && i4.rank_drop == kfull - r);

%% ------------------------- T5: deflation effect on the split operator ----
% With the exact reference factor as the smoother, Ahat = sign(D) + E exactly;
% deflating the full range of E must beat the smoother alone outright.
rng(1);
b = randn(n, 1);
TOLV = 1e-10;  mit = n;  tau = 0.5;
Vd = varvisc_build_Esketch_V(ctx, A2, Pref, ...
         struct('k', kfull, 'q', 2, 'reorth', true, 'Cn', Cref));
[~, fl0, ~, it_none] = src.precond.two_level_split_solve(A2, b, TOLV, mit, Pref, [], tau);
[~, fl1, ~, it_defl] = src.precond.two_level_split_solve(A2, b, TOLV, mit, Pref, Vd, tau);
[np, nf] = chk(np, nf, sprintf('T5  full-range deflation beats the smoother alone (%d < %d its)', ...
                               it_defl, it_none), ...
    fl0 == 0 && fl1 == 0 && it_defl < it_none);

%% ----------------------------------------------- registry closure tests ----
% droptol mode: a no-fill pattern on this RANDOM sparsity fixture drops nearly
% everything and neither arm converges -- the closure test needs a smoother
% that smooths, and mode is a registry knob, not part of the contract under test.
params  = struct('DEFLAT_SM_EIG', 10, 'SKETCH_OVERSAMPLE', 2, 'DEFLAT_Q', 2, ...
                 'ILDL_PREC_REFRESH', 1, 'ESKETCH_REF_REFRESH', Inf, ...
                 'DEFLAT_TAU', 0.5, 'ILDL_MODE', 'droptol', 'ILDL_DROPTOL', 1e-4);
solvers = varvisc_define_solver_list(params);
skeys   = cellfun(@(s) s.key, solvers, 'UniformOutput', false);

% T6  registered, registered LAST (the featured slot), and the D-sketch arm is
%     gone -- this arm REPLACES it.
[np, nf] = chk(np, nf, 'T6  two_level_esketch registered LAST, lowrank arm removed', ...
    strcmp(skeys{end}, 'two_level_esketch') && ...
    ~any(strcmp(skeys, 'two_level_lowrank_sketch')));

sl = solvers{strcmp(skeys, 'two_level_esketch')};
si = solvers{strcmp(skeys, 'ildl_nofill')};

% ONE shared pc, exactly as solve_stokes_varvisc drives it, so both arms use the
% same ILDL smoother object and differ only by the coarse space.
pc = struct('step', 1, ...
            'cache', containers.Map('KeyType', 'char', 'ValueType', 'any'));
KK = {A1, A2};
its_e = zeros(2, 1);  fls = zeros(2, 1);  err_e = zeros(2, 1);
its_i = zeros(2, 1);  err_i = zeros(2, 1);
for j = 1:2
    pc.step = j;  pc.K = KK{j};
    [xe, fls(j), ~, its_e(j)] = sl.solve(KK{j}, b, 1e-8, 4 * n, pc);
    err_e(j) = norm(b - KK{j} * xe) / norm(b);
    [xi, ~, ~, its_i(j)]      = si.solve(KK{j}, b, 1e-8, 4 * n, pc);
    err_i(j) = norm(b - KK{j} * xi) / norm(b);
end
fprintf('    iters  esketch %s  vs ILDL %s\n', mat2str(its_e(:)'), mat2str(its_i(:)'));

% T6b engine contract: scalar iteration counts, converged, accuracy on par with
%     the arm it shares a smoother with (split relres is the smoother's metric).
[np, nf] = chk(np, nf, 'T6b scalar its, converged, accuracy on par with ILDL', ...
    all(arrayfun(@isscalar, its_e)) && all(fls == 0) && ...
    max(err_e) <= 10 * max(err_i));

% T6c step 1 is the controlled comparison: dK = 0 there, so the arm IS the ILDL
%     arm and the counts must be identical, not merely close.
[np, nf] = chk(np, nf, sprintf('T6c step 1 (dK = 0) identical to ILDL (%d its)', its_e(1)), ...
    its_e(1) == its_i(1));

% T6d the recycled object is the reference FACTORIZATION: built at step 1 and
%     frozen; the effective coarse width is stashed for post-run reporting.
e  = pc.cache('esketch_ref');
ei = pc.cache('esketch_info');
[np, nf] = chk(np, nf, sprintf('T6d reference frozen at step 1; info stashed (k=%d, ncols=%d)', ...
                               ei.val.k, ei.val.ncols), ...
    e.step == 1 && e.val.n == n && ei.step == 2 && ei.val.k == 20);

%% ------------------------------------------------------------ summary ----
fprintf('\n%d passed, %d FAILED\n', np, nf);
if nf > 0, error('test_varvisc_Esketch_V:failures', '%d checks failed', nf); end

%==========================================================================
function s = subspace_residual(Qbase, Y)
%SUBSPACE_RESIDUAL  sin of the largest principal angle of span(Y) into span(Qbase).
% Basis-invariant: depends on the two SPANS only, not on how either is written.
    Q = orth_trunc(Qbase);
    Z = orth_trunc(Y);
    if isempty(Z), s = 0; return; end
    if isempty(Q), s = 1; return; end
    s = norm(Z - Q * (Q' * Z));
end

%==========================================================================
function [np, nf] = chk(np, nf, name, cond)
    if cond
        np = np + 1;  fprintf('  ok   %s\n', name);
    else
        nf = nf + 1;  fprintf('  FAIL %s\n', name);
    end
end
