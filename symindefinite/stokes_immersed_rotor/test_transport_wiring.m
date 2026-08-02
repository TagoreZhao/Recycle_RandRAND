% TEST_TRANSPORT_WIRING  Coordinate transport of the cached deflation bases, as
% wired into define_solver_list's production solver closures.
%
% THE DEFECT THIS PINS DOWN.  MINRES runs on the SPLIT operator
% Ahat_n = C_n^-1 K_n C_n^-T with yhat = C_n^T x, so a coarse basis V expressed in
% hat coordinates denotes the PHYSICAL subspace C_n^-T span(V) -- V is a
% REPRESENTATION, not a subspace.  The benchmark refreshes the ILDL factor every
% step (ILDL_PREC_REFRESH = 1) but freezes the basis (DEFLAT_PREC_REFRESH = Inf),
% and ldl re-derives the permutation p, the scaling S and the 1x1/2x2 pivot
% structure from K_n.  Reusing the same numbers therefore deflates a DIFFERENT
% physical subspace every step.  That is a change of coordinates on the ambient
% space, which moves spans -- not a change of basis within one, which would be
% harmless because deflation only sees the span.
%
% The fix (cached_basis / current_C in define_solver_list): cache the basis in
% PHYSICAL coordinates U = C^-T V and map it forward as V_n = orth(C_n^T U).
%
% Uses the REAL benchmark KKT sequence (build_stokes_sequence reproduces
% solve_stokes_immersed's assembly exactly), not a synthetic perturbation: the
% coupling block's SPARSITY PATTERN has to move for ldl to re-pivot, and a
% value-only perturbation leaves amd's permutation untouched.
%
% Run:  test_transport_wiring

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
repoRoot = fileparts(fileparts(thisFileDir));      % .../Recycle_RandRAND
addpath(repoRoot);
addpath(fullfile(repoRoot, 'symindefinite', 'linear_solves', ...
                 'subspace_recycle', 'kernel'));
add_recycle_paths();                                % kernel + subspace_capture
rng(1);

NSTEP = 3;
SMEIG = 40;      % small coarse space keeps the test fast; the defect is k-independent
RECK  = 30;
tolv  = 1e-8;

S = build_stokes_sequence(struct('case_name', 'bar_rotating', 'h0', 0.1, ...
                                 'nsteps', NSTEP, 'quiet', true));
n   = S.n;
mit = min(2000, n);
K   = arrayfun(@(j) seq_K(S, j), 1:NSTEP, 'UniformOutput', false);

fprintf('=== test_transport_wiring (n=%d, nC=%d, k=%d) ===\n', n, S.nC, SMEIG);
npass = 0;
capt  = @(Qt, Qc) subspace_capture_directed(Qt, Qc, [], ...
            struct('true_is_orth', true, 'comp_is_orth', true)).eigspace_err_2;

%% ---- T1: consecutive steps really do re-pivot -------------------------
% Guards every later check: if p did not move, naive reuse would be harmless and
% the rest of this test would be vacuous.
P1 = src.precond.make_ildl_precond(K{1}, struct('mode', 'nofill'));
P2 = src.precond.make_ildl_precond(K{2}, struct('mode', 'nofill'));
[C1, i1] = ildl_coordinate_map(P1);
[C2, i2] = ildl_coordinate_map(P2, i1);
assert(i2.perm_hamming_frac > 0.1, ...
       ['T1: the ldl permutation barely moved between steps 1 and 2 ' ...
        '(%.1f%%) -- this test cannot detect the defect'], ...
       100 * i2.perm_hamming_frac);
fprintf('  PASS T1: ldl re-pivots between steps (%.1f%% of p moved, nnzL ratio %.3f)\n', ...
        100 * i2.perm_hamming_frac, i2.nnzL_ratio);
npass = npass + 1;

%% ---- T2: transport preserves the PHYSICAL span, naive reuse does not ---
o  = struct('method', 'gaussian', 'sm_eig', SMEIG, 'lg_eig', 0, 'q', 2);
V1 = src.precond.build_deflation_V(K{1}, P1, o, decomposition(K{1}));
U1 = P1.applyCtinv(V1);                       % the physical subspace, once

[Q1, r1] = orth_trunc(U1);
assert(r1 == SMEIG, 'T2: the physical basis lost rank (%d of %d)', r1, SMEIG);

V2t = transport_V(U1, P2, C2);                % the fix
err_transported = capt(Q1, orth_trunc(P2.applyCtinv(V2t)));
err_naive       = capt(Q1, orth_trunc(P2.applyCtinv(V1)));   % what the code did

assert(err_transported < 1e-8, ...
       'T2: transport moved the physical span (err %.3e)', err_transported);
assert(err_naive > 1e-2, ...
       ['T2 NEGATIVE CONTROL FAILED: frozen hat-coordinate reuse preserved the ' ...
        'physical span (err %.3e), so there is nothing to fix'], err_naive);
fprintf(['  PASS T2: physical span preserved by transport (%.1e) and destroyed ' ...
         'by frozen reuse (%.3f)\n'], err_transported, err_naive);
npass = npass + 1;

%% ---- T3: transport is what makes the frozen basis worth anything -------
% Same operator, same smoother, same coarse-space dimension -- only the
% coordinate representation differs.
[~, ~, ~, it_frozen] = src.precond.two_level_split_solve(K{2}, S.b{2}, tolv, mit, P2, V1,  0.5);
[~, ~, ~, it_transp] = src.precond.two_level_split_solve(K{2}, S.b{2}, tolv, mit, P2, V2t, 0.5);
[~, ~, ~, it_none]   = src.precond.two_level_split_solve(K{2}, S.b{2}, tolv, mit, P2, [],  0.5);
fprintf('    step 2 MINRES: ILDL only %d | frozen V %d | transported V %d\n', ...
        it_none, it_frozen, it_transp);
assert(it_transp < it_frozen, ...
       'T3: transported basis is not faster than the frozen one (%d vs %d its)', ...
       it_transp, it_frozen);
assert(it_transp < it_none, ...
       'T3: transported deflation does not beat no deflation at all (%d vs %d its)', ...
       it_transp, it_none);
fprintf('  PASS T3: transported %d its vs frozen %d vs ILDL-only %d\n', ...
        it_transp, it_frozen, it_none);
npass = npass + 1;

%% ---- T4: the production closures cache PHYSICALLY and memo per step ----
params = struct('DEFLAT_SM_EIG', SMEIG, 'DEFLAT_RECYCLE_K', RECK, ...
                'DEFLAT_TAU', 0.5, 'DEFLAT_Q', 2, ...
                'ILDL_PREC_REFRESH', 1, 'DEFLAT_PREC_REFRESH', Inf, ...
                'DINVERSE_PREC_REFRESH', Inf);
solvers = define_solver_list(params);
skeys   = cellfun(@(s) s.key, solvers, 'UniformOutput', false);
sg      = solvers{strcmp(skeys, 'two_level_gaussian')};
sk      = solvers{strcmp(skeys, 'two_level_krylov')};

% One shared pc, exactly as +src/+stokes/solve_stokes_immersed drives it, so the
% two solvers share the ildl / dinv / V_gaussian cache keys.
pc = struct('step', 1, 'cache', containers.Map('KeyType', 'char', 'ValueType', 'any'));
its_g = zeros(NSTEP, 1);  its_k = zeros(NSTEP, 1);
for j = 1:NSTEP
    pc.step = j;
    [~, fg, ~, its_g(j)] = sg.solve(K{j}, S.b{j}, tolv, mit, pc);
    [~, fk, ~, its_k(j)] = sk.solve(K{j}, S.b{j}, tolv, mit, pc);
    assert(fg == 0 && fk == 0, 'T4: step %d did not converge (flags %d/%d)', j, fg, fk);
end

e = pc.cache('V_gaussian');
assert(isfield(e, 'U') && isfield(e, 'V'), 'T4: cache entry lost its physical basis');
assert(e.step == 1, 'T4: U was rebuilt at step %d, but DEFLAT_PREC_REFRESH = Inf', e.step);
assert(e.hstep == NSTEP, 'T4: per-step memo stamped %d, expected %d', e.hstep, NSTEP);
assert(size(e.U, 1) == n && size(e.V, 1) == n, 'T4: cached bases have the wrong shape');
assert(norm(e.V' * e.V - eye(size(e.V, 2)), 'fro') < 1e-10, ...
       'T4: the transported basis handed to deflation_Psqrt_apply is not orthonormal');

% The basis actually used at the last step still denotes the step-1 subspace.
Plast = pc.cache('ildl_nofill').val;
err_wired = capt(orth_trunc(e.U), orth_trunc(Plast.applyCtinv(e.V)));
assert(err_wired < 1e-8, ...
       'T4: the wired-in basis drifted from its physical subspace (err %.3e)', err_wired);
fprintf(['  PASS T4: cache holds U (built step %d) + step-%d memo, orthonormal, ' ...
         'physical drift %.1e\n'], e.step, e.hstep, err_wired);
npass = npass + 1;

%% ---- T5: the recycled Krylov block is carried physically too -----------
assert(isKey(pc.cache, 'krylov_U'), 'T5: the harvest was not cached');
ek = pc.cache('krylov_U');
assert(~isfield(ek, 'W'), 'T5: stale hat-coordinate field krylov_W is still written');
assert(ek.step == NSTEP && size(ek.U, 1) == n, 'T5: harvest has the wrong stamp/shape');
assert(size(ek.U, 2) >= min(RECK, its_k(NSTEP)), ...
       'T5: harvested %d columns, expected at least %d', ...
       size(ek.U, 2), min(RECK, its_k(NSTEP)));
assert(all(isfinite(ek.U(:))), 'T5: harvest contains non-finite entries');

% Step 1 is identical by construction (nothing recycled yet) -- the controlled
% comparison the two entries exist for.
assert(its_k(1) == its_g(1), ...
       'T5: step 1 must be identical (krylov %d vs gaussian %d)', its_k(1), its_g(1));
assert(all(its_k(2:end) <= its_g(2:end)), ...
       'T5: recycling made a later step slower (krylov %s vs gaussian %s)', ...
       mat2str(its_k(:)'), mat2str(its_g(:)'));
fprintf('  PASS T5: harvest carried physically; iters gaussian %s vs krylov %s\n', ...
        mat2str(its_g(:)'), mat2str(its_k(:)'));
npass = npass + 1;

fprintf('=== test_transport_wiring: %d checks passed ===\n', npass);
