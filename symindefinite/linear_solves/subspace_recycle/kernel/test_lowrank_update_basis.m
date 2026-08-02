% TEST_LOWRANK_UPDATE_BASIS  Unit tests for the proposed missing-component block.
%
% T2 is the central one: it verifies the span identity
%     K_n^-1 * range([dC, Sel])  ==  K_ref^-1 * range([dC, Sel])
% which is what makes a FROZEN factorization an exact source for the update and
% removes any dependence on a well-conditioned Woodbury capacitance.
%
% T5 is the scientific claim rather than an implementation check: with the
% coordinates held fixed, does [V_ref, What] capture the step-n small eigenspace
% that V_ref alone misses?  It is asserted loosely here (the quantitative answer
% is run_eigenspace_motion's job) but the numbers are printed.
%
% Run:  test_lowrank_update_basis

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
add_recycle_paths();
rng(1);

S = build_stokes_sequence(struct('case_name', 'bar_rotating', 'h0', 0.15, ...
                                 'nsteps', 4, 'use_cache', false, 'quiet', true));
n = S.n;  m = S.nC;  NSTEP = 4;
K1 = seq_K(S, 1);  Kn = seq_K(S, NSTEP);
P1 = src.precond.make_ildl_precond(K1, struct('mode', 'nofill'));
C1 = ildl_coordinate_map(P1);            % coordinates FROZEN at step 1 throughout

fprintf('=== test_lowrank_update_basis (n=%d, nC=%d, step %d) ===\n', n, m, NSTEP);
npass = 0;
ctx = [];

%% ---- T1: the Woodbury identity K_n^-1 U = (K_ref^-1 U)(Cap \ B) -------
[U, ~] = seq_dCblk(S, NSTEP, 1);
dK1 = decomposition(K1);
Y0  = dK1 \ full(U);
Bm  = full([sparse(m,m), speye(m); speye(m), sparse(m,m)]);
Cap = Bm + full(U' * Y0);
Xw  = Y0 * (Cap \ Bm);
e1  = norm(Kn * Xw - U, 'fro') / norm(full(U), 'fro');
assert(e1 < 1e-9, 'T1: Woodbury identity failed (%.3e)', e1);
fprintf('  PASS T1: K_n^-1 U via frozen factorization (%.1e), rcond(Cap)=%.1e\n', ...
        e1, rcond(Cap));
npass = npass + 1;

%% ---- T2: 'invref' and 'exactsolve' produce the SAME SPAN --------------
[W_inv, i_inv, ctx] = lowrank_update_basis(S, NSTEP, P1, ctx, ...
                          struct('mode', 'invref',     'ref', 1, 'Cn', C1));
[W_ex,  i_ex]       = lowrank_update_basis(S, NSTEP, P1, ctx, ...
                          struct('mode', 'exactsolve', 'ref', 1, 'Cn', C1));
oo  = struct('true_is_orth', true, 'comp_is_orth', true);
e2a = subspace_capture_directed(W_ex, W_inv, [], oo).eigspace_err_2;
e2b = subspace_capture_directed(W_inv, W_ex, [], oo).eigspace_err_2;
assert(max(e2a, e2b) < 1e-8, ...
       'T2: frozen-factorization span differs from the exact one (%.3e / %.3e)', e2a, e2b);
assert(i_inv.n_backsolves == m, ...
       'T2: expected %d backsolves per step, got %d', m, i_inv.n_backsolves);
assert(i_ex.n_backsolves == 2*m, 'T2: exactsolve should solve 2*nC columns');
fprintf(['  PASS T2: invref span == exactsolve span (%.1e / %.1e) at %d ' ...
         'backsolves vs %d\n'], e2a, e2b, i_inv.n_backsolves, i_ex.n_backsolves);
npass = npass + 1;

%% ---- T3: disk_static is the null control ------------------------------
Ss  = build_stokes_sequence(struct('case_name', 'disk_static', 'h0', 0.15, ...
                                   'nsteps', 4, 'use_cache', false, 'quiet', true));
Ps  = src.precond.make_ildl_precond(seq_K(Ss, 1), struct('mode', 'nofill'));
[Ws, is_] = lowrank_update_basis(Ss, 4, Ps, [], struct('mode', 'invref', 'ref', 1));
assert(is_.dC_nnz == 0, 'T3: disk_static dC should be exactly zero');
assert(is_.ncols == Ss.nC, ...
       'T3: expected the block to collapse to nC=%d columns, got %d', Ss.nC, is_.ncols);
assert(is_.rank_drop == Ss.nC, 'T3: expected rank drop of nC');
assert(size(Ws, 1) == Ss.n, 'T3: wrong row count');
fprintf('  PASS T3: disk_static -> dC=0, block collapses %d -> %d columns\n', ...
        is_.ncols_raw, is_.ncols);
npass = npass + 1;

%% ---- T4: [dC, Sel] really generates the operator change ---------------
dKmat = Kn - K1;
Qu    = orth_trunc(full(U));
resid = norm(full(dKmat) - Qu * (Qu' * full(dKmat)), 'fro') / norm(full(dKmat), 'fro');
assert(resid < 1e-12, 'T4: range(K_n - K_ref) not contained in range(U) (%.3e)', resid);
fprintf('  PASS T4: range(K_n - K_ref) subset range([dC,Sel]) (residual %.1e)\n', resid);
npass = npass + 1;

%% ---- T5: does the block capture what a frozen V misses? ---------------
% Coordinates frozen at step 1, so Ahat_n is a genuine rank-2nC perturbation of
% Ahat_ref and the containment argument applies.
kbase = 30;
M1 = C1 * C1';  M1 = (M1 + M1') / 2;
[Uref, ~] = eigs(K1, M1, kbase, 'smallestabs', 'Tolerance', 1e-11, 'MaxIterations', 2000);
[Utru, ~] = eigs(Kn, M1, kbase, 'smallestabs', 'Tolerance', 1e-11, 'MaxIterations', 2000);
Vref = transport_V(Uref, P1, C1);
Vtru = transport_V(Utru, P1, C1);
Vaug = augment_recycle_V(Vref, W_inv);

err_frozen = subspace_capture_directed(Vtru, Vref, [], oo).eigspace_err_2;
err_aug    = subspace_capture_directed(Vtru, Vaug, [], ...
                struct('true_is_orth', true)).eigspace_err_2;
fro_frozen = subspace_capture_directed(Vtru, Vref, [], oo).eigspace_err_fro;
fro_aug    = subspace_capture_directed(Vtru, Vaug, [], ...
                struct('true_is_orth', true)).eigspace_err_fro;
assert(err_aug <= err_frozen + 1e-12, ...
       'T5: augmenting made capture WORSE (%.3e -> %.3e)', err_frozen, err_aug);
fprintf(['  PASS T5: capture of the true step-%d eigenspace (k=%d, C frozen)\n' ...
         '          V_ref alone      : err_2 = %.3e   err_fro = %.3e\n' ...
         '          [V_ref, What]    : err_2 = %.3e   err_fro = %.3e   (+%d cols)\n'], ...
        NSTEP, kbase, err_frozen, fro_frozen, err_aug, fro_aug, ...
        size(Vaug, 2) - size(Vref, 2));
npass = npass + 1;

%% ---- T6: 'raw' mode and reported diagnostics --------------------------
[W_raw, i_raw] = lowrank_update_basis(S, NSTEP, P1, ctx, ...
                     struct('mode', 'raw', 'ref', 1, 'Cn', C1));
assert(size(W_raw, 1) == n && i_raw.n_backsolves == 0, ...
       'T6: raw mode should use no backsolves');
assert(isfinite(i_inv.rcond_capacitance) && i_inv.rcond_capacitance > 0, ...
       'T6: capacitance rcond not reported');
assert(norm(W_inv' * W_inv - eye(size(W_inv, 2)), 'fro') < 1e-10, ...
       'T6: returned block is not orthonormal');
ok = false;
try
    lowrank_update_basis(S, NSTEP, P1, ctx, struct('mode', 'nope'));
catch ME
    ok = strcmp(ME.identifier, 'lowrank_update_basis:unknownMode');
end
assert(ok, 'T6: unknown mode should error with a clear identifier');
fprintf(['  PASS T6: raw mode (0 backsolves, %d cols), orthonormal output, ' ...
         'rcond=%.1e, bad mode errors\n'], i_raw.ncols, i_inv.rcond_capacitance);
npass = npass + 1;

fprintf('=== test_lowrank_update_basis: %d checks passed ===\n', npass);
