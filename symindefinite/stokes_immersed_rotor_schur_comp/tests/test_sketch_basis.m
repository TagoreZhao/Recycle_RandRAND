% TEST_SKETCH_BASIS  Unit gate for build_sketch_V and the deflation projector.
%
% The coarse space of the deflate_gaussian arm is a Gaussian sketch of the
% INVERSE Schur complement
%
%     S^-1,   applied through the frozen Cholesky factor of S_1,
%
% mirroring src.precond.build_deflation_V's 'gaussian' method.  There is no
% split factor in this study: BOTH halves of the scheme live on S itself, and
% T6/T7 below pin that numerically rather than in prose.  The projector
%
%     P = (I - V V') + tau * V (V' S V)^-1 V'
%
% is built on S -- first power, no squaring, because S is SPD (the squared/sqrt
% form belongs to the INDEFINITE sibling only).  P is itself symmetric positive
% definite, which is what lets it be handed to pcg directly, with no
% B = L^-T P L^-1 composition to get wrong.
%
% Every subspace claim is made through src.precond.subspace_capture -- angles
% and projector residuals -- never through individual columns, which are not
% basis-invariant quantities.
%
% SPECTRAL CONTEXT (measured here, and the reason T3 is worded as it is).  At
% h0 = 0.1 the eigenvalues of the raw S are
%
%     1.51e-5, 1.35e-4, 3.50e-4, 5.88e-4, ... up to lam_max = 0.542,
%     kappa = 3.59e4
%
% i.e. a genuinely graded low end, NOT the single isolated mode the split
% operator had.  Deflating the smallest 20 takes kappa to 3.15e2, a 114x
% reduction -- so on this operator deflation has something real to remove.
% Inverse power iteration separates modes at rate (lam_j/lam_{j+1})^(2q):
% lam_1/lam_2 = 0.11 and lam_2/lam_3 = 0.38, so the first two modes are
% recoverable at q = 2, while the near-degenerate interior (lam_20/lam_21 =
% 0.969) is not.  T3 therefore asserts what the spectrum permits and reports the
% whole-block figure without asserting it.
%
% Run:  cd tests; test_sketch_basis

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(fileparts(thisFileDir));
add_schur_paths();
assert_local_helpers();
import src.precond.*
rng(1);

params = make_schur_params();
params.h0 = 0.1;                        % nS ~ 542
k         = 20;
q         = params.q;

cfg = schur_make_cfg('bar_rotating', params, []);
ctx = schur_context_init(cfg, params);
st  = schur_step_operator(ctx, params.dt, zeros(ctx.nU, 1));

S  = st.S;
nS = size(S, 1);

Rf   = chol(S, 'lower');
Sinv = @(X) Rf' \ (Rf \ X);             % the handle build_sketch_V sketches

fprintf('  nS = %d, k = %d, q = %d\n', nS, k, q);

npass = 0;

% --- reference: the k smallest eigenvectors of S, exactly ------------------
% S is small (nS ~ 542) and already dense, so use eig, not eigs -- exact, and it
% gives the whole spectrum for T6.  Columns are kept in ascending eigenvalue
% order and are NOT re-orthogonalized: orth() returns an arbitrary basis of the
% span and would destroy the per-eigenvector column identity that
% subspace_capture's residual_per_vec depends on.
[Uf, Df]     = eig(full(S));
[lam_S, ord] = sort(real(diag(Df)), 'ascend');
Uf = Uf(:, ord);
Ve = Uf(:, 1:k);

% tau matches production: solve_schur_sequence auto-resolves it to lambda_max.
tau = lam_S(end);

fprintf('  spec(S): %s ... %.4g   (lam_%d/lam_%d = %.4f, kappa = %.4g)\n', ...
        mat2str(round(lam_S(1:4)', 8)), lam_S(end), k, k+1, ...
        lam_S(k) / lam_S(k+1), lam_S(end) / lam_S(1));

% --- the object under test -------------------------------------------------
rng(7);
V = build_sketch_V(Sinv, nS, k, q);

% T1: orthonormal columns
assert(size(V, 1) == nS, 'T1a: V has %d rows, expected %d', size(V, 1), nS);
assert(norm(V' * V - eye(size(V, 2)), 'fro') < 1e-10, ...
    'T1b: V is not orthonormal (||V''V - I||_F = %.3e)', ...
    norm(V' * V - eye(size(V, 2)), 'fro'));
npass = npass + 2;

if size(V, 2) < k
    warning('test_sketch_basis:rankDrop', ...
        'orth dropped %d of %d sketch columns (realized width %d)', ...
        k - size(V, 2), k, size(V, 2));
end
fprintf('  realized basis width: %d of %d\n', size(V, 2), k);

% T2: the coarse matrix deflation_P_apply forms must be SPD.  This is a hard
% precondition, not a nicety: deflation_P_apply chol()s E and errors otherwise.
E = V' * (S * V);
E = (E + E') / 2;
[~, cflag] = chol(E);
assert(cflag == 0, 'T2a: V''*S*V is not SPD (chol flag %d)', cflag);
Papply = deflation_P_apply(V, S, tau, [], 0);
z = Papply(randn(nS, 1));
assert(all(isfinite(z)), 'T2b: deflation_P_apply produced non-finite output');
npass = npass + 2;

% T3: the sketch captures the SEPARATED small modes of S.  Only the modes the
% spectrum actually separates can be asserted (see the header): mode 1 sits at
% 1.5e-5 against 1.35e-4, a ratio of 0.11, and is recovered at q = 2.  The
% near-degenerate interior is unreachable by any power method and is reported,
% not asserted.
capt = subspace_capture(Ve, V);
fprintf('  capture vs exact V:  frob_res = %.4f (whole block, bulk-dominated)\n', ...
        capt.frob_residual_rel);
fprintf('  per-mode residual, first 5: %s\n', ...
        mat2str(round(capt.residual_per_vec(1:min(5, k))', 4)));
assert(capt.residual_per_vec(1) < 0.05, ...
    'T3: the sketch missed the most separated mode (residual %.4f); with lam_1/lam_2 = 0.11 it must be captured at q = %d', ...
    capt.residual_per_vec(1), q);
npass = npass + 1;

% T4: more power iterations must not capture LESS.  This is the check that
% catches the classic wiring bug -- iterating the FORWARD operator instead of
% the inverse converges to the LARGEST modes, so capture would degrade with q.
rng(7);  V1 = build_sketch_V(Sinv, nS, k, 1);
rng(7);  V4 = build_sketch_V(Sinv, nS, k, 4);
c1 = subspace_capture(Ve, V1);
c4 = subspace_capture(Ve, V4);
fprintf('  frob_res:  q=1 -> %.4f,  q=4 -> %.4f   (most separated mode %.4f -> %.4f)\n', ...
        c1.frob_residual_rel, c4.frob_residual_rel, ...
        c1.residual_per_vec(1), c4.residual_per_vec(1));
assert(c4.frob_residual_rel <= c1.frob_residual_rel + 1e-8, ...
    'T4a: capture got WORSE with more power iterations (q=1 %.4f -> q=4 %.4f); the handle is probably the forward operator, not the inverse', ...
    c1.frob_residual_rel, c4.frob_residual_rel);
assert(c4.residual_per_vec(1) <= c1.residual_per_vec(1) + 1e-8, ...
    'T4b: the most separated mode got WORSE with more power iterations (%.4f -> %.4f)', ...
    c1.residual_per_vec(1), c4.residual_per_vec(1));
npass = npass + 2;

% T5: the result is a property of the SPAN, not of the basis returned.  Feed a
% re-orthogonalized, re-mixed copy of V and require an identical capture.
[Vm, ~] = qr(V * orth(randn(size(V, 2))), 0);
cm = subspace_capture(Ve, Vm);
assert(abs(cm.frob_residual_rel - capt.frob_residual_rel) < 1e-8, ...
    'T5: capture changed under a change of basis within the same span (%.3e vs %.3e)', ...
    cm.frob_residual_rel, capt.frob_residual_rel);
npass = npass + 1;

% =========================================================================
% T6/T7: the projector is built on S, and is a valid pcg preconditioner
% =========================================================================
% T6: with V = the EXACT smallest eigenvectors of S, span(V) is invariant, so
% P*S must have exactly k eigenvalues moved to tau and leave the remaining n-k
% eigenvalues of S untouched.  That is the definitive check that P was built on
% S: had it been built on S^2, or on a preconditioned form, the captured modes
% would land somewhere else entirely.
%
% NOTE the expected spectrum is assembled rather than counted: tau = lam_max is
% itself an eigenvalue of S, so "count the modes sitting at tau" would report
% k+1 and a naive tail comparison would be off by one.
Pmat  = deflation_P_apply(Ve, S, tau, 'matrix', 0);
lam_P = sort(real(eig(Pmat * full(S))), 'ascend');

lam_expect = sort([repmat(tau, k, 1); lam_S(k+1:end)], 'ascend');
assert(numel(lam_P) == numel(lam_expect), ...
    'T6a: deflated operator has %d modes, expected %d', ...
    numel(lam_P), numel(lam_expect));
assert(max(abs(lam_P - lam_expect)) < 1e-8 * tau, ...
    'T6b: spec(P*S) is not "k modes at tau, tail untouched" (max deviation %.3e) -- P is not built on S', ...
    max(abs(lam_P - lam_expect)));
fprintf('  P*S: %d smallest modes moved to tau = %g; kappa %.4g -> %.4g\n', ...
        k, tau, lam_S(end) / lam_S(1), max(lam_P) / min(lam_P));
npass = npass + 2;

% T7: P is SPD in its own right.  This is what makes the split composition
% unnecessary -- pcg requires a symmetric positive definite preconditioner, and
% with no L to absorb, P must supply that itself or pcg is being misused.
assert(norm(Pmat - Pmat', 'fro') < 1e-10 * norm(Pmat, 'fro'), ...
    'T7a: P is not symmetric (||P - P''||_F / ||P||_F = %.3e)', ...
    norm(Pmat - Pmat', 'fro') / norm(Pmat, 'fro'));
lam_Pm = sort(real(eig((Pmat + Pmat') / 2)), 'ascend');
assert(lam_Pm(1) > 0, ...
    'T7b: P is not positive definite (lambda_min = %.3e); it cannot be a pcg preconditioner', ...
    lam_Pm(1));
fprintf('  spec(P) in [%.4g, %.4g]\n', lam_Pm(1), lam_Pm(end));
npass = npass + 2;

fprintf('\ntest_sketch_basis: ALL %d ASSERTIONS PASSED\n', npass);
