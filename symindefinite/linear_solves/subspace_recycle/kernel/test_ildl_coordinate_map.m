% TEST_ILDL_COORDINATE_MAP  Unit tests for the explicit ILDL split factor.
%
% T2 (block-apply correctness) is the one to watch: every helper in this study
% hands make_ildl_precond's applyCinv/applyCtinv an n-by-k BLOCK rather than a
% single vector, relying on a property that is real (scatter_fwd/scatter_back are
% column-aware) but undocumented in the function's own header.
%
% T4 checks that the split transform preserves inertia (Sylvester): Ahat must
% still be indefinite, which is why the coarse correction has to be built on
% Ahat^2 rather than Ahat.
%
% Run:  test_ildl_coordinate_map

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
add_recycle_paths();
rng(1);

S = build_stokes_sequence(struct('case_name', 'bar_rotating', 'h0', 0.15, ...
                                 'nsteps', 3, 'use_cache', false, 'quiet', true));
K = seq_K(S, 1);
n = S.n;
P = src.precond.make_ildl_precond(K, struct('mode', 'nofill'));
[C, info] = ildl_coordinate_map(P);

fprintf('=== test_ildl_coordinate_map (n=%d, nnzL=%d) ===\n', n, info.nnzL);
npass = 0;

%% ---- T1: C agrees with the apply handles ------------------------------
x  = randn(n, 1);
e1 = norm(P.applyCinv(C * x) - x) / norm(x);
e2 = norm(C' * P.applyCtinv(x) - x) / norm(x);
assert(e1 < 1e-9 && e2 < 1e-9, 'T1: C vs handles %.3e / %.3e', e1, e2);
fprintf('  PASS T1: C^-1(Cx)=x (%.1e), C''(C^-T x)=x (%.1e)\n', e1, e2);
npass = npass + 1;

%% ---- T2: the applies are block-aware ----------------------------------
X  = randn(n, 5);
Y1 = P.applyCinv(X);
Y2 = zeros(n, 5);
for j = 1:5, Y2(:, j) = P.applyCinv(X(:, j)); end
eb = norm(Y1 - Y2, 'fro') / norm(Y2, 'fro');
Z1 = P.applyCtinv(X);
Z2 = zeros(n, 5);
for j = 1:5, Z2(:, j) = P.applyCtinv(X(:, j)); end
ec = norm(Z1 - Z2, 'fro') / norm(Z2, 'fro');
assert(eb < 1e-14 && ec < 1e-14, 'T2: block apply mismatch %.3e / %.3e', eb, ec);
fprintf('  PASS T2: applyCinv/applyCtinv are block-aware (%.1e / %.1e)\n', eb, ec);
npass = npass + 1;

%% ---- T3: M = C C' is SPD and applyMinv inverts it ---------------------
M  = C * C';  M = (M + M') / 2;
em = norm(P.applyMinv(M * x) - x) / norm(x);
assert(em < 1e-9, 'T3: M^-1(Mx) = x failed (%.3e)', em);
assert(min(eig(full(M(1:min(200, n), 1:min(200, n))))) > 0, ...
       'T3: leading block of M is not positive definite');
fprintf('  PASS T3: M = CC'' SPD, M^-1(Mx)=x (%.1e)\n', em);
npass = npass + 1;

%% ---- T4: the split transform preserves inertia (Sylvester) ------------
Ahat = full(C \ (K / C'));
Ahat = (Ahat + Ahat') / 2;
lk = eig(full(K));  la = eig(Ahat);
tolk = 1e-9 * max(abs(lk));  tola = 1e-9 * max(abs(la));
assert(sum(lk > tolk) == sum(la > tola) && sum(lk < -tolk) == sum(la < -tola), ...
       'T4: inertia not preserved (K: %d+/%d-, Ahat: %d+/%d-)', ...
       sum(lk > tolk), sum(lk < -tolk), sum(la > tola), sum(la < -tola));
assert(sum(la < -tola) > 0 && sum(la > tola) > 0, 'T4: Ahat should be indefinite');
fprintf('  PASS T4: inertia preserved (%d+ / %d-), Ahat still indefinite\n', ...
        sum(la > tola), sum(la < -tola));
npass = npass + 1;

%% ---- T5: drift fields against a reference ------------------------------
P2 = src.precond.make_ildl_precond(seq_K(S, 3), struct('mode', 'nofill'));
[~, info2] = ildl_coordinate_map(P2, info);
assert(isfinite(info2.perm_hamming) && info2.perm_hamming >= 0 && ...
       info2.perm_hamming <= n, 'T5: perm_hamming out of range');
assert(isnan(info.perm_hamming), 'T5: perm_hamming should be NaN without a ref');
assert(isfinite(info2.nnzL_ratio) && info2.nnzL_ratio > 0, 'T5: bad nnzL_ratio');
fprintf(['  PASS T5: drift vs step 1 -> perm_hamming=%d/%d (%.1f%%), ' ...
         'nnzL_ratio=%.4f\n'], info2.perm_hamming, n, ...
        100*info2.perm_hamming_frac, info2.nnzL_ratio);
npass = npass + 1;

fprintf('=== test_ildl_coordinate_map: %d checks passed ===\n', npass);
