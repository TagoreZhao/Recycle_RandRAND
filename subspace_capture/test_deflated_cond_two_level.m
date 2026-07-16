% TEST_DEFLATED_COND_TWO_LEVEL  Correctness checks for deflated_cond_two_level.
%
% Small synthetic checks (seconds, no mesh / no large eigensolves). Every
% check throws on failure; the script prints one PASS line per test group
% and a final summary. Run from anywhere:
%
%   run('subspace_capture/test_deflated_cond_two_level.m')

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(thisFileDir);
addpath(repoRoot);      % src.* package (for the cross-check vs deflation_Psqrt_apply)
addpath(thisFileDir);   % deflated_cond_two_level.m

rng(7);
n = 200;
k = 20;

% Synthetic SPD operator with a known ascending spectrum.
[Qo, ~] = qr(randn(n), 0);
d_true  = logspace(-3, 2, n)';
T       = Qo * diag(d_true) * Qo';
T       = (T + T') / 2;
Tfun    = @(X) T * X;
TinvFun = @(X) T \ X;

V_exact = Qo(:, 1:k);               % exact smallest-k eigenvectors
tau     = d_true(k + 1);            % lam_cut analogue
kappa_exact_analytic = d_true(n) / tau;

tol_tight = 1e-6;

%% Test 1: exact eigvec deflation matches the analytic condition number -----
out = deflated_cond_two_level(V_exact, Tfun, TinvFun, tau, n);
assert(out.ok, 'T1: ok must be true');
assert(out.r == k, 'T1: rank must be k');
assert(abs(out.kappa   / kappa_exact_analytic - 1) < tol_tight, ...
       'T1: kappa must match lam_max/lam_cut');
assert(abs(out.lam_min / tau       - 1) < tol_tight, 'T1: lam_min ~ tau');
assert(abs(out.lam_max / d_true(n) - 1) < tol_tight, 'T1: lam_max ~ d(n)');

% W_is_orth fast path must give the same answer for an orthonormal input.
out_orth = deflated_cond_two_level(V_exact, Tfun, TinvFun, tau, n, ...
                                   struct('W_is_orth', true));
assert(out_orth.ok && abs(out_orth.kappa / out.kappa - 1) < tol_tight, ...
       'T1: W_is_orth path must match');
fprintf('PASS T1: exact deflation matches analytic kappa\n');

%% Test 2: empty W = no deflation -> kappa(Tsym) -----------------------------
out = deflated_cond_two_level(zeros(n, 0), Tfun, TinvFun, tau, n);
assert(out.ok, 'T2: ok must be true');
assert(out.r == 0, 'T2: rank must be 0');
assert(abs(out.kappa / (d_true(n) / d_true(1)) - 1) < tol_tight, ...
       'T2: kappa must equal cond(Tsym)');
fprintf('PASS T2: empty W degenerates to cond(Tsym)\n');

%% Test 3: random W cannot beat exact deflation ------------------------------
out = deflated_cond_two_level(randn(n, k), Tfun, TinvFun, tau, n);
assert(out.ok, 'T3: ok must be true');
assert(out.kappa >= kappa_exact_analytic * (1 - 1e-8), ...
       'T3: exact deflation is optimal');
fprintf('PASS T3: random W gives kappa >= exact (ratio %.3e)\n', ...
        out.kappa / kappa_exact_analytic);

%% Test 4: basis invariance (non-orthonormal input, same span) ---------------
out_ref = deflated_cond_two_level(V_exact, Tfun, TinvFun, tau, n);
out_mix = deflated_cond_two_level(V_exact * (randn(k) + 5 * eye(k)), ...
                                  Tfun, TinvFun, tau, n);
assert(out_mix.ok && out_mix.r == k, 'T4: full-rank mixed basis');
assert(abs(out_mix.kappa / out_ref.kappa - 1) < tol_tight, ...
       'T4: kappa must depend only on span(W)');
fprintf('PASS T4: basis invariance\n');

%% Test 5: rank-deficient W truncates to its span ----------------------------
v = V_exact(:, 1);
out_rd = deflated_cond_two_level([v, v, 2 * v], Tfun, TinvFun, tau, n);
out_1  = deflated_cond_two_level(v, Tfun, TinvFun, tau, n);
assert(out_rd.ok && out_rd.r == 1, 'T5: rank must truncate to 1');
assert(abs(out_rd.kappa / out_1.kappa - 1) < 1e-8, ...
       'T5: kappa must match single-vector deflation');
fprintf('PASS T5: rank-deficient W handled\n');

%% Test 6: dense cross-check vs src.precond.deflation_Psqrt_apply ------------
W6     = randn(n, 8);
Worth  = orth(W6);
Phalf  = src.precond.deflation_Psqrt_apply(Worth, T, tau, 'matrix');
H      = Phalf * T * Phalf;
H      = (H + H') / 2;
ev     = eig(H);
out = deflated_cond_two_level(W6, Tfun, TinvFun, tau, n);
assert(out.ok, 'T6: ok must be true');
assert(abs(out.lam_max / max(ev) - 1) < 1e-8, 'T6: lam_max vs dense');
assert(abs(out.lam_min / min(ev) - 1) < 1e-8, 'T6: lam_min vs dense');
assert(abs(out.kappa / (max(ev) / min(ev)) - 1) < 1e-8, 'T6: kappa vs dense');
fprintf('PASS T6: matches dense eig of P^{1/2} T P^{1/2}\n');

%% Test 7: indefinite operator -> ok=false, no throw -------------------------
d_bad = d_true;  d_bad(1) = -1;
Tbad  = Qo * diag(d_bad) * Qo';
Tbad  = (Tbad + Tbad') / 2;
out = deflated_cond_two_level(V_exact, @(X) Tbad * X, @(X) Tbad \ X, tau, n);
assert(~out.ok, 'T7: ok must be false for indefinite coarse matrix');
assert(isnan(out.kappa), 'T7: kappa must be NaN');
assert(~isempty(out.err), 'T7: err must be set');
fprintf('PASS T7: indefinite operator fails gracefully\n');

fprintf('\nAll deflated_cond_two_level tests passed.\n');
