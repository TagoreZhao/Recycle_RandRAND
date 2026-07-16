% TEST_SUBSPACE_CAPTURE_DIRECTED  Correctness checks for subspace_capture_directed.
%
% Small synthetic checks (seconds, no mesh / no eigensolves). Every check
% throws on failure; the script prints one PASS line per test group and a
% final summary. Run from anywhere:
%
%   run('subspace_capture/test_subspace_capture_directed.m')

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(thisFileDir);
addpath(repoRoot);      % src.* package (for the cross-check vs old metric)
addpath(thisFileDir);   % subspace_capture_directed.m

rng(7);
n = 200;
k = 10;

% Orthonormal ground-truth basis and an orthonormal complement block.
[Qfull, ~] = qr(randn(n, 2 * k + 5), 0);
V_true = Qfull(:, 1:k);              % target eigenspace
W_extra = Qfull(:, k+1:2*k+5);       % directions orthogonal to V_true

tol_eq = 1e-10;

%% Test 1: basis invariance of the directed sines --------------------------
O1 = rand_orth(k);  O2 = rand_orth(k);
info1 = subspace_capture_directed(V_true * O1, V_true(:, 1:k-2));
info2 = subspace_capture_directed(V_true * O2, V_true(:, 1:k-2));
assert(max(abs(info1.sin_angles_directed - info2.sin_angles_directed)) < tol_eq, ...
       'T1: directed sines must be basis-invariant');
% The old per-column residuals DO depend on the basis (that was the problem).
assert(max(abs(info1.residual_per_vec - info2.residual_per_vec)) > 1e-3, ...
       'T1: per-column residuals expected to differ across bases here');
fprintf('PASS T1: basis invariance (and old metric shown basis-dependent)\n');

%% Test 2: exact containment -> zero error ---------------------------------
info = subspace_capture_directed(V_true, V_true * rand_orth(k));
assert(info.eigspace_err_2 < tol_eq,  'T2: err_2 should be ~0');
assert(info.eigspace_err_fro < tol_eq, 'T2: err_fro should be ~0');
assert(info.n_angle_below_1pct == k,  'T2: all k directions captured');
assert(info.r_true == k && info.r_comp == k, 'T2: ranks');
fprintf('PASS T2: exact containment\n');

%% Test 3: known principal angles ------------------------------------------
% span(U) = {e1, e2}; span(W) = {e1, cos(th) e2 + sin(th) e3}.
% Principal angles are exactly [0, th].
th = 0.3;
U3 = zeros(n, 2); U3(1,1) = 1; U3(2,2) = 1;
W3 = zeros(n, 2); W3(1,1) = 1; W3(2,2) = cos(th); W3(3,2) = sin(th);
info = subspace_capture_directed(U3, W3);
assert(max(abs(sort(info.sin_angles_directed) - sort([0; sin(th)]))) < tol_eq, ...
       'T3: sines must match the constructed angles');
assert(abs(info.eigspace_err_2 - sin(th)) < tol_eq, 'T3: err_2 = sin(theta)');
fprintf('PASS T3: known principal angles recovered\n');

%% Test 4: partial capture (one direction fully missed) --------------------
V_comp = [V_true(:, 1:k-1) * rand_orth(k-1), W_extra(:, 1:3)];   % 9 of 10 + noise dirs
info = subspace_capture_directed(V_true * rand_orth(k), V_comp);
assert(abs(info.eigspace_err_2 - 1) < tol_eq, 'T4: missed direction -> err_2 = 1');
assert(info.n_angle_below_1pct == k - 1, 'T4: exactly k-1 directions captured');
fprintf('PASS T4: partial capture counts\n');

%% Test 5: rank deficiency must not inflate capture ------------------------
v = V_true(:, 1);  w = W_extra(:, 1);
% 5a: all-duplicate block spans only v.
info = subspace_capture_directed(V_true, [v, v, 2*v]);
assert(info.r_comp == 1, 'T5a: rank of [v v 2v] is 1');
assert(info.n_angle_below_1pct == 1, 'T5a: only v is captured');
% 5b: dependent column in the MIDDLE (pivoted-QR truncation must keep w).
info = subspace_capture_directed(w, [v, v + 1e-15 * randn(n,1), w]);
assert(info.r_comp == 2, 'T5b: numerical rank of [v, v+eps, w] is 2');
assert(info.eigspace_err_2 < 1e-6, 'T5b: w must be captured despite mid-block dependency');
% 5c: near-duplicate columns.
info = subspace_capture_directed(V_true, [v, v + 1e-14 * w]);
assert(info.r_comp == 1, 'T5c: near-duplicate block has numerical rank 1');
fprintf('PASS T5: rank-deficient candidate blocks\n');

%% Test 6: directed (asymmetric) containment -------------------------------
V_comp = [V_true, W_extra] * rand_orth(k + size(W_extra, 2));  % strictly bigger space
info = subspace_capture_directed(V_true * rand_orth(k), V_comp);
assert(info.eigspace_err_2 < tol_eq, 'T6: V_true inside a larger span -> err ~ 0');
assert(info.r_comp > info.r_true, 'T6: candidate space is strictly larger');
fprintf('PASS T6: directed containment (m >> k)\n');

%% Test 7: edge cases --------------------------------------------------------
info = subspace_capture_directed(V_true, zeros(n, 3));           % r_comp = 0
assert(info.eigspace_err_2 == 1 && all(info.sin_angles_directed == 1), ...
       'T7: zero candidate -> everything missed');
info = subspace_capture_directed(zeros(n, 2), V_true);           % r_true = 0
assert(info.eigspace_err_2 == 0 && isempty(info.sin_angles_directed), ...
       'T7: empty target -> zero error');
info = subspace_capture_directed(V_true(:, 1), V_true(:, 1));    % k = 1
assert(info.eigspace_err_2 < tol_eq, 'T7: single captured vector');
info = subspace_capture_directed(V_true, V_true(:, 1:5), [0.5; 1e-2]);
assert(isequal(size(info.n_angle_below), [2 1]), 'T7: custom thresholds shape');
assert(info.n_angle_below(1) >= info.n_angle_below(2), 'T7: monotone counts');
% Krylov-style: r_comp < r_true pads the uncapturable directions with sin=1.
info = subspace_capture_directed(V_true, V_true(:, 1:4));
assert(sum(info.sin_angles_directed > 1 - tol_eq) == k - 4, ...
       'T7: r_true - r_comp padded ones');
fprintf('PASS T7: edge cases\n');

%% Test 8: consistency with the old src.precond.subspace_capture -----------
% Slightly perturbed copy of the true space: both metrics see the same
% subspace geometry when V_true is orthonormal.
V_comp = orth(V_true + 0.05 * randn(n, k));
info_new = subspace_capture_directed(V_true, V_comp);
info_old = src.precond.subspace_capture(V_true, V_comp);
assert(abs(info_new.eigspace_err_2 - sin(max(info_old.principal_angles))) < 1e-8, ...
       'T8: err_2 == sin(max principal angle) of the old function');
assert(info_new.eigspace_err_2 >= info_old.max_residual - 1e-12, ...
       'T8: worst-direction error dominates worst-column error');
fprintf('PASS T8: consistency with src.precond.subspace_capture\n');

fprintf('\nAll subspace_capture_directed tests passed.\n');

%% -------------------------------------------------------------------------
function O = rand_orth(p)
%RAND_ORTH  Random p-by-p orthogonal matrix (Haar via QR sign fix).
    [O, R] = qr(randn(p), 0);
    O = O * diag(sign(diag(R)));
end
