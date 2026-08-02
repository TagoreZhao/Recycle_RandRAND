% TEST_TRANSPORT_V  Unit tests for the coordinate-transport fix and orth_trunc.
%
% transport_V is the H1 repair: re-express a deflation space in the CURRENT
% split coordinates instead of freezing the step-1 representation.  T5 is the
% one that shows why the repair is needed at all — transporting into a later
% step's coordinates produces a genuinely DIFFERENT subspace, which is precisely
% what the production code fails to do.
%
% Run:  test_transport_V

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
add_recycle_paths();
rng(1);

S = build_stokes_sequence(struct('case_name', 'bar_rotating', 'h0', 0.15, ...
                                 'nsteps', 3, 'use_cache', false, 'quiet', true));
K1 = seq_K(S, 1);  K3 = seq_K(S, 3);
n  = S.n;  k = 20;
P1 = src.precond.make_ildl_precond(K1, struct('mode', 'nofill'));
P3 = src.precond.make_ildl_precond(K3, struct('mode', 'nofill'));
C1 = ildl_coordinate_map(P1);

fprintf('=== test_transport_V (n=%d, k=%d) ===\n', n, k);
npass = 0;

%% ---- T1: round trip through the same coordinates ----------------------
Vr = orth_trunc(randn(n, k));
Ur = P1.applyCtinv(Vr);                    % split -> original
Vb = transport_V(Ur, P1, C1);              % original -> split
e1 = subspace_capture_directed(Vr, Vb, [], ...
        struct('true_is_orth', true, 'comp_is_orth', true)).eigspace_err_2;
assert(e1 < 1e-9, 'T1: round trip lost the subspace (err %.3e)', e1);
fprintf('  PASS T1: C''(C^-T V) spans V (capture err %.1e)\n', e1);
npass = npass + 1;

%% ---- T2: transported eigenvectors are eigenvectors of Ahat ------------
M = C1 * C1';  M = (M + M') / 2;
[U1, ~] = eigs(K1, M, k, 'smallestabs', 'Tolerance', 1e-12, 'MaxIterations', 2000);
V1 = transport_V(U1, P1, C1);
Ahat = @(Y) P1.applyCinv(K1 * P1.applyCtinv(Y));
AV   = Ahat(V1);
R    = AV - V1 * (V1' * AV);               % invariant-subspace residual
e2   = norm(R, 'fro') / norm(AV, 'fro');
assert(e2 < 1e-8, 'T2: transported eigvecs are not Ahat-invariant (%.3e)', e2);
fprintf('  PASS T2: span(C''U) is Ahat-invariant (residual %.1e)\n', e2);
npass = npass + 1;

%% ---- T3: output is orthonormal and full rank --------------------------
o3 = norm(V1' * V1 - eye(size(V1, 2)), 'fro');
assert(o3 < 1e-12, 'T3: output not orthonormal (%.3e)', o3);
assert(size(V1, 2) == k, 'T3: lost columns (%d of %d)', size(V1, 2), k);
fprintf('  PASS T3: orthonormal (%.1e), kept all %d columns\n', o3, k);
npass = npass + 1;

%% ---- T4: orth_trunc drops numerically dependent columns ---------------
q  = orth_trunc(randn(n, 3));
Yd = [q(:,1), q(:,2), q(:,1) + 1e-15*q(:,3), q(:,1) - q(:,2)];
[Qd, rd] = orth_trunc(Yd);
assert(rd == 2, 'T4: expected numerical rank 2, got %d', rd);
assert(size(Qd, 2) == 2 && norm(Qd'*Qd - eye(2), 'fro') < 1e-12, ...
       'T4: truncated basis is malformed');
[Q0, r0] = orth_trunc(zeros(n, 0));
assert(isequal(size(Q0), [n 0]), 'T4: empty input should give an n-by-0 basis');
assert(r0 == 0, 'T4: empty input should give rank 0');
fprintf('  PASS T4: orth_trunc rank truncation (4 cols -> rank %d) and empty input\n', rd);
npass = npass + 1;

%% ---- T5: a later step's coordinates give a DIFFERENT subspace ---------
% This is the H1 mechanism in miniature: the same physical eigenvectors U1, but
% expressed in step-3 coordinates, span a different subspace than the frozen
% step-1 representation.  The production code keeps V1 and uses it against
% Ahat_3 anyway.
V3   = transport_V(U1, P3);
edif = subspace_capture_directed(V3, V1, [], ...
          struct('true_is_orth', true, 'comp_is_orth', true)).eigspace_err_2;
assert(edif > 1e-3, ...
       'T5: transport into step-3 coordinates changed nothing (err %.3e)', edif);
% and what the frozen V1 actually deflates at step 3, pulled back:
U1eff = P3.applyCtinv(V1);
edrift = subspace_capture_directed(U1, U1eff, []).eigspace_err_2;
fprintf(['  PASS T5: step-3 coords give a different span (err %.3e); ' ...
         'frozen V1 pulled back through C_3^-T misses U1 by %.3e\n'], edif, edrift);
npass = npass + 1;

fprintf('=== test_transport_V: %d checks passed ===\n', npass);
