%TEST_KERNEL  Unit tests for the coordinate_drift kernel helpers.
%
%   Run:  cd tests; test_kernel
%
%   These seven helpers are the load-bearing primitives of the whole study: if
%   `gap` is not basis-invariant, or `gap_M` secretly agrees with `gap` by
%   construction, every number in the document is meaningless.  So they are
%   tested first and independently of any experiment.
%
%   No file outside coordinate_drift/ is touched.

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(here), 'kernel'));
add_paths();

np = 0; nf = 0;
fprintf('=== test_kernel ===\n');

%% ---------------------------------------------------------------- gap ----
rng(0);
n = 40; k = 6;
X = randn(n, k);  Y = randn(n, k);

% T1  basis invariance: the gap depends only on the spans.
R1 = randn(k) + 3*eye(k);  R2 = randn(k) + 3*eye(k);
[np, nf] = chk(np, nf, 'T1  gap is invariant under X->X*R, Y->Y*R', ...
    abs(gap(X, Y) - gap(X*R1, Y*R2)) < 1e-10);

% T2  identical spans -> 0; orthogonal complements -> 1.
[np, nf] = chk(np, nf, 'T2a gap(X,X*R) = 0', gap(X, X*R1) < 1e-12);
Z = null(orth(X)');
[np, nf] = chk(np, nf, 'T2b gap(X, X^perp) = 1', abs(gap(X, Z(:,1:k)) - 1) < 1e-12);

% T3  known angle in 2D: gap = sin(theta).
th = 0.3;
[np, nf] = chk(np, nf, 'T3  gap([1;0],[cos;sin]) = sin(theta)', ...
    abs(gap([1;0], [cos(th); sin(th)]) - sin(th)) < 1e-12);

% T4  principal angles are returned ascending and match.
[~, gi] = gap([eye(2); zeros(n-2,2)], [cos(th) 0; 0 1; zeros(n-2,2)]);
[np, nf] = chk(np, nf, 'T4  principal angles ascending, max matches gap', ...
    issorted(gi.angles) && abs(sin(max(gi.angles)) - gap([eye(2); zeros(n-2,2)], ...
        [cos(th) 0; 0 1; zeros(n-2,2)])) < 1e-10);

%% -------------------------------------------------------------- gap_M ----
% T5  M = I reduces to the Euclidean gap.
[np, nf] = chk(np, nf, 'T5  gap_M(.,.,I) = gap', ...
    abs(gap_M(X, Y, speye(n)) - gap(X, Y)) < 1e-10);

% T6  THE ISOMETRY (Thm 1.2), computed by disjoint code paths:
%     gap_M(X,Y,M) uses M-orthogonal projectors and never sees C;
%     gap(C'X,C'Y) uses Euclidean projectors and never sees M.
B = randn(n) / sqrt(n);  C = chol(B*B' + 2*eye(n), 'lower');
M = C * C';
[np, nf] = chk(np, nf, 'T6  gap_M(X,Y,CC'') = gap(C''X, C''Y)   [isometry]', ...
    abs(gap_M(X, Y, M) - gap(C'*X, C'*Y)) < 1e-9);

% T7  gap_M is itself basis-invariant.
[np, nf] = chk(np, nf, 'T7  gap_M invariant under X->X*R', ...
    abs(gap_M(X, Y, M) - gap_M(X*R1, Y*R2, M)) < 1e-9);

%% ----------------------------------------------------- pencil_subspace ----
A = randn(n);  A = (A + A')/2;                 % symmetric indefinite
[U, lam, pi_] = pencil_subspace(A, M, 4, struct('mode','dense'));
% span(U) is invariant for M^-1 A -- stated as a span identity, since U is
% returned Euclidean-orthonormal while the pencil eigenvectors are M-orthogonal.
[np, nf] = chk(np, nf, 'T8  span(U) is M^-1A-invariant', ...
    gap(U, M \ (A * U)) < 1e-9);
[np, nf] = chk(np, nf, 'T9  returned |lambda| are the k smallest', ...
    max(abs(lam)) <= min(abs(pi_.lam_ext(5:end))) + 1e-10);

[U2, ~, ~] = pencil_subspace(A, M, 4, struct('mode','eigs'));
[np, nf] = chk(np, nf, 'T10 eigs and dense modes give the same subspace', ...
    gap(U, U2) < 1e-6);

% T11  Thm 1.1: C'*U spans an invariant subspace of Ahat = C^-1 A C^-T.
Ahat = (C \ A) / C';   Ahat = (Ahat + Ahat')/2;
Vh   = C' * U;
[np, nf] = chk(np, nf, 'T11 span(C''U) is Ahat-invariant  [Thm 1.1]', ...
    gap(Vh, Ahat*Vh) < 1e-9);

%% -------------------------------------------------------- gauge_split ----
% T12  pure regauge: C2 = C1*Q exactly => metric part vanishes.
Qr = orth(randn(n));
g1 = gauge_split(C, C*Qr, U);
[np, nf] = chk(np, nf, 'T12 pure regauge: delta_metric = 0  [Prop 2.4]', ...
    g1.delta_metric < 1e-8 && g1.relM < 1e-12);
[np, nf] = chk(np, nf, 'T13 pure regauge: delta_chart = delta_gauge', ...
    abs(g1.delta_chart - g1.delta_gauge) < 1e-8);

% T14  triangle inequality on a genuinely different pair.
B2 = randn(n)/sqrt(n);  C2 = chol(B2*B2' + 2*eye(n), 'lower');
g2 = gauge_split(C, C2, U);
[np, nf] = chk(np, nf, 'T14 delta_chart <= delta_metric + delta_gauge', g2.triangle_ok);

%% --------------------------------------------------- deflated_spectrum ----
% T15  basis invariance of the deflated spectrum (Thm 1.3).
lamd = [-1.5; -0.02; 0.03; 1.0; 2.0];
Qd   = orth(randn(5));  Ad = Qd*diag(lamd)*Qd';  Ad = (Ad+Ad')/2;
[Vd, Dd] = eig(Ad);  [~,od] = sort(abs(diag(Dd)));  Vsel = Vd(:, od(1:2));
o1 = deflated_spectrum(Ad, Vsel,        0.5);
o2 = deflated_spectrum(Ad, Vsel*orth(randn(2)), 0.5);
[np, nf] = chk(np, nf, 'T15 deflated spectrum invariant under V->V*Q  [Thm 1.3]', ...
    norm(o1.lam - o2.lam, inf) < 1e-10);

% T16  exact invariant V: the captured modes move to sqrt(tau)*sign(lambda).
%      This is the check that pins down WHICH operator MINRES sees -- G*Ahat,
%      not G*Ahat*G (which would give tau/lambda and blow up on exactly the
%      modes being deflated).
tau = 0.5;
pred = sort([sqrt(tau) * sign(lamd(od(1:2))); lamd(od(3:end))]);
[np, nf] = chk(np, nf, 'T16 exact V: coarse eigenvalues -> sqrt(tau)*sign(lambda)', ...
    norm(sort(o1.lam) - pred, inf) < 1e-9);

% T17  the reported spectrum really is that of G*Ahat.
[np, nf] = chk(np, nf, 'T17 reported spectrum equals eig(G*Ahat)', ...
    norm(sort(real(eig(o1.G * Ad))) - o1.lam, inf) < 1e-9);

% T18  pins down MATLAB's minres convention: the fifth argument is M, and
%      minres applies M^{-1}.  Passing M = A must therefore converge at once.
%      This is what makes T16/T17 the right reading of two_level_split_solve.
dpos = logspace(-2, 1, 12)';
[~, ~, ~, it1] = minres(diag(dpos), ones(12,1), 1e-12, 30, diag(dpos));
[np, nf] = chk(np, nf, 'T18 minres 5th argument is M, applied as M^{-1}', it1 == 1);

%% ------------------------------------ the SPD coarse correction (T19-T23) ----
% An SPD system does NOT need the squared operator: E = V'Ahat V is already
% positive definite, so the direct form P = (I-VV') + tau V E^{-1} V' applies.
% These five checks pin the SPD branch, and T22 pins the exact sense in which
% the two forms agree -- which is why the study's earlier ichol numbers, taken
% with the indefinite form, were a reparametrization rather than an error.
lams = [0.02; 0.05; 0.4; 1.0; 2.0];                % SPD spectrum
Qs   = orth(randn(5));  As = Qs*diag(lams)*Qs';  As = (As+As')/2;
[Vs_, Ds_] = eig(As);  [~, os] = sort(diag(Ds_));  Vsp = Vs_(:, os(1:2));
os1 = deflated_spectrum(As, Vsp, tau, 'spd');

% T19  exact invariant V, SPD form: the captured modes move to tau itself.
%      Contrast T16, where the indefinite form sends them to sqrt(tau)*sign.
pred_spd = sort([tau; tau; lams(os(3:end))]);
[np, nf] = chk(np, nf, 'T19 SPD exact V: coarse eigenvalues -> tau', ...
    norm(sort(os1.lam) - pred_spd, inf) < 1e-9);

% T20  the reported SPD spectrum really is that of P*Ahat.
[np, nf] = chk(np, nf, 'T20 SPD reported spectrum equals eig(P*Ahat)', ...
    norm(sort(real(eig(os1.G * As))) - os1.lam, inf) < 1e-9);

% T21  basis invariance of the SPD form (Thm 1.3, the p = 1 case).  The
%      tolerance scales with cond(E) exactly as in the indefinite case -- but E
%      is no longer squared, so it is a far smaller number and this passes with
%      correspondingly more room.
Vrot = Vsp * orth(randn(2));
[P1, E1s] = src.precond.deflation_P_apply(Vsp,  As, tau, 'handle');
 P2s      = src.precond.deflation_P_apply(Vrot, As, tau, 'handle');
Zt = randn(5, 4);
opdiff_spd = norm(P1(Zt) - P2s(Zt), 2) / norm(Zt, 2);
[np, nf] = chk(np, nf, 'T21 SPD P depends on span(V) only  [Thm 1.3]', ...
    opdiff_spd < 1e-12 * max(1, cond(E1s)));

% T22  THE EQUIVALENCE.  On an exactly invariant span(V) with SPD Ahat,
%      (V'Ahat^2 V)^{-1/2} = |L|^{-1} = (V'Ahat V)^{-1}, so the two forms are the
%      same operator up to tau <-> sqrt(tau).  This is what makes "the ichol
%      family was run with the indefinite form" a reparametrization at theta = 0
%      -- and, by exp6, a genuine difference away from it.
Pa = src.precond.deflation_P_apply(Vsp,     As,    sqrt(tau), 'matrix');
Pb = src.precond.deflation_Psqrt_apply(Vsp, As*As, tau,       'matrix');
[np, nf] = chk(np, nf, 'T22 SPD forms agree on invariant V when tau -> sqrt(tau)', ...
    norm(Pa - Pb, 'fro') / norm(Pa, 'fro') < 1e-12);

% T23  REGRESSION GUARD.  two_level_solve_local('indef') must reproduce the
%      production src.precond.two_level_split_solve iteration for iteration; the
%      refactor is only allowed to add the SPD branch, not to move the old one.
Ki  = randn(n);  Ki = (Ki + Ki')/2;                % symmetric indefinite
Pch = chart_struct(C, 'indef', tau);
bi  = ones(n, 1);
Vi  = orth_trunc(C' * U);
[~, ~, ~, itA] = src.precond.two_level_split_solve(Ki, bi, 1e-8, 400, Pch, Vi, tau);
[~, ~, ~, itB] = two_level_solve_local(          Ki, bi, 1e-8, 400, Pch, Vi, tau);
[~, ~, ~, itC] = src.precond.two_level_split_solve(Ki, bi, 1e-8, 400, Pch, [], tau);
[~, ~, ~, itD] = two_level_solve_local(          Ki, bi, 1e-8, 400, Pch, [], tau);
[np, nf] = chk(np, nf, 'T23 local indef solver matches two_level_split_solve', ...
    itA == itB && itC == itD);

%% ------------------------------------------------------------ summary ----
fprintf('\n%d passed, %d FAILED\n', np, nf);
if nf > 0, error('test_kernel:failures', '%d checks failed', nf); end

%==========================================================================
function [np, nf] = chk(np, nf, name, cond)
    if cond
        np = np + 1;  fprintf('  ok   %s\n', name);
    else
        nf = nf + 1;  fprintf('  FAIL %s\n', name);
    end
end
