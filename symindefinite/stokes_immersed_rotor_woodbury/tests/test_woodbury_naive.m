%TEST_WOODBURY_NAIVE  The generic kernel is the identity, and reports itself honestly.
%
%   Run:  cd tests; test_woodbury_naive
%
%   woodbury_naive is the textbook evaluation on small dense systems; woodbury_solve
%   is the same expression, in the same order, against a frozen sparse factorization.
%   test_stress_metrics uses the first to characterize the identity's failure modes
%   and the README carries the second's numbers, so the two must be the same method.
%   This test pins that, plus the diagnostics' own definitions -- a cancellation
%   factor that does not compute what it claims would quietly invalidate every
%   conclusion drawn from it.
%
%   See also: woodbury_naive, woodbury_solve, test_stress_metrics.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
studyDir    = fileparts(thisFileDir);
addpath(studyDir);
add_woodbury_paths();
assert_woodbury_helpers();

rng(0);
fprintf('=== test_woodbury_naive ===\n');

np = 0;  nf = 0;

% --- T1  it solves (A + UCV)x = b on random well-conditioned systems ------
NTRIAL = 20;
mx_dir = 0;  mx_res = 0;
for t = 1:NTRIAL
    m  = 6 + mod(t, 5);
    k  = 1 + mod(t, 3);
    A  = eye(m) + 0.3 * randn(m);
    U  = randn(m, k);
    V  = randn(k, m);
    C  = eye(k) + 0.2 * randn(k);
    b  = randn(m, 1);
    M  = A + U * C * V;
    if cond(M) > 1e6 || cond(A) > 1e6 || cond(C) > 1e6
        continue                                  % not a stability test; see T4-T6
    end
    xd = M \ b;
    [x, d] = woodbury_naive(A, U, C, V, b, xd);
    mx_dir = max(mx_dir, norm(x - xd) / norm(xd));
    mx_res = max(mx_res, d.relres);
end
[np, nf] = chk(np, nf, ...
    sprintf('T1a matches the direct solve on %d random systems (%.2e < 1e-10)', ...
            NTRIAL, mx_dir), ...
    mx_dir < 1e-10);
[np, nf] = chk(np, nf, ...
    sprintf('T1b and its residual is at rounding level (%.2e < 1e-12)', mx_res), ...
    mx_res < 1e-12);

% --- T2  rectangular V, so V ~= U' is really supported -------------------
% The Stokes path uses V = U', but the identity does not require it, and a kernel
% that silently assumed symmetry would pass every test above.
A = eye(7) + 0.4 * randn(7);
U = randn(7, 2);  V = randn(2, 7);  C = [2 0.3; -0.1 1.5];  b = randn(7, 1);
xd = (A + U * C * V) \ b;
[x, ~] = woodbury_naive(A, U, C, V, b, xd);
relNS = norm(x - xd) / norm(xd);
[np, nf] = chk(np, nf, ...
    sprintf('T2  handles nonsymmetric V ~= U'' (%.2e < 1e-10)', relNS), relNS < 1e-10);

% --- T3  every documented field is present and finite --------------------
[~, d] = woodbury_naive(A, U, C, V, b, xd);
want = {'kappa_A','kappa_M','kappa_S','cancel_S','cancel_sub','resid', ...
        'relres','bwd','fwd','S','w','z','Y'};
missing = want(~isfield(d, want));
[np, nf] = chk(np, nf, ...
    sprintf('T3a info carries every documented field (%d missing)', numel(missing)), ...
    isempty(missing));
scal = {'kappa_A','kappa_M','kappa_S','cancel_S','cancel_sub','resid','relres','bwd','fwd'};
allfin = all(cellfun(@(f) isscalar(d.(f)) && isfinite(d.(f)), scal));
[np, nf] = chk(np, nf, 'T3b every reported scalar is finite', allfin);

% --- T4  the diagnostics compute what they claim -------------------------
% Recomputed here from d's own intermediates, so a definition that drifted from
% the docstring fails rather than being taken on trust.
Ci  = inv(C);
cS  = (norm(Ci) + norm(V) * norm(d.Y)) / norm(d.S);
Yw  = d.Y * d.w;
cSb = (norm(d.z) + norm(Yw)) / norm(d.z - Yw);
[np, nf] = chk(np, nf, ...
    sprintf('T4a cancel_S == its definition (%.3e vs %.3e)', d.cancel_S, cS), ...
    abs(d.cancel_S - cS) <= 1e-12 * cS);
[np, nf] = chk(np, nf, ...
    sprintf('T4b cancel_sub == its definition (%.3e vs %.3e)', d.cancel_sub, cSb), ...
    abs(d.cancel_sub - cSb) <= 1e-12 * cSb);
Mm  = A + U * C * V;
bwd = norm(b - Mm * (d.z - Yw)) / (norm(Mm) * norm(d.z - Yw) + norm(b));
[np, nf] = chk(np, nf, ...
    sprintf('T4c bwd == ||r||/(||A+UCV|| ||x|| + ||b||) (%.3e)', d.bwd), ...
    abs(d.bwd - bwd) <= 1e-12 * max(bwd, realmin));

% --- T5  total cancellation is reported as Inf, not NaN ------------------
% The alpha = 1e16 row of the stress sweep returns x == 0 exactly.  0/0 would make
% cancel_sub NaN and silently drop that row out of every max() and comparison --
% the one row that matters most.
[~, dz] = woodbury_naive(1, 1, 1e16, 1, 1, 1e-16);
[np, nf] = chk(np, nf, ...
    sprintf('T5  x == 0 gives cancel_sub = Inf, not NaN (%.3e)', dz.cancel_sub), ...
    isinf(dz.cancel_sub) && ~isnan(dz.cancel_sub));

% --- T6  the double-double reference is actually higher precision --------
mx_dd = 0;
for a = 10 .^ (0:2:18)
    mx_dd = max(mx_dd, abs(dd_woodbury_scalar(a) - 1/(1+a)) * (1+a));
end
[np, nf] = chk(np, nf, ...
    sprintf('T6  dd_woodbury_scalar accurate at every alpha (%.2e < 1e-13)', mx_dd), ...
    mx_dd < 1e-13);

% --- T7  it agrees with the sparse production path on the real problem ---
% The load-bearing test: woodbury_solve's frozen-factorization arithmetic and
% woodbury_naive's dense A\b must be the same method, or the constructed families
% say nothing about the benchmark.
sopts = struct('case_name', 'bar_rotating', 'h0', 0.1, 'dt', 0.02, ...
               'Tstep', 61, 'nsteps', 4, 'verify', false, ...
               'use_cache', true, 'quiet', true);
S   = build_stokes_sequence(sopts);
ctx = woodbury_context_init(S);
nC  = S.nC;
K1  = full(seq_K(S, ctx.ref));
Cm  = [zeros(nC), eye(nC); eye(nC), zeros(nC)];   % C == C^{-1} == B
mx_agree = 0;
for n = 2:S.nsteps
    b  = S.b{n};
    Un = full([full(seq_dCblk2(S, n, ctx.ref)), full(S.Sel)]);
    xn = woodbury_naive(K1, Un, Cm, Un', b);
    xw = woodbury_solve(ctx, S, n, b);
    mx_agree = max(mx_agree, norm(xn - xw) / norm(xw));
end
fprintf('  dense-naive vs sparse-frozen on %s: %.2e\n', S.case_name, mx_agree);
[np, nf] = chk(np, nf, ...
    sprintf('T7  same answer as woodbury_solve on the rotor (%.2e < 1e-11)', mx_agree), ...
    mx_agree < 1e-11);

fprintf('\n  %d passed, %d failed\n', np, nf);
if nf > 0
    error('test_woodbury_naive:fail', '%d assertion(s) failed.', nf);
end

%==========================================================================
function dC = seq_dCblk2(S, n, ref)
%SEQ_DCBLK2  The dC half of the update, without the [dC, Sel] packaging.
    [~, dC] = seq_dCblk(S, n, ref);
end

%==========================================================================
function [np, nf] = chk(np, nf, name, cond)
    if cond
        np = np + 1;
        fprintf('  PASS  %s\n', name);
    else
        nf = nf + 1;
        fprintf(2, '  FAIL  %s\n', name);
    end
end
