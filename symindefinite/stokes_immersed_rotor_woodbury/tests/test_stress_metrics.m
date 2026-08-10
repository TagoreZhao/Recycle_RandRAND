%TEST_STRESS_METRICS  Pin Woodbury's floating-point failure on constructed systems.
%
%   Run:  cd tests; test_stress_metrics
%
%   The rotor benchmark reports a Woodbury forward error of ~2e-14, which is only
%   evidence about the METHOD if the method is capable of failing.  This test makes
%   it fail, on two systems whose condition numbers are all O(1), so that the benign
%   result on the physical sequence is a measured negative rather than an assumption.
%
%   FAMILY 1 -- the final subtraction.  A = [1], U = V = [1], C = [alpha], b = [1].
%   The exact solution is x = 1/(1+alpha).  Woodbury evaluates it as
%
%       z = 1,  Y = 1,  S = 1/alpha + 1,  w = alpha/(1+alpha),  x = z - Y w
%
%   i.e. as 1 - alpha/(1+alpha): two numbers of size 1 subtracted to produce an
%   answer of size 1e-16.  kappa(A) = kappa(A+UCV) = kappa(S) = 1 throughout -- the
%   system could not be better conditioned -- yet near alpha = 1e16 the subtraction
%   returns exactly 0 and every digit is gone.
%
%   FAMILY 2 -- the small matrix.  With a0 = 1/3, A = [a0(1+eta)], C = [-a0],
%   U = V = [1], b = [1].  Then S = C^{-1} + V A^{-1} U = -1/a0 + 1/A cancels two
%   numbers of size 3 down to one of size 3*eta, so its one-ulp absolute error is an
%   O(eps/eta) RELATIVE error.  Here it is the small matrix that fails while the
%   final subtraction is harmless -- the two mechanisms are separated on purpose,
%   and family 2's cancel_sub stays at exactly 1 throughout.
%
%   THE REFERENCE IS EXACT, which is what makes this evidence rather than two
%   inaccurate numbers being compared.  A and C = -a0 are doubles of the same
%   magnitude, so A + UCV = A - a0 is exact by Sterbenz and the solution is exactly
%   1/(A - a0) -- known in closed form, never solved for.
%
%   WHY a0 = 1/3 AND NOT 1.  The obvious choice A = 1+delta is DEGENERATE: the true
%   1/(1+delta) = 1 - delta + delta^2 - ... sits within delta^2 of the representable
%   1-delta, so fl(1/A) carries a rounding of delta^2 rather than of eps and the
%   mechanism never fires (the realized error saturates near sqrt(eps) ~ 1e-8).
%   Anchoring at a non-power-of-two puts 1/A in a generic position between doubles,
%   the rounding is a full ulp, and the error tracks cancel_S*eps across 12 decades.
%
%   Both families keep every condition number below 10, which is the whole point:
%   these are not ill-conditioned problems being solved badly, they are
%   well-conditioned problems being EVALUATED badly.
%
%   See also: woodbury_naive, dd_woodbury_scalar, run_woodbury_scalar_stress.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
studyDir    = fileparts(thisFileDir);
addpath(studyDir);

fprintf('=== test_stress_metrics ===\n');
fprintf('  eps = %.4e\n\n', eps);

np = 0;  nf = 0;

% =========================================================================
% FAMILY 1: the alpha sweep
% =========================================================================
ALPHA = 10 .^ (0:18)';
na    = numel(ALPHA);

f1 = struct('fwd', zeros(na,1), 'fwd_bs', zeros(na,1), 'fwd_dd', zeros(na,1), ...
            'bwd', zeros(na,1), 'bwd_bs', zeros(na,1), ...
            'kA', zeros(na,1), 'kM', zeros(na,1), 'kS', zeros(na,1), ...
            'cS', zeros(na,1), 'cSub', zeros(na,1));

fprintf('%9s %10s %10s %10s %9s %9s %9s %9s %10s\n', 'alpha', 'err_wood', ...
        'err_bslash', 'err_dd', 'k(A)', 'k(A+UCV)', 'k(S)', 'cancelS', 'cancelSub');
for i = 1:na
    a = ALPHA(i);
    A = 1;  U = 1;  V = 1;  C = a;  b = 1;
    xex = 1 / (1 + a);                       % exact, and exact to eps in double

    [~, d] = woodbury_naive(A, U, C, V, b, xex);

    M    = A + U * C * V;
    xbs  = M \ b;
    rbs  = b - M * xbs;

    f1.fwd(i)    = d.fwd;
    f1.fwd_bs(i) = abs(xbs - xex) / abs(xex);
    f1.fwd_dd(i) = abs(dd_woodbury_scalar(a) - xex) / abs(xex);
    f1.bwd(i)    = d.bwd;
    f1.bwd_bs(i) = norm(rbs) / (norm(M) * norm(xbs) + norm(b));
    f1.kA(i)     = d.kappa_A;
    f1.kM(i)     = d.kappa_M;
    f1.kS(i)     = d.kappa_S;
    f1.cS(i)     = d.cancel_S;
    f1.cSub(i)   = d.cancel_sub;

    fprintf('%9.1e %10.3e %10.3e %10.3e %9.2f %9.2f %9.2f %9.2e %10.3e\n', ...
            a, f1.fwd(i), f1.fwd_bs(i), f1.fwd_dd(i), f1.kA(i), f1.kM(i), ...
            f1.kS(i), f1.cS(i), f1.cSub(i));
end

i16 = find(ALPHA == 1e16, 1);

% --- T1  the evaluation is right where nothing cancels -------------------
[np, nf] = chk(np, nf, ...
    sprintf('T1  alpha = 1: naive Woodbury is exact (%.2e < 1e-15)', f1.fwd(1)), ...
    f1.fwd(1) < 1e-15);

% --- T2  the premise: nothing here is ill conditioned --------------------
% If any of these grew, the failure below could be blamed on the problem rather
% than on the formula.  They do not: all three stay at 1.
[np, nf] = chk(np, nf, ...
    sprintf('T2  every condition number stays O(1) (max k(A)=%.2f k(M)=%.2f k(S)=%.2f)', ...
            max(f1.kA), max(f1.kM), max(f1.kS)), ...
    max([f1.kA; f1.kM; f1.kS]) < 10);

% --- T3  family 1 does NOT fail through the small matrix -----------------
[np, nf] = chk(np, nf, ...
    sprintf('T3  small-matrix cancellation absent in family 1 (max cancel_S = %.2e < 10)', ...
            max(f1.cS)), ...
    max(f1.cS) < 10);

% --- T4  it fails through the final subtraction --------------------------
[np, nf] = chk(np, nf, ...
    sprintf('T4  final-subtraction factor blows up at alpha=1e16 (%.2e > 1e15)', ...
            f1.cSub(i16)), ...
    f1.cSub(i16) > 1e15);

[np, nf] = chk(np, nf, ...
    sprintf('T5  naive Woodbury loses EVERY digit at alpha=1e16 (fwd err %.3f >= 0.5)', ...
            f1.fwd(i16)), ...
    f1.fwd(i16) >= 0.5);

% Not merely forward-inaccurate: the computed iterate does not even solve a nearby
% system, so this is not a conditioning story dressed up as an accuracy story.
[np, nf] = chk(np, nf, ...
    sprintf('T6  and it is not backward stable either (bwd err %.3e > 1e-3)', ...
            f1.bwd(i16)), ...
    f1.bwd(i16) > 1e-3);

% --- T7  the same system solved directly is untouched --------------------
[np, nf] = chk(np, nf, ...
    sprintf('T7  direct solve unaffected at alpha=1e16 (fwd %.2e, bwd %.2e < 1e-14)', ...
            f1.fwd_bs(i16), f1.bwd_bs(i16)), ...
    f1.fwd_bs(i16) < 1e-14 && f1.bwd_bs(i16) < 1e-14);

% --- T8  precision, not formulation, is what was lost --------------------
% Same expression, same order of operations, ~32 digits: the answer comes back.
[np, nf] = chk(np, nf, ...
    sprintf('T8  double-double Woodbury survives alpha=1e16 (%.2e < 1e-12)', ...
            f1.fwd_dd(i16)), ...
    f1.fwd_dd(i16) < 1e-12);

% --- T9  the error is PREDICTED by cancel_sub, not by kappa --------------
% Upper bound cancel_sub*eps, checked over the whole sweep.  The alpha=1e16 row has
% cancel_sub = Inf (the computed x is exactly zero), which satisfies it trivially.
ratio = f1.fwd ./ max(f1.cSub * eps, realmin);
[np, nf] = chk(np, nf, ...
    sprintf('T9  fwd err <= 4*cancel_sub*eps at every alpha (max ratio %.2f)', ...
            max(ratio)), ...
    all(ratio <= 4));

% =========================================================================
% FAMILY 2: cancellation inside the small matrix
% =========================================================================
fprintf('\n--- family 2: S = C^{-1} + V A^{-1} U formed by cancellation ---\n');
A0  = 1 / 3;                                 % the anchor: NOT a power of two
ETA = 10 .^ -(15.3:-1:4.3)';                 % offset off the decades, see header
nd  = numel(ETA);

f2 = struct('fwd', zeros(nd,1), 'fwd_bs', zeros(nd,1), 'kA', zeros(nd,1), ...
            'kM', zeros(nd,1), 'kS', zeros(nd,1), 'cS', zeros(nd,1), ...
            'cSub', zeros(nd,1), 'pred', zeros(nd,1));

fprintf('%10s %10s %10s %9s %9s %9s %10s %10s %10s\n', 'eta', 'err_wood', ...
        'err_bslash', 'k(A)', 'k(A+UCV)', 'k(S)', 'cancelS', 'cancelSub', 'cS*eps');
for i = 1:nd
    A = A0 * (1 + ETA(i));  U = 1;  V = 1;  C = -A0;  b = 1;
    xex = 1 / (A - A0);                      % A + UCV == A - a0 EXACTLY (Sterbenz)

    [~, d] = woodbury_naive(A, U, C, V, b, xex);

    M   = A + U * C * V;
    xbs = M \ b;

    f2.fwd(i)    = d.fwd;
    f2.fwd_bs(i) = abs(xbs - xex) / abs(xex);
    f2.kA(i)     = d.kappa_A;
    f2.kM(i)     = d.kappa_M;
    f2.kS(i)     = d.kappa_S;
    f2.cS(i)     = d.cancel_S;
    f2.cSub(i)   = d.cancel_sub;
    f2.pred(i)   = d.cancel_S * eps;

    fprintf('%10.3e %10.3e %10.3e %9.2f %9.2f %9.2f %10.3e %10.3e %10.3e\n', ...
            ETA(i), f2.fwd(i), f2.fwd_bs(i), f2.kA(i), f2.kM(i), f2.kS(i), ...
            f2.cS(i), f2.cSub(i), f2.pred(i));
end

i14 = 2;                                     % eta ~ 5e-15, deep in the bad regime

[np, nf] = chk(np, nf, ...
    sprintf('T10 family 2 is well conditioned too (max k(A)=%.2f k(M)=%.2f k(S)=%.2f)', ...
            max(f2.kA), max(f2.kM), max(f2.kS)), ...
    max([f2.kA; f2.kM; f2.kS]) < 10);

[np, nf] = chk(np, nf, ...
    sprintf('T11 small-matrix cancellation IS the mechanism (cancel_S %.2e > 1e10)', ...
            f2.cS(i14)), ...
    f2.cS(i14) > 1e10);

% The separation that makes the two families independent evidence: here the final
% subtraction is harmless, so T12's error cannot be the family-1 mechanism in
% disguise.
[np, nf] = chk(np, nf, ...
    sprintf('T12 while the final subtraction is harmless (cancel_sub %.2e < 10)', ...
            f2.cSub(i14)), ...
    f2.cSub(i14) < 10);

[np, nf] = chk(np, nf, ...
    sprintf('T13 and Woodbury still loses ~13 digits (fwd err %.2e > 1e-3)', ...
            f2.fwd(i14)), ...
    f2.fwd(i14) > 1e-3);

% The same system solved directly.  A + UCV is (1+delta)-1, exact by Sterbenz, so
% this is not backslash being lucky -- there is nothing here to get wrong.
[np, nf] = chk(np, nf, ...
    sprintf('T14 direct solve is exact on family 2 (max err %.2e < 1e-15)', ...
            max(f2.fwd_bs)), ...
    max(f2.fwd_bs) < 1e-15);

% Predicted by cancel_S the way family 1 was predicted by cancel_sub: the two
% factors are not interchangeable diagnostics, each governs its own mechanism.
% Two-sided, unlike T9 -- here the bound is not merely respected, it is TIGHT, so a
% change that made the error 100x smaller would fail this too.
ratio2 = f2.fwd ./ max(f2.pred, realmin);
[np, nf] = chk(np, nf, ...
    sprintf('T15 fwd err tracks cancel_S*eps at every eta (ratio in [%.3f, %.3f])', ...
            min(ratio2), max(ratio2)), ...
    all(ratio2 <= 4) && all(ratio2 >= 0.01));

% Across 12 decades, not at one lucky eta: the error IS the cancellation factor.
lc = corr(log10(f2.fwd), log10(f2.cS));
[np, nf] = chk(np, nf, ...
    sprintf('T16 log(err) vs log(cancel_S) correlate over 12 decades (r = %.4f)', lc), ...
    lc > 0.99);

fprintf('\n  %d passed, %d failed\n', np, nf);
if nf > 0
    error('test_stress_metrics:fail', '%d assertion(s) failed.', nf);
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
