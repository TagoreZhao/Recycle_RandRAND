%TEST_RECURSIVE_GROWTH  Chaining Woodbury updates does not compound the error.
%
%   Run:  cd tests; test_recursive_growth
%
%   The intuition that a sequence of low-rank updates must degrade -- each one
%   inheriting and amplifying the last one's rounding -- is the reason the
%   production path re-anchors dC against a FIXED reference every step.  This test
%   builds the chained scheme that intuition warns about and measures it, so the
%   design choice rests on a number rather than on the intuition.
%
%   The two arms compute the same matrix inverse: the incremental updates
%   Cblk_k - Cblk_{k-1} telescope to the total Cblk_n - Cblk_1 (T1 pins this).  So
%   every difference between them is floating point, and a chain that compounded
%   would show as growth in the ratio.  It does not: the ratio is flat in depth and
%   the per-level cancellation is the reason (T7).
%
%   This is a REGRESSION GUARD ON A NEGATIVE RESULT.  If a future change makes the
%   chain degrade, T4/T5 fail and the negative result has to be re-earned.
%
%   See also: woodbury_chain_build, woodbury_chain_apply, run_woodbury_recursive.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
studyDir    = fileparts(thisFileDir);
addpath(studyDir);
add_woodbury_paths();
assert_woodbury_helpers();

rng(0);
NLEVEL = 12;
fprintf('=== test_recursive_growth (%d levels) ===\n', NLEVEL);

sopts = struct('case_name', 'bar_rotating', 'h0', 0.1, 'dt', 0.02, ...
               'Tstep', 61, 'nsteps', NLEVEL, 'verify', false, ...
               'use_cache', true, 'quiet', true);
S   = build_stokes_sequence(sopts);
ctx = woodbury_context_init(S);
lev = woodbury_chain_build(ctx, S, NLEVEL);

np = 0;  nf = 0;

% --- T1  the premise: the increments telescope ---------------------------
% Without this the two arms are not solving the same problem and nothing below
% means anything.
acc = 0 * S.Cblk{1};
for k = 2:NLEVEL
    acc = acc + (S.Cblk{k} - S.Cblk{k-1});
end
tel = norm(acc - (S.Cblk{NLEVEL} - S.Cblk{1}), 'fro') / ...
      max(norm(S.Cblk{NLEVEL} - S.Cblk{1}, 'fro'), eps);
[np, nf] = chk(np, nf, ...
    sprintf('T1  incremental updates telescope to the total (%.2e < 1e-14)', tel), ...
    tel < 1e-14);

% --- collect both arms ---------------------------------------------------
lv   = (2:NLEVEL)';
er   = nan(size(lv));  ef = nan(size(lv));
csub = nan(size(lv));  differ = false;
for i = 1:numel(lv)
    n    = lv(i);
    b    = S.b{n};
    xref = S.xref{n};
    xr   = woodbury_chain_apply(ctx, lev, n, b);
    [xf, info] = woodbury_solve(ctx, S, n, b);

    er(i)   = norm(xr - xref) / norm(xref);
    ef(i)   = norm(xf - xref) / norm(xref);
    zr      = woodbury_chain_apply(ctx, lev, n - 1, b);
    csub(i) = (norm(zr) + norm(xr - zr)) / max(norm(xr), realmin);
    differ  = differ || ~isequal(xr, xf);
end
ratio = er ./ max(ef, realmin);
slope = polyfit(lv, log10(er), 1);
fprintf('  recursive %.2e -> %.2e | fixed %.2e -> %.2e | ratio med %.2f max %.2f\n', ...
        er(1), er(end), ef(1), ef(end), median(ratio), max(ratio));

% --- T2  non-vacuity: the arms really are different arithmetic -----------
% If woodbury_chain_apply silently collapsed to the production path, every
% assertion below would pass while testing nothing.
[np, nf] = chk(np, nf, ...
    'T2  the two arms produce different iterates (chain is really chained)', differ);

% --- T3  both are accurate ----------------------------------------------
[np, nf] = chk(np, nf, ...
    sprintf('T3  both arms accurate at every level (rec %.2e, fix %.2e < 1e-12)', ...
            max(er), max(ef)), ...
    max(er) < 1e-12 && max(ef) < 1e-12);

% --- T4  no compounding relative to the fixed reference ------------------
[np, nf] = chk(np, nf, ...
    sprintf('T4  chaining costs a bounded factor (median %.2f < 5)', median(ratio)), ...
    median(ratio) < 5);

% --- T5  and no growth with depth ---------------------------------------
% A chain that amplified would show a positive slope in log10(error) vs level.
% 0.05/level is a factor of 10 per 20 levels -- far below anything the intuition
% predicts, and far above the observed scatter.
[np, nf] = chk(np, nf, ...
    sprintf('T5  log10(err) does not grow with depth (slope %.4f/level < 0.05)', ...
            slope(1)), ...
    slope(1) < 0.05);

% --- T6  the same at the operator level, not just for one RHS ------------
n  = NLEVEL;
Kn = seq_K(S, n);
dr = 0;  df = 0;
for p = 1:3
    v  = randn(S.n, 1);
    xt = Kn \ v;
    dr = max(dr, norm(woodbury_chain_apply(ctx, lev, n, v) - xt) / norm(xt));
    df = max(df, norm(woodbury_solve(ctx, S, n, v) - xt) / norm(xt));
end
[np, nf] = chk(np, nf, ...
    sprintf('T6  operator drift bounded and comparable (rec %.2e, fix %.2e)', dr, df), ...
    dr < 1e-12 && dr < 10 * df);

% --- T7  WHY there is no compounding ------------------------------------
% Each level's final subtraction is inert, so each correction is individually
% backward stable and the errors add instead of amplifying.  This is the assertion
% that explains T4/T5 rather than merely restating them -- and the one that would
% fire first if the sequence ever moved into a regime where chaining does hurt.
[np, nf] = chk(np, nf, ...
    sprintf('T7  per-level cancel_sub stays O(1) (max %.3f < 10)', max(csub)), ...
    max(csub) < 10);

fprintf('\n  %d passed, %d failed\n', np, nf);
if nf > 0
    error('test_recursive_growth:fail', '%d assertion(s) failed.', nf);
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
