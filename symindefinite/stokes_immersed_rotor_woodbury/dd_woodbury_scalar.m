function x = dd_woodbury_scalar(alpha)
%DD_WOODBURY_SCALAR  The scalar Woodbury expression in ~32-digit arithmetic.
%   X = DD_WOODBURY_SCALAR(ALPHA)
%
%   Evaluates, for A = U = V = b = 1 and C = alpha,
%
%       S = 1/alpha + 1,   w = 1/S,   x = 1 - w
%
%   which is Woodbury's answer to (1 + alpha) x = 1, using double-double arithmetic
%   (two doubles carrying ~32 significant digits).  The result is rounded back to a
%   single double on return.
%
%   WHY THIS EXISTS.  In double precision this expression returns exactly 0 near
%   alpha = 1e16 while the true answer is 1e-16.  There are two possible readings:
%   the FORMULA is wrong for this problem, or the PRECISION is insufficient for the
%   formula.  Running the identical expression, in the identical order, at 32 digits
%   settles it -- the answer comes back, so the identity is fine and it is the
%   working precision that the cancellation consumed.  That distinction is the whole
%   content of "Woodbury is unstable": it is not a bad formula, it is a formula that
%   spends precision it does not have to spend.
%
%   No Symbolic Math Toolbox is installed here, so this is the higher-precision
%   reference.  The arithmetic is the standard Dekker/Knuth error-free
%   transformations (two_sum, two_prod via splitting) composed into add/mul/div, as
%   in Bailey's QD library -- about 1e-32 relative accuracy, ample for a comparison
%   against a double-precision answer.
%
%   See also: woodbury_naive, run_woodbury_scalar_stress.

    [ch, cl] = dd_div(1, 0, alpha, 0);        % C^{-1} = 1/alpha
    [sh, sl] = dd_add(ch, cl, 1, 0);          % S     = C^{-1} + V A^{-1} U
    [wh, wl] = dd_div(1, 0, sh, sl);          % w     = S^{-1} (V A^{-1} b)
    [xh, ~]  = dd_add(1, 0, -wh, -wl);        % x     = z - Y w

    x = xh;                                   % xh is already the rounded double
end

%==========================================================================
% Error-free transformations
%==========================================================================
function [s, e] = two_sum(a, b)
%TWO_SUM  s = fl(a+b) and e = the exact rounding error, so a+b == s+e exactly.
    s  = a + b;
    bb = s - a;
    e  = (a - (s - bb)) + (b - bb);
end

function [s, e] = quick_two_sum(a, b)
%QUICK_TWO_SUM  two_sum specialized to |a| >= |b| (three flops instead of six).
    s = a + b;
    e = b - (s - a);
end

function [hi, lo] = split(a)
%SPLIT  Dekker: a == hi + lo with each holding 26 bits, so products are exact.
    c  = 134217729 * a;                       % 2^27 + 1
    hi = c - (c - a);
    lo = a - hi;
end

function [p, e] = two_prod(a, b)
%TWO_PROD  p = fl(a*b) and e = the exact rounding error (no FMA in MATLAB).
    p = a * b;
    [ah, al] = split(a);
    [bh, bl] = split(b);
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
end

%==========================================================================
% Double-double arithmetic
%==========================================================================
function [rh, rl] = dd_add(ah, al, bh, bl)
    [s1, e1] = two_sum(ah, bh);
    [s2, e2] = two_sum(al, bl);
    e1 = e1 + s2;
    [s1, e1] = quick_two_sum(s1, e1);
    e1 = e1 + e2;
    [rh, rl] = quick_two_sum(s1, e1);
end

function [rh, rl] = dd_mul(ah, al, bh, bl)
    [p, e] = two_prod(ah, bh);
    e = e + (ah * bl + al * bh);
    [rh, rl] = quick_two_sum(p, e);
end

function [rh, rl] = dd_div(ah, al, bh, bl)
%DD_DIV  Long division: three Newton-style correction terms, each exact to a double.
    q1 = ah / bh;
    [th, tl] = dd_mul(bh, bl, q1, 0);
    [rh, rl] = dd_add(ah, al, -th, -tl);      % remainder a - q1*b

    q2 = rh / bh;
    [th, tl] = dd_mul(bh, bl, q2, 0);
    [rh, ~]  = dd_add(rh, rl, -th, -tl);   % only the high word feeds the last term

    q3 = rh / bh;

    [qh, ql] = quick_two_sum(q1, q2);
    [rh, rl] = dd_add(qh, ql, q3, 0);
end
