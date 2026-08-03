function [x, fl, rr, it] = two_level_solve_local(K, b, tol, mit, P, V, tau, kind)
%TWO_LEVEL_SOLVE_LOCAL  Two-level solve, in the form the family needs.
%
%   [x, fl, rr, it] = TWO_LEVEL_SOLVE_LOCAL(K, b, tol, mit, P, V, tau, kind)
%
%   Same signature as src.precond.two_level_split_solve, plus a trailing KIND.
%   KIND defaults to P.defl_kind (set by make_case / chart_struct), and there is
%   no fallback beyond that: the SPD and indefinite schemes are different
%   operators solved by different Krylov methods, so guessing is not allowed.
%
%   kind = 'indef'   symmetric indefinite -- the production path, reproduced
%                    exactly.  MINRES on the split operator Ahat = C^-1 K C^-T
%                    with the coarse correction as the fifth argument:
%
%                        minres(Afun, C^-1 b, tol, mit, Pdef),   x = C^-T y
%
%                    Pdef = (I-VV') + sqrt(tau) V (V'Ahat^2 V)^{-1/2} V'.  Since
%                    MINRES applies its fifth argument as M^{-1}, the operator
%                    iterated on is Pdef*Ahat, NOT Pdef*Ahat*Pdef.  Pinned
%                    against src.precond.two_level_split_solve by test T23.
%
%   kind = 'spd'     SPD -- the reference path of
%                    Preconditioner_Recycle/report/ball_surface (and its copy at
%                    +src/+solver/solve_deflate_M_P.m:381-387).  PCG on K itself
%                    with the sandwich as a single left preconditioner:
%
%                        B = C^-T P C^-1,   pcg(K, b, tol, mit, B)
%
%                    P = (I-VV') + tau V (V'Ahat V)^{-1} V', built on Ahat and
%                    not on Ahat^2.  B is symmetric because P is, which is what
%                    PCG requires; the preconditioned operator B*K is similar to
%                    the SPD P^{1/2} Ahat P^{1/2}, so its spectrum is real and
%                    positive and CG is legal.
%
%   The two branches iterate on the SAME preconditioned operator they would in
%   production, so iteration counts are meaningful within a family.  They are
%   NOT directly comparable ACROSS families -- different Krylov method, and
%   captured modes land on tau (spd) versus +-sqrt(tau) (indef).
%
%   V = [] means "no coarse space": MINRES/PCG with the C-only preconditioner,
%   which is the right baseline for the "worse than nothing" measurements of
%   Obs 4.3.
%
%   Inputs match two_level_split_solve: K the step matrix, b the right-hand
%   side, tol/mit the Krylov settings, P a struct supplying .applyCinv = C^-1
%   and .applyCtinv = C^-T (an ILDL struct, a make_case struct, or a
%   chart_struct wrapper), V a dense orthonormal coarse basis or [], tau the
%   coarse weight.
%
%   rr is the relative residual reported by the underlying solver: of the SPLIT
%   operator for 'indef' (as in two_level_split_solve), of K itself for 'spd'
%   (as in the ball_surface reference).  Both branches are run to the same tol.
%
%   See also: coarse_correction, deflated_spectrum, make_case,
%             src.precond.two_level_split_solve.

    if nargin < 7 || isempty(tau), tau = 0.5; end
    if nargin < 8 || isempty(kind)
        if isstruct(P) && isfield(P, 'defl_kind') && ~isempty(P.defl_kind)
            kind = P.defl_kind;
        else
            error('two_level_solve_local:kind', ...
                  ['kind is required: pass it explicitly or set P.defl_kind.  ' ...
                   'The SPD and indefinite schemes are different operators.']);
        end
    end

    Afun = @(y) P.applyCinv(K * P.applyCtinv(y));   % Ahat = C^-1 K C^-T

    switch lower(kind)
        case 'indef'
            btil = P.applyCinv(b);                  % C^-1 b
            if isempty(V)
                [y, fl, rr, it] = minres(Afun, btil, tol, mit);          % C only
            else
                Pdef = coarse_correction(V, Afun, tau, 'indef', 'handle');
                [y, fl, rr, it] = minres(Afun, btil, tol, mit, Pdef);
            end
            x = P.applyCtinv(y);                    % recover x = C^-T y

        case 'spd'
            % B = C^-T P C^-1 applied to the ORIGINAL system, the ball_surface
            % pattern.  With V = [] this degenerates to B = C^-T C^-1 = M^-1,
            % the plain ichol preconditioner -- the correct no-coarse-space
            % baseline.
            if isempty(V)
                Bapply = @(r) P.applyCtinv(P.applyCinv(r));
            else
                Pdef   = coarse_correction(V, Afun, tau, 'spd', 'handle');
                Bapply = @(r) P.applyCtinv(Pdef(P.applyCinv(r)));
            end
            [x, fl, rr, it] = pcg(K, b, tol, mit, Bapply);

        otherwise
            error('two_level_solve_local:kind', ...
                  'unknown kind "%s" (expected ''spd'' or ''indef'')', kind);
    end
end
