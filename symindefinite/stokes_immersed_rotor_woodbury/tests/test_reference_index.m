%TEST_REFERENCE_INDEX  The frozen reference may be any step, and the gate that
%   guards it must fire on singularity, not on ill conditioning.
%
%   Run:  cd tests; test_reference_index
%
%   TWO PROPERTIES, ONE FILE, because they are the same change.
%
%   (1) woodbury_context_init takes an optional REF.  Every other test in this
%       folder freezes step 1, so nothing else covers ref != 1 -- and the whole
%       Woodbury identity is stated relative to whichever step is frozen, so
%       exactness must not depend on which one that is.  R3 is the load-bearing
%       check: it solves EVERY step from EVERY reference.
%
%       REF is not a refresh cadence.  There is still exactly one factorization
%       per context and no path that refactorizes mid-run; R7 pins that by
%       asserting the cost invariant is independent of REF.
%
%   (2) The applier check gates on ||K_ref Y - Sel||/||Sel||, which is bounded
%       below by cond(K_ref)*eps.  A FIXED 1e-8 threshold therefore asserts
%       cond(K_ref) < ~4.5e7 and reports any worse reference as "badApply -- the
%       ldl permutation/scaling convention has changed", which is a misdiagnosis:
%       the convention is fine and the operator is merely ill conditioned.  R9
%       pins that an ill-conditioned but INVERTIBLE reference is accepted and
%       still solves; R10 pins that a genuinely singular one is still refused,
%       and refused by name.
%
%   See also: woodbury_context_init, woodbury_solve, test_context_reuse.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
studyDir    = fileparts(thisFileDir);
addpath(studyDir);
add_woodbury_paths();
assert_woodbury_helpers();

rng(0);
fprintf('=== test_reference_index ===\n');

sopts = struct('case_name', 'bar_rotating', 'h0', 0.1, 'dt', 0.02, ...
               'Tstep', 61, 'nsteps', 5, 'verify', true, ...
               'use_cache', true, 'quiet', true);
S  = build_stokes_sequence(sopts);
nC = S.nC;
ns = S.nsteps;

np = 0;  nf = 0;

% --- R1  the default did not move ---------------------------------------
c0 = woodbury_context_init(S);
c1 = woodbury_context_init(S, 1);
same = c0.ref == 1 && c1.ref == 1 && ...
       isequal(c0.perm, c1.perm) && isequal(c0.Sscale, c1.Sscale) && ...
       isequal(c0.YSel, c1.YSel) && isequal(c0.SelYSel, c1.SelYSel);
[np, nf] = chk(np, nf, 'R1  woodbury_context_init(S) is bit-for-bit (S,1)', same);

% --- R2  every reference inverts its own K ------------------------------
ctxs = cell(ns, 1);
relY = nan(ns, 1);
for r = 1:ns
    ctxs{r} = woodbury_context_init(S, r);
    relY(r) = norm(seq_K(S, r) * ctxs{r}.YSel - full(S.Sel), 'fro') / ...
              max(norm(full(S.Sel), 'fro'), eps);
end
[np, nf] = chk(np, nf, ...
    sprintf('R2  K_r * YSel == Sel for every r (worst %.2e < 1e-10)', max(relY)), ...
    all(cellfun(@(c) c.ref, ctxs)' == (1:ns)) && all(relY < 1e-10));

% --- R3  exactness is independent of WHICH step is frozen ---------------
% The property the whole revision rests on.  Nothing else in the suite runs
% woodbury_solve from a reference other than step 1.
worst = 0;  worst_rn = [0 0];
for r = 1:ns
    for n = 1:ns
        xw = woodbury_solve(ctxs{r}, S, n, S.b{n});
        xd = seq_K(S, n) \ S.b{n};
        e  = norm(xw - xd) / max(norm(xd), eps);
        if e > worst, worst = e;  worst_rn = [r n]; end
    end
end
[np, nf] = chk(np, nf, ...
    sprintf('R3  every (ref %d, step %d) reproduces K_n\\b (worst %.2e < 1e-10)', ...
            worst_rn(1), worst_rn(2), worst), ...
    worst < 1e-10);

% --- R4  at n == r the update is a no-op, computed not branched ---------
okR4 = true;
for r = 1:ns
    [x, info] = woodbury_solve(ctxs{r}, S, r, S.b{r});
    [~, dC]   = seq_dCblk(S, r, r);
    xa = woodbury_apply_ref(ctxs{r}, S.b{r});
    okR4 = okR4 && info.dC_is_zero && nnz(dC) == 0 && ...
           norm(x - xa) / max(norm(xa), eps) < 1e-10;
end
[np, nf] = chk(np, nf, 'R4  n == r gives dC == 0 and x == K_r^{-1}b', okR4);

% --- R5  the drift denominator follows the reference --------------------
okR5 = true;
for r = 1:ns
    okR5 = okR5 && ctxs{r}.Cblk_ref_normF == norm(S.Cblk{r}, 'fro');
end
[~, iR5] = woodbury_solve(ctxs{ns}, S, 1, S.b{1});
[~, dC1] = seq_dCblk(S, 1, ns);
okR5 = okR5 && abs(iR5.dC_rel - norm(dC1,'fro')/ctxs{ns}.Cblk_ref_normF) < 1e-14;
[np, nf] = chk(np, nf, 'R5  Cblk_ref_normF and info.dC_rel track the reference', okR5);

% --- R6  a bad ref is refused, by name ----------------------------------
bad  = {0, 1.5, ns + 1, [1 2], '2'};
okR6 = true;
for k = 1:numel(bad)
    try
        woodbury_context_init(S, bad{k});
        okR6 = false;                              % should not reach here
        fprintf(2, '    R6: ref = %s was accepted\n', mat2str(bad{k}));
    catch ME
        okR6 = okR6 && contains(lower(ME.message), 'ref');
    end
end
[np, nf] = chk(np, nf, 'R6  ref = {0, 1.5, nsteps+1, [1 2], ''2''} each throw naming ref', okR6);

% --- R7  the cost invariant does not depend on the anchor ---------------
% This is the answer to "is REF the forbidden refresh knob".  One factorization,
% nC backsolves of setup, whatever the anchor is.
f0   = sort(fieldnames(c0));
okR7 = true;
for r = 1:ns
    okR7 = okR7 && ctxs{r}.n_backsolves_setup == nC && ...
           isequal(sort(fieldnames(ctxs{r})), f0);
end
[np, nf] = chk(np, nf, ...
    sprintf('R7  n_backsolves_setup == nC = %d and the field set is fixed', nC), okR7);

% --- R8  rcond_D is reported on the SUCCESS path ------------------------
% It was previously computed only inside the failure branch, so no healthy run
% could report the conditioning its own gate is expressed in.
rc = cellfun(@(c) c.rcond_D, ctxs);
kk = condest(seq_K(S, 1));
[np, nf] = chk(np, nf, ...
    sprintf('R8a rcond_D present, finite, in (0,1] (worst %.2e)', min(rc)), ...
    all(isfinite(rc)) && all(rc > 0) && all(rc <= 1));
[np, nf] = chk(np, nf, ...
    sprintf('R8b 1/rcond_D within 3 decades of condest(K_1) (%.2e vs %.2e)', ...
            1/rc(1), kk), ...
    abs(log10((1/rc(1)) / kk)) < 3);

% --- R9  an ill-conditioned but INVERTIBLE reference is accepted --------
% The ill conditioning has to be somewhere Sel can SEE it, or apply_relres never
% moves and the loosened branch is never entered -- an earlier version of this
% test scaled a pressure row, reached condest(K_1) = 1.3e14, and still measured
% apply_relres = 3e-12, i.e. it asserted nothing.  Weakening one CONSTRAINT
% instead puts the near-null direction squarely in the multiplier block, which is
% both what Sel selects and what the physical mechanism does (a rotor bar aligned
% with the mesh axes loses rank in exactly that block).  K_ref stays symmetric
% and invertible, and K_n = K_r + U B U' still holds because seq_dCblk forms the
% difference from the same modified Cblk.
% cond(K_ref) scales as tau^-2 here, and the window is narrow at both ends:
% tau = 1e-5 gives 7.3e11, whose apply_relres (1.6e-9) still clears the old fixed
% gate and so proves nothing; tau = 1e-7 gives 7.3e15, past 1/eps, where
% singularReference correctly fires.  tau = 1e-6 lands at ~7e13, inside the band
% the loosening exists for.
%
% The SAME column is weakened at every step, not only at the reference.  Weakening
% it at the reference alone leaves dC carrying the full restoration of a constraint
% that is 1e6 times too weak, which drives rcond(Cap) to 7.6e-18 -- a genuine
% capacitance breakdown.  That is a real phenomenon and the experiment's subject,
% but it is not what R9 is about: here the reference must be ill conditioned while
% the UPDATE stays well posed, so the accuracy claim tests the gate and not a
% second failure mode piled on top.
Sill = S;
for n = 1:ns
    Sill.Cblk{n}(:, 1) = 1e-6 * Sill.Cblk{n}(:, 1);
end
kill = condest(seq_K(Sill, 1));
ws = warning('off', 'all');
try
    cill = woodbury_context_init(Sill, 1);
    idl  = '';
catch ME
    cill = [];
    idl  = ME.identifier;
end
warning(ws);
[np, nf] = chk(np, nf, ...
    sprintf('R9a condest(K_1) = %.2e reference accepted (got "%s")', kill, idl), ...
    isempty(idl));
if isempty(idl)
    fprintf('  ill-conditioned reference: apply_relres = %.2e, rcond_D = %.2e, cond_ref = %.2e\n', ...
            cill.apply_relres, cill.rcond_D, cill.cond_ref);
    ws  = warning('off', 'all');
    xw  = woodbury_solve(cill, Sill, 3, Sill.b{3});
    xd  = seq_K(Sill, 3) \ Sill.b{3};
    warning(ws);
    eR9 = norm(xw - xd) / max(norm(xd), eps);
else
    eR9 = inf;
end
% Non-vacuity: the point of R9 is the branch the OLD fixed 1e-8 gate would have
% rejected.  If apply_relres sits below 1e-8 the fixture stopped stressing it and
% R9a is passing for the wrong reason.
[np, nf] = chk(np, nf, ...
    sprintf('R9b the loosened branch actually ran (apply_relres %.2e > 1e-8)', ...
            local_get(cill, 'apply_relres')), ...
    ~isempty(cill) && cill.apply_relres > 1e-8 && isfinite(cill.cond_ref));
[np, nf] = chk(np, nf, ...
    sprintf('R9c ...and it still solves step 3 (%.2e < 1e-6)', eR9), eR9 < 1e-6);

% --- R10  a singular reference is still refused, by name ----------------
% Regression guard on test_context_reuse T10, now at a reference other than 1:
% loosening the gate for ill conditioning must not loosen it for singularity.
% A PRESSURE row: Cblk writes into the velocity rows and Sel into the multiplier
% rows, so a pressure row is the one place a zeroed row/column pair survives into
% K_ref unchanged, keeping the failure genuine singularity rather than a rejected
% factorization.
prow = S.nU + 1;
Sbad = S;
Sbad.K0(prow, :) = 0;
Sbad.K0(:, prow) = 0;
ws = warning('off', 'all');
try
    woodbury_context_init(Sbad, 3);
    idb = '(no error raised)';
catch ME
    idb = ME.identifier;
end
warning(ws);
[np, nf] = chk(np, nf, ...
    sprintf('R10 singular reference at r = 3 named singularReference (%s)', idb), ...
    strcmp(idb, 'woodbury_context_init:singularReference'));

fprintf('\n  %d passed, %d failed\n', np, nf);
if nf > 0
    error('test_reference_index:fail', '%d assertion(s) failed.', nf);
end

%==========================================================================
function v = local_get(ctx, f)
%LOCAL_GET  Field of a context that may be [] because its init threw.
    if isempty(ctx), v = NaN; else, v = ctx.(f); end
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
