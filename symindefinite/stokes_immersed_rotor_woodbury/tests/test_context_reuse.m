%TEST_CONTEXT_REUSE  The frozen context is correct, stateless and paid for once.
%
%   Run:  cd tests; test_context_reuse
%
%   The whole cost argument of this study is "one factorization, then nC
%   backsolves per step".  That rests on the context being genuinely frozen:
%
%     * YSel = K_1^{-1}Sel must actually be that (checked against K_1 directly,
%       not against the routine that produced it);
%     * it must be solved ONCE -- the per-step count has to read nC, not 2nC;
%     * the context must carry no per-step state.  The sharpest test of that is
%       ORDER INDEPENDENCE: solving the sequence backwards must give the same
%       answers as forwards.  If any step leaked information into ctx, or if a
%       later call depended on an earlier one, that breaks here.
%
%   See also: woodbury_context_init, woodbury_solve.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
studyDir    = fileparts(thisFileDir);
addpath(studyDir);
add_woodbury_paths();
assert_woodbury_helpers();

rng(0);
fprintf('=== test_context_reuse ===\n');

sopts = struct('case_name', 'bar_rotating', 'h0', 0.1, 'dt', 0.02, ...
               'Tstep', 61, 'nsteps', 5, 'verify', true, ...
               'use_cache', true, 'quiet', true);
S   = build_stokes_sequence(sopts);
ctx = woodbury_context_init(S);
nC  = S.nC;

np = 0;  nf = 0;

% --- T1  one-time cost is nC backsolves ---------------------------------
[np, nf] = chk(np, nf, ...
    sprintf('T1  setup cost is nC = %d backsolves', nC), ...
    ctx.n_backsolves_setup == nC);

% --- T2  YSel really is K_1^{-1} Sel ------------------------------------
K1     = seq_K(S, ctx.ref);
relSel = norm(K1 * ctx.YSel - full(ctx.Sel), 'fro') / ...
         max(norm(full(ctx.Sel), 'fro'), eps);
[np, nf] = chk(np, nf, ...
    sprintf('T2  K_1 * YSel == Sel (%.2e < 1e-10)', relSel), ...
    relSel < 1e-10);

% --- T3  the cached block matches, and is symmetric ---------------------
SelYSel_ref = full(ctx.Sel' * ctx.YSel);
relBlk = norm(ctx.SelYSel - (SelYSel_ref + SelYSel_ref')/2, 'fro') / ...
         max(norm(ctx.SelYSel, 'fro'), eps);
[np, nf] = chk(np, nf, ...
    sprintf('T3a SelYSel == Sel''*YSel (%.2e < 1e-12)', relBlk), ...
    relBlk < 1e-12);
[np, nf] = chk(np, nf, ...
    'T3b SelYSel is exactly symmetric', ...
    isequal(ctx.SelYSel, ctx.SelYSel'));

% --- T4  forward pass, then a repeat call at the same step --------------
x_fwd = cell(S.nsteps, 1);
nbs   = nan(S.nsteps, 1);
for n = 1:S.nsteps
    [x_fwd{n}, info] = woodbury_solve(ctx, S, n, S.b{n});
    nbs(n) = info.n_backsolves;
end
[np, nf] = chk(np, nf, ...
    sprintf('T4  per-step cost is nC = %d, never 2nC = %d', nC, 2*nC), ...
    all(nbs == nC));

x_again = woodbury_solve(ctx, S, 3, S.b{3});
[np, nf] = chk(np, nf, ...
    'T5  repeated call at the same step is bit-for-bit identical', ...
    isequal(x_again, x_fwd{3}));

% --- T6  order independence: the context holds no per-step state --------
x_rev = cell(S.nsteps, 1);
for n = S.nsteps:-1:1
    x_rev{n} = woodbury_solve(ctx, S, n, S.b{n});
end
same = true;
for n = 1:S.nsteps
    same = same && isequal(x_rev{n}, x_fwd{n});
end
[np, nf] = chk(np, nf, ...
    'T6  solving the sequence backwards gives identical iterates', same);

% --- T7  the applier really inverts K_1, end to end ---------------------
% A wrong permutation or scaling in woodbury_apply_ref would still return
% plausible-looking vectors, so this is checked against K_1 directly.
fprintf('  nnz(K_1) = %d   nnz(L) = %d   fill = %.2f\n', ...
        ctx.nnzK1, ctx.nnzL, ctx.fill_ratio);
v      = randn(S.n, 1);
relInv = norm(woodbury_apply_ref(ctx, K1 * v) - v) / norm(v);
[np, nf] = chk(np, nf, ...
    sprintf('T7a K_1^{-1}(K_1 v) == v (%.2e < 1e-8)', relInv), ...
    relInv < 1e-8);
[np, nf] = chk(np, nf, ...
    sprintf('T7b fill ratio recorded and sane (%.2f)', ctx.fill_ratio), ...
    isfinite(ctx.fill_ratio) && ctx.fill_ratio >= 1);

% --- T8  the applier BATCHES across columns -----------------------------
% The entire cost argument of this study is that nC backsolves are cheap relative
% to a refactorization, and that holds only because the sparse triangular solves
% batch.  MATLAB's decomposition object does NOT batch -- it costs nC times a
% single solve, which made the method look ~5x slower than refactorizing before
% the applier was rewritten.  This pins the property, with a deliberately loose
% factor so it tests batching rather than the machine's mood.
% The 0.75 factor is deliberately loose.  A non-batching implementation scales as
% ratio ~= nC (measured: decomposition gives 0.47 ms/column at 1 AND at nC
% columns); batching gives 7.8 at nC=16 / h0=0.1 and 4.4 at nC=20 / h0=0.05.  The
% threshold separates those regimes without measuring the machine's mood.
R1 = randn(S.n, 1);
Rk = randn(S.n, nC);
t1 = min([timeit(@() woodbury_apply_ref(ctx, R1)), ...
          timeit(@() woodbury_apply_ref(ctx, R1)), ...
          timeit(@() woodbury_apply_ref(ctx, R1))]);
tk = min([timeit(@() woodbury_apply_ref(ctx, Rk)), ...
          timeit(@() woodbury_apply_ref(ctx, Rk)), ...
          timeit(@() woodbury_apply_ref(ctx, Rk))]);
fprintf('  apply: 1 col %.3f ms | %d cols %.3f ms | ratio %.2f (nC = %d)\n', ...
        1e3*t1, nC, 1e3*tk, tk/t1, nC);
[np, nf] = chk(np, nf, ...
    sprintf('T8  %d columns cost < 0.75*nC single solves (ratio %.2f < %.1f)', ...
            nC, tk/t1, 0.75*nC), ...
    tk < 0.75 * nC * t1);

% --- T9  the applier beats the decomposition object it replaced ----------
% This is the claim woodbury_apply_ref's docstring rests on, and it is the reason
% the study concludes the update is CHEAPER than refactorizing rather than
% several times more expensive.  If a future MATLAB release fixes
% decomposition's multi-column path, this test failing is the signal to simplify
% the applier away -- not a defect.
dK   = decomposition(K1);
tdec = min([timeit(@() dK \ Rk), timeit(@() dK \ Rk)]);
fprintf('  %d cols: applier %.3f ms | decomposition %.3f ms | speedup %.1fx\n', ...
        nC, 1e3*tk, 1e3*tdec, tdec/tk);
[np, nf] = chk(np, nf, ...
    sprintf('T9  applier >= 2x faster than decomposition (%.1fx)', tdec/tk), ...
    tk < 0.5 * tdec);

fprintf('\n  %d passed, %d failed\n', np, nf);
if nf > 0
    error('test_context_reuse:fail', '%d assertion(s) failed.', nf);
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
