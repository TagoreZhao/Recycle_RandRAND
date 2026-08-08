%TEST_GMRES_LOWRANK  Unit tests for the gmres_exact_inv_frozen solver arm.
%
%   Run:  cd tests; test_gmres_lowrank
%
%   The arm exists to test ONE claim.  Consecutive KKT systems differ only
%   through the moving coupling block,
%
%       K_n = K_1 + dK,   dK = [0 0 dC'; 0 0 0; dC 0 0],   rank(dK) = 2 rank(dC),
%
%   so preconditioning K_n on the LEFT with the exact SIGNED inverse of K_1 gives
%
%       K_1^-1 K_n = I + K_1^-1 dK,
%
%   an identity plus a rank-r update whose minimal polynomial has degree <= r+1.
%   Unrestarted GMRES must therefore terminate in at most r+1 iterations.  MINRES
%   cannot see this operator at all -- it needs an SPD preconditioner, which is
%   why the existing exact_ldl_frozen arm uses M = |K_1| and sees sign(D_1) plus
%   the same low-rank update instead.
%
%   T1-T3 pin the mathematics on a synthetic system whose update rank is known
%   exactly.  T4 pins the ENGINE CONTRACT (a scalar iteration count), which no
%   amount of correct mathematics would catch.  T5-T6 run the real KKT pair when
%   extract_kkt_examples has been run; they are skipped otherwise.
%
%   No solves of the full benchmark are run and nothing is written to disk.

clear; clc;
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(fileparts(fileparts(root)));   % repo root, for src.*
addpath(root);

np = 0; nf = 0;
fprintf('=== test_gmres_lowrank ===\n');

TOL   = 1e-10;
MAXIT = 300;

%% ------------------------------------------------ synthetic KKT sequence ----
% Same block shape as the benchmark: an SPD velocity block, a negative-definite
% stabilized pressure block and a zero (lambda,lambda) block, with only the
% coupling block moving between "steps".
rng(0);
r  = 6;                                % r Lagrange points -> nC = 2r coupling rows
nC = 2 * r;

Au = blkdiag(gallery('poisson', 11), gallery('poisson', 11));   % SPD, 2 components
nU = size(Au, 1);
nP = 120;
B  = sprandn(nP, nU, 0.05);
Lp = speye(nP) * 1e-3;

mk_K = @(C) [ Au,            B',            C'          ; ...
              B,            -Lp,            sparse(nP, nC) ; ...
              C,             sparse(nC, nP), sparse(nC, nC) ];

C1 = sprandn(nC, nU, 0.2);
C9 = sprandn(nC, nU, 0.2);
K1 = mk_K(C1);   K1 = (K1 + K1')/2;
K9 = mk_K(C9);   K9 = (K9 + K9')/2;
n  = size(K1, 1);

b = randn(n, 1);
dec1 = decomposition(K1, 'ldl');
Minv = @(v) dec1 \ v;

% The rank the bound is stated in terms of, computed rather than assumed.
rank_dC = rank(full(C9 - C1));
rank_dK = 2 * rank_dC;

% T1  the update really is rank 2*rank(dC) in the KKT shape -- the premise.
[np, nf] = chk(np, nf, 'T1  dK has rank 2*rank(dC)', ...
    rank(full(K9 - K1)) == rank_dK && rank_dC == nC);

% T2  THE CLAIM: full GMRES on I + rank-r terminates in <= r+1 iterations.
[~, fl2, ~, it2] = gmres(K9, b, [], TOL, MAXIT, Minv);
its2 = it2(end);
[np, nf] = chk(np, nf, ...
    sprintf('T2  GMRES(K_1^-1 K_9) converges in <= %d its (got %d)', rank_dK + 1, its2), ...
    fl2 == 0 && its2 <= rank_dK + 1);

% T3  zero update -> the operator IS the identity -> exactly one iteration.
%     This is the disk_static invariant, and the cheapest wiring check there is.
[~, fl3, ~, it3] = gmres(K1, b, [], TOL, MAXIT, Minv);
[np, nf] = chk(np, nf, 'T3  zero update converges in exactly 1 iteration', ...
    fl3 == 0 && it3(end) == 1);

%% ------------------------------------------------------ engine contract ----
% solve_stokes_immersed assigns the 4th output into ONE element of a per-step
% array, but MATLAB's gmres returns iter as a 1x2 [outer inner].  A registry
% entry that forwards it verbatim throws at run time, and no mathematical
% assertion above would notice.  Exercise the registered closure itself.
solvers = define_solver_list(struct('GMRES_MAXIT', MAXIT));
keys    = cellfun(@(s) s.key, solvers, 'UniformOutput', false);
ig      = find(strcmp(keys, 'gmres_exact_inv_frozen'), 1);

ok = ~isempty(ig);
if ok
    pc = struct('nU', nU, 'nP', nP, 'nC', nC, 'K', K1, 'step', 1);
    pc.cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    x_ref = K1 \ b;
    [xg, flg, ~, itg] = solvers{ig}.solve(K1, b, TOL, n, pc);
    ok = isscalar(itg) && isnumeric(itg) && flg == 0 && ...
         norm(xg - x_ref) / norm(x_ref) < 1e-8;
end
[np, nf] = chk(np, nf, 'T4  registry entry returns a SCALAR iteration count', ok);

% T5  the frozen factor is built once and reused: step 2 with a DIFFERENT K must
%     not refactorize (EXACT_PREC_REFRESH = Inf), and must still solve K_9.
ok = ~isempty(ig);
if ok
    pc.step = 2;
    x_ref9 = K9 \ b;
    [xg9, flg9, ~, itg9] = solvers{ig}.solve(K9, b, TOL, n, pc);
    ok = flg9 == 0 && isscalar(itg9) && itg9 <= rank_dK + 1 && ...
         norm(xg9 - x_ref9) / norm(x_ref9) < 1e-6 && isKey(pc.cache, 'gmres_frozen_dec');
    if ok
        e  = pc.cache('gmres_frozen_dec');
        ok = e.step == 1;              % built at step 1, never rebuilt
    end
end
[np, nf] = chk(np, nf, 'T5  step-1 factor stays frozen and still meets the bound', ok);

%% -------------------------------------------------- real KKT pair (opt) ----
f1 = fullfile(root, 'stokes_kkt_example_h0p1_step01.mat');
f9 = fullfile(root, 'stokes_kkt_example_h0p1_step09.mat');
if exist(f1, 'file') && exist(f9, 'file')
    S1 = load(f1);  S9 = load(f9);
    nCr = S9.meta.nC;
    decR = decomposition((S1.A + S1.A')/2, 'ldl');
    [xr, flr, ~, itr] = gmres(S9.A, S9.b, [], 1e-10, MAXIT, @(v) decR \ v);
    itsr = itr(end);

    % T6  the claim on the real operator, at the bound the benchmark predicts.
    [np, nf] = chk(np, nf, ...
        sprintf('T6  real KKT: GMRES <= 2*nC+1 = %d its (got %d)', 2*nCr + 1, itsr), ...
        flr == 0 && itsr <= 2 * nCr + 1);

    % T7  ... and it is a genuine solve, not an early stopping-test hit.
    [np, nf] = chk(np, nf, 'T7  real KKT: solution matches the direct solve', ...
        norm(xr - S9.x_ref) / norm(S9.x_ref) < 1e-8);
else
    fprintf('  skip T6-T7 (run extract_kkt_examples to generate the fixtures)\n');
end

%% ------------------------------------------------------------ summary ----
fprintf('\n%d passed, %d FAILED\n', np, nf);
if nf > 0, error('test_gmres_lowrank:failures', '%d checks failed', nf); end

%==========================================================================
function [np, nf] = chk(np, nf, name, cond)
    if cond
        np = np + 1;  fprintf('  ok   %s\n', name);
    else
        nf = nf + 1;  fprintf('  FAIL %s\n', name);
    end
end
