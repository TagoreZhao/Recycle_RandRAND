%TEST_MAKE_AMG_PREC_ABLATE  Unit tests for the local ablatable AMG factory.
%
%   Verifies make_amg_prec_ablate against src.precond.make_amg_preconditioner
%   on a small 2D 5-point Laplacian:
%     T1  exact-setting equivalence with the +src factory
%     T2  symmetry of M when preSmooth == postSmooth (and nonsymmetry when not)
%     T3  inexact coarse solves converge to the exact one (jacobi nu up, pcg tol dn)
%     T4  block apply == column-wise apply
%     T5  pure coarse-grid correction (0,0) has rank <= coarseN
%     T6  info diagnostics are sane
%     T7  sjlt projector: size control, symmetry, block apply, empty-col patch
%     T8  sjlt exact-inverse limit: nc = n, pure CGC  =>  M = A^{-1}
%
%   Usage:  cd amg_subspace; test_make_amg_prec_ablate

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(thisFileDir);
addpath(repoRoot);       % src.* packages
addpath(thisFileDir);    % local factory

rng(42);

% --- Small SPD test problem: 2D 5-point Laplacian + ichol nofill -----------
ng = 30;
A  = gallery('poisson', ng);          % n = 900, SPD
n  = size(A, 1);
L  = ichol(A, struct('type', 'nofill'));
Lt = L';

% Options shared by both factories (exact coarse solve unless overridden).
base = {'maxLevels', 3, 'minCoarseSize', 100, 'theta', 0.05, ...
        'omegaSmooth', 2/3, 'omegaInterp', 0, 'maxAggSize', 16};

%% T1: exact-setting equivalence with src.precond.make_amg_preconditioner
prepost = [1 1; 2 2; 1 0; 0 1];
for cs = {'backslash', 'chol'}
    for withIchol = [false, true]
        if withIchol
            extra = {'fineSmootherL', L, 'fineSmootherLt', Lt};
        else
            extra = {};
        end
        for ip = 1:size(prepost, 1)
            args = [base, {'preSmooth', prepost(ip,1), ...
                           'postSmooth', prepost(ip,2), ...
                           'coarseSolve', cs{1}}, extra];
            Mloc = make_amg_prec_ablate(A, args{:});
            Msrc = src.precond.make_amg_preconditioner(A, args{:});
            r  = randn(n, 1);
            d  = norm(Mloc(r) - Msrc(r)) / norm(Msrc(r));
            assert(d <= 1e-12, ...
                   'T1: mismatch vs src factory (%s, ichol=%d, pre=%d post=%d): %.3e', ...
                   cs{1}, withIchol, prepost(ip,1), prepost(ip,2), d);
        end
    end
end
fprintf('PASS T1: exact settings match src.precond.make_amg_preconditioner\n');

%% T2: symmetry iff preSmooth == postSmooth
r1 = randn(n, 1);  r2 = randn(n, 1);
for pp = [1 2]
    args = [base, {'preSmooth', pp, 'postSmooth', pp, 'coarseSolve', 'chol', ...
                   'fineSmootherL', L, 'fineSmootherLt', Lt}];
    M = make_amg_prec_ablate(A, args{:});
    s = abs(r2' * M(r1) - r1' * M(r2)) / (norm(r1) * norm(r2));
    assert(s <= 1e-10, 'T2: M not symmetric for pre=post=%d: %.3e', pp, s);
end
args = [base, {'preSmooth', 1, 'postSmooth', 0, 'coarseSolve', 'chol', ...
               'fineSmootherL', L, 'fineSmootherLt', Lt}];
M = make_amg_prec_ablate(A, args{:});
s = abs(r2' * M(r1) - r1' * M(r2)) / (norm(r1) * norm(r2));
assert(s > 1e-8, 'T2: (1,0) V-cycle unexpectedly symmetric: %.3e', s);
fprintf('PASS T2: symmetric iff preSmooth == postSmooth (asym check %.2e)\n', s);

%% T3: inexact coarse solves approach the exact one
argsExact = [base, {'preSmooth', 1, 'postSmooth', 1, 'coarseSolve', 'chol', ...
                    'fineSmootherL', L, 'fineSmootherLt', Lt}];
Mexact = make_amg_prec_ablate(A, argsExact{:});
r  = randn(n, 1);
xe = Mexact(r);

nus  = [2 8 32 128];
errs = zeros(size(nus));
for iv = 1:numel(nus)
    argsJ = [base, {'preSmooth', 1, 'postSmooth', 1, ...
                    'coarseSolve', 'jacobi', 'coarseJacobiSweeps', nus(iv), ...
                    'fineSmootherL', L, 'fineSmootherLt', Lt}];
    Mj = make_amg_prec_ablate(A, argsJ{:});
    errs(iv) = norm(Mj(r) - xe) / norm(xe);
end
assert(all(diff(errs) <= 0), 'T3: jacobi coarse error not nonincreasing: %s', ...
       mat2str(errs, 3));
assert(errs(end) < errs(1), 'T3: jacobi coarse error did not decrease');

argsP = [base, {'preSmooth', 1, 'postSmooth', 1, 'coarseSolve', 'pcg', ...
                'coarsePcgTol', 1e-10, 'coarsePcgMaxit', 500, ...
                'fineSmootherL', L, 'fineSmootherLt', Lt}];
Mp = make_amg_prec_ablate(A, argsP{:});
dp = norm(Mp(r) - xe) / norm(xe);
assert(dp <= 1e-6, 'T3: tight-tol pcg coarse differs from exact: %.3e', dp);
fprintf('PASS T3: inexact coarse solves converge (jacobi %s, pcg %.2e)\n', ...
        mat2str(errs, 3), dp);

%% T4: block apply equals column-wise apply
mB = 7;
R  = randn(n, mB);
for cs = {'chol', 'jacobi'}
    args = [base, {'preSmooth', 1, 'postSmooth', 1, 'coarseSolve', cs{1}, ...
                   'coarseJacobiSweeps', 3, ...
                   'fineSmootherL', L, 'fineSmootherLt', Lt}];
    M = make_amg_prec_ablate(A, args{:});
    Xblk = M(R);
    Xcol = zeros(n, mB);
    for j = 1:mB
        Xcol(:, j) = M(R(:, j));
    end
    d = norm(Xblk - Xcol, 'fro') / norm(Xcol, 'fro');
    assert(d <= 1e-12, 'T4: block/column mismatch (%s): %.3e', cs{1}, d);
end
fprintf('PASS T4: block apply matches column-wise apply\n');

%% T5: pure coarse-grid correction has rank <= coarseN
args = [base, {'preSmooth', 0, 'postSmooth', 0, 'coarseSolve', 'chol', ...
               'maxLevels', 2}];
[M, info] = make_amg_prec_ablate(A, args{:});
mWide = info.coarseN + 20;
X = M(randn(n, mWide));
rk = rank(X);
assert(rk <= info.coarseN, ...
       'T5: rank(M*G) = %d exceeds coarseN = %d', rk, info.coarseN);
fprintf('PASS T5: pure CGC rank %d <= coarseN %d\n', rk, info.coarseN);

%% T6: info diagnostics
args = [base, {'preSmooth', 1, 'postSmooth', 1, 'coarseSolve', 'chol', ...
               'fineSmootherL', L, 'fineSmootherLt', Lt}];
[~, info] = make_amg_prec_ablate(A, args{:});
assert(info.setupTime > 0, 'T6: setupTime not positive');
assert(info.workPerApply > 0 && info.workUnits > 0, 'T6: work proxy not positive');
lvN = [info.levels.n];
assert(all(diff(lvN) < 0), 'T6: level sizes not strictly decreasing: %s', ...
       mat2str(lvN));
assert(info.coarseN == lvN(end) && info.nLevels == numel(lvN), ...
       'T6: coarseN/nLevels inconsistent');
assert(strcmp(info.coarseType, 'chol'), 'T6: unexpected coarseType %s', ...
       info.coarseType);
fprintf('PASS T6: info diagnostics sane (levels %s, WU/apply %.2f)\n', ...
        mat2str(lvN), info.workUnits);

%% T7: sjlt projector basics
rng(7);
ncS = 150;
args = [base, {'preSmooth', 1, 'postSmooth', 1, 'coarseSolve', 'chol', ...
               'projector', 'sjlt', 'sjltNc', ncS, 'sjltNnzPerCol', 4, ...
               'maxLevels', 2, 'fineSmootherL', L, 'fineSmootherLt', Lt}];
[M, info] = make_amg_prec_ablate(A, args{:});
assert(info.coarseN == ncS, 'T7: coarseN %d ~= sjltNc %d', info.coarseN, ncS);
assert(strcmp(info.projector, 'sjlt'), 'T7: projector not echoed');
s = abs(r2' * M(r1) - r1' * M(r2)) / (norm(r1) * norm(r2));
assert(s <= 1e-10, 'T7: sjlt V-cycle not symmetric for pre=post=1: %.3e', s);
R = randn(n, 5);
Xblk = M(R);
for j = 1:5
    assert(norm(Xblk(:, j) - M(R(:, j))) <= 1e-12 * norm(Xblk(:, j)), ...
           'T7: sjlt block/column mismatch');
end
% Empty-column patch: s = 1, nc large relative to n forces empty columns
% w.h.p.; the coarse operator must still be SPD (chol succeeds => coarseType
% stays 'chol').
rng(8);
argsE = [base, {'preSmooth', 1, 'postSmooth', 1, 'coarseSolve', 'chol', ...
                'projector', 'sjlt', 'sjltNc', 600, 'sjltNnzPerCol', 1, ...
                'maxLevels', 2}];
[Me, infoE] = make_amg_prec_ablate(A, argsE{:});
assert(strcmp(infoE.coarseType, 'chol'), ...
       'T7: coarse chol failed for sjlt nc=600, s=1 (singular Ac?)');
xe = Me(randn(n, 1));
assert(all(isfinite(xe)), 'T7: non-finite apply for patched sjlt');
fprintf('PASS T7: sjlt projector (nc control, symmetry, block, empty-col patch)\n');

%% T8: sjlt exact-inverse limit (nc = n, pure coarse-grid correction)
% Square full-rank Omega makes CGC = Omega*(Omega'*A*Omega)^{-1}*Omega' equal
% A^{-1} exactly -- the anchor of the coarse-size sweep.  s = 8 here: the
% error is amplified by cond(Omega)^2, and a SQUARE sjlt draw at s = 4 is
% numerically singular (cond ~1e18 measured), while s = 8 gives cond ~1e3.
rng(9);
args = [base, {'preSmooth', 0, 'postSmooth', 0, 'coarseSolve', 'chol', ...
               'projector', 'sjlt', 'sjltNc', n, 'sjltNnzPerCol', 8, ...
               'maxLevels', 2, 'minCoarseSize', 1}];
[M, info] = make_amg_prec_ablate(A, args{:});
r = randn(n, 1);
d = norm(M(r) - A \ r) / norm(A \ r);
assert(d <= 1e-6, 'T8: nc=n pure CGC differs from A^{-1}: %.3e', d);
fprintf('PASS T8: sjlt nc=n pure CGC equals A^{-1} (rel err %.2e, coarse=%s)\n', ...
        d, info.coarseType);

%% T9: single-pass SA coarse size is a live knob via maxAggSize (section E)
% The E_sa_coarse_size sweep varies maxAggSize to move the SA coarse size.
% Each maxAggSize must yield a valid 2-level SPD hierarchy the sweep can run,
% and coarseN must actually RESPOND to the knob (>= 2 distinct sizes over the
% sweep).  The magnitude/direction of the response is mesh-dependent -- clean
% and monotone on the sphere FEM operator, weak on this structured 5-point
% grid where aggregates fragment -- so only the responds-at-all invariant is
% asserted here.
masList = [2 3 4 6 16];
ncByMas = zeros(size(masList));
for iv = 1:numel(masList)
    args = {'maxLevels', 2, 'minCoarseSize', 1, 'theta', 0.05, ...
            'omegaSmooth', 2/3, 'omegaInterp', 0, 'maxAggSize', masList(iv), ...
            'preSmooth', 1, 'postSmooth', 1, 'coarseSolve', 'chol', ...
            'fineSmootherL', L, 'fineSmootherLt', Lt};
    [~, infoE] = make_amg_prec_ablate(A, args{:});
    assert(infoE.nLevels == 2, 'T9: expected 2 levels, got %d', infoE.nLevels);
    assert(strcmp(infoE.coarseType, 'chol'), ...
           'T9: coarse chol failed for maxAggSize=%d', masList(iv));
    assert(infoE.coarseN >= 1 && infoE.coarseN < n, ...
           'T9: coarseN=%d out of range for maxAggSize=%d', ...
           infoE.coarseN, masList(iv));
    ncByMas(iv) = infoE.coarseN;
end
assert(numel(unique(ncByMas)) >= 2, ...
       'T9: maxAggSize is an inert knob, coarseN constant: %s', ...
       mat2str(ncByMas));
fprintf('PASS T9: maxAggSize is a live SA coarseN knob (mas %s -> nc %s)\n', ...
        mat2str(masList), mat2str(ncByMas));

fprintf('\nAll make_amg_prec_ablate tests passed.\n');
