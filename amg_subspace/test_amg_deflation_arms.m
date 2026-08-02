% TEST_AMG_DEFLATION_ARMS  Unit tests for the deflation-vs-preconditioning study.
%
% Covers the four new pieces the study rests on:
%   precond_spectrum   -- Ritz spectrum of a preconditioned operator
%   amg_sketch_basis   -- Gaussian sketch of an AMG V-cycle into a basis
%   amg_sketch_tau     -- deflation shift estimated from that basis alone
%   amg_deflation_arms -- the preconditioner recipes compared in the driver
% plus amg_bench_snapshot, the shared test-problem builder.
%
% Script style with assert (matching test_make_amg_prec_ablate.m), not a
% matlab.unittest class.
%
% Usage:
%   cd amg_subspace
%   test_amg_deflation_arms

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(thisFileDir);
addpath(repoRoot);
addpath(fullfile(repoRoot, 'subspace_capture'));
addpath(thisFileDir);

fprintf('\n=== test_amg_deflation_arms ===\n');

rng(42);
A  = gallery('poisson', 30);          % n = 900, SPD, kappa ~ 3.6e2
n  = size(A, 1);
R  = chol(A);                         % upper, A = R'*R
Afun = @(X) A * X;
L  = ichol(A, struct('type', 'nofill'));
Lt = L';

lamA     = sort(eig(full(A)), 'ascend');
lam_maxA = lamA(end);
lam_minA = lamA(1);

% Lanczos budget: 500 of 900 steps with full reorthogonalization.  Generous on
% purpose -- these tests are about correctness of the routine, not about how
% few steps it can get away with.
specOpts = struct('m', 500, 'n_tail', 50);
RTOL     = 1e-2;                      % 1% agreement for all spectral checks

%% ------------------------------------------------------------------------
%% T1: precond_spectrum against a dense eigendecomposition
%% ------------------------------------------------------------------------
Vex30  = [];                                   % filled below, reused by T3
[Vall, Dall] = eig(full(A));
[dsort, idx] = sort(real(diag(Dall)), 'ascend');
Vall  = real(Vall(:, idx));
Vex30 = Vall(:, 1:30);
tau31 = dsort(31);

precs = { ...
    [],                                                  'identity'; ...
    @(X) Lt \ (L \ X),                                   'ichol'; ...
    src.precond.deflation_P_apply(orth(randn(n, 20)), Afun, dsort(21), 'handle'), 'deflation' ...
};

for ip = 1:size(precs, 1)
    Mfun = precs{ip, 1};
    name = precs{ip, 2};

    out = precond_spectrum(Mfun, R, n, specOpts);
    assert(out.ok, 'T1 (%s): precond_spectrum failed: %s', name, out.err);

    % Dense truth: H = R*M*R' is symmetric and similar to M*A.
    if isempty(Mfun)
        Hd = full(R * R');
    else
        Hd = full(R * Mfun(full(R')));
    end
    Hd   = (Hd + Hd') / 2;
    lamd = sort(real(eig(Hd)), 'ascend');

    rel_min = abs(out.lam_min - lamd(1))   / lamd(1);
    rel_max = abs(out.lam_max - lamd(end)) / lamd(end);
    rel_kap = abs(out.kappa - lamd(end)/lamd(1)) / (lamd(end)/lamd(1));
    assert(rel_min < RTOL, 'T1 (%s): lam_min off by %.3g', name, rel_min);
    assert(rel_max < RTOL, 'T1 (%s): lam_max off by %.3g', name, rel_max);
    assert(rel_kap < RTOL, 'T1 (%s): kappa off by %.3g',   name, rel_kap);
    assert(numel(out.ritz_low) == 50 && numel(out.ritz_high) == 50, ...
           'T1 (%s): ritz tails have the wrong length', name);
    assert(issorted(out.ritz), 'T1 (%s): ritz values are not ascending', name);

    fprintf('PASS T1 (%-9s): lam_min %.4e  lam_max %.4e  kappa %.4e (m_used=%d)\n', ...
            name, out.lam_min, out.lam_max, out.kappa, out.m_used);
end

%% ------------------------------------------------------------------------
%% T2: precond_spectrum against deflated_cond_two_level (exact lam_min)
%% ------------------------------------------------------------------------
Vr   = orth(randn(n, 40));
tauR = dsort(41);
Pr   = src.precond.deflation_P_apply(Vr, Afun, tauR, 'handle');

spec = precond_spectrum(Pr, R, n, specOpts);
assert(spec.ok, 'T2: precond_spectrum failed: %s', spec.err);

dA   = decomposition(A, 'chol');
cond2 = deflated_cond_two_level(Vr, Afun, @(X) dA \ X, tauR, n, ...
                                struct('W_is_orth', true));
assert(cond2.ok, 'T2: deflated_cond_two_level failed: %s', cond2.err);

rel = abs(spec.kappa - cond2.kappa) / cond2.kappa;
assert(rel < RTOL, 'T2: kappa mismatch %.4e vs %.4e (rel %.3g)', ...
       spec.kappa, cond2.kappa, rel);
fprintf('PASS T2: kappa %.4e (Lanczos) vs %.4e (exact) -- rel %.2e\n', ...
        spec.kappa, cond2.kappa, rel);

%% ------------------------------------------------------------------------
%% T3: analytic anchor -- exact deflation fixes the bottom, not the top
%% ------------------------------------------------------------------------
Pex  = src.precond.deflation_P_apply(Vex30, Afun, tau31, 'handle');
sex  = precond_spectrum(Pex, R, n, specOpts);
assert(sex.ok, 'T3: precond_spectrum failed: %s', sex.err);

rel_lmin = abs(sex.lam_min - tau31)    / tau31;
rel_lmax = abs(sex.lam_max - lam_maxA) / lam_maxA;
rel_kap  = abs(sex.kappa - lam_maxA/tau31) / (lam_maxA/tau31);
assert(rel_lmin < RTOL, 'T3: lam_min %.4e should equal tau %.4e', sex.lam_min, tau31);
assert(rel_lmax < RTOL, 'T3: lam_max %.4e should equal lam_max(A) %.4e -- ' , ...
       sex.lam_max, lam_maxA);
assert(rel_kap  < RTOL, 'T3: kappa off by %.3g', rel_kap);
fprintf(['PASS T3: exact deflation lifts lam_min %.4e -> %.4e (= tau) and ', ...
         'leaves lam_max at %.4e (kappa %.4e)\n'], ...
        lam_minA, sex.lam_min, sex.lam_max, sex.kappa);

%% ------------------------------------------------------------------------
%% T4: sketch pipeline -- reproducible, orthonormal, sketch-only shift
%% ------------------------------------------------------------------------
[Mamg, ainfo] = make_amg_prec_ablate(A, 'maxLevels', 2, 'minCoarseSize', 100, ...
                                     'preSmooth', 1, 'postSmooth', 1, ...
                                     'coarseSolve', 'chol');
m_sk = 40;

[V1, i1] = amg_sketch_basis(Mamg, n, m_sk, 3, 7);
[V2, i2] = amg_sketch_basis(Mamg, n, m_sk, 3, 7);
assert(isequal(V1, V2), 'T4: sketch is not reproducible at a fixed seed');
assert(i1.r_defl == size(V1, 2) && i1.r_defl <= m_sk, 'T4: r_defl bookkeeping wrong');
assert(norm(V1'*V1 - eye(size(V1,2)), 'fro') < 1e-10, 'T4: V is not orthonormal');

% Incremental continuation must reach the same SUBSPACE as a from-scratch run
% at the same q.  Compare spans, not bases: each orth() is free to return any
% rotation of the same span, so Vb'*V1 is an arbitrary orthogonal matrix.
[Va, ia] = amg_sketch_basis(Mamg, n, m_sk, 1, 7);
[Vb, ~]  = amg_sketch_basis(Mamg, n, m_sk, 3, 7, ia);
assert(norm(V1 - Vb*(Vb'*V1), 'fro') < 1e-6, ...
       'T4: incremental continuation does not reproduce the direct run');
assert(size(Va, 2) == m_sk, 'T4: q=1 sketch lost rank unexpectedly');

% A different preconditioner on the SAME seed must give a different subspace:
% the sketch really is carrying the preconditioner's information.
[Mweak, ~] = make_amg_prec_ablate(A, 'maxLevels', 2, 'minCoarseSize', 100, ...
                                  'preSmooth', 0, 'postSmooth', 0, ...
                                  'coarseSolve', 'chol');
Vw = amg_sketch_basis(Mweak, n, m_sk, 3, 7);
capt = subspace_capture_directed(V1, Vw, [], struct('true_is_orth', true, ...
                                                    'comp_is_orth', true));
assert(capt.eigspace_err_2 > 1e-6, ...
       'T4: two different V-cycles produced the same subspace');

% Shift comes from the sketch alone and equals the top Ritz value.
[tau_s, tinfo] = amg_sketch_tau(V1, Afun);
Eref = V1' * (A * V1);  Eref = (Eref + Eref')/2;
assert(tinfo.ok && tau_s > 0, 'T4: tau estimate failed: %s', tinfo.err);
assert(abs(tau_s - max(eig(Eref))) < 1e-10 * tau_s, ...
       'T4: tau is not lam_max(V''AV)');
fprintf(['PASS T4: sketch reproducible, orthonormal (r=%d/%d), incremental q ', ...
         'consistent, tau=%.4e from sketch only\n'], i1.r_defl, m_sk, tau_s);

%% ------------------------------------------------------------------------
%% T5: the deflation projector is symmetric and positive definite
%% ------------------------------------------------------------------------
Pt = src.precond.deflation_P_apply(V1, Afun, tau_s, 'handle');
maxAsym = 0;  minQuad = Inf;
for j = 1:20
    y1 = randn(n, 1);  y2 = randn(n, 1);
    maxAsym = max(maxAsym, abs(y2'*Pt(y1) - y1'*Pt(y2)) / (norm(y1)*norm(y2)));
    minQuad = min(minQuad, (y1'*Pt(y1)) / (y1'*y1));
end
assert(maxAsym < 1e-10, 'T5: P is not symmetric (asym %.3g)', maxAsym);
assert(minQuad > 0,     'T5: P is not positive definite (min quad %.3g)', minQuad);
fprintf('PASS T5: P symmetric (asym %.2e) and positive definite (min RQ %.3e)\n', ...
        maxAsym, minQuad);

%% ------------------------------------------------------------------------
%% T6: every arm builds a handle pcg AND precond_spectrum both accept
%% ------------------------------------------------------------------------
arms = amg_deflation_arms();
ctx  = struct('A', A, 'Afun', Afun, 'L', L, 'Lt', Lt, 'Mamg', Mamg, ...
              'n', n, 'M_is_sym', true);
b    = A * ones(n, 1);
armOpts = struct('m', 200, 'n_tail', 20);

for ia_ = 1:numel(arms)
    a = arms(ia_);
    if a.needs_V
        s = a.build(ctx, V1, tau_s);
    else
        s = a.build(ctx, [], []);
    end
    assert(s.valid, 'T6 (%s): builder reported invalid', a.id);
    assert(isempty(s.prec) || isa(s.prec, 'function_handle'), ...
           'T6 (%s): prec is neither [] nor a handle', a.id);

    if isempty(s.prec)
        [~, fl] = pcg(A, b, 1e-8, 500);
    else
        [~, fl] = pcg(A, b, 1e-8, 500, s.prec);
    end
    assert(fl == 0, 'T6 (%s): pcg did not converge (flag %d)', a.id, fl);

    sp = precond_spectrum(s.prec, R, n, armOpts);
    assert(sp.ok, 'T6 (%s): precond_spectrum failed: %s', a.id, sp.err);
    fprintf('PASS T6 (%-15s): pcg flag 0, kappa %.4e\n', a.id, sp.kappa);
end

%% ------------------------------------------------------------------------
%% T7: guards -- nonsymmetric M, and the rank-limited pure-CGC config
%% ------------------------------------------------------------------------
% Nonsymmetric V-cycle (preSmooth ~= postSmooth): the arms that apply M must be
% gated out, but the sketch-based deflation arm is unaffected -- it only ever
% uses M to GENERATE a basis, never to precondition.
[Mns, ~] = make_amg_prec_ablate(A, 'maxLevels', 2, 'minCoarseSize', 100, ...
                                'preSmooth', 1, 'postSmooth', 0, ...
                                'coarseSolve', 'chol');
X  = randn(n, 3);
asym_ns = norm(X' * Mns(X) - (Mns(X))' * X, 'fro') / norm(X' * Mns(X), 'fro');
assert(asym_ns > 1e-8, 'T7: pre~=post was expected to give a nonsymmetric M');

gated = {arms([arms.needs_sym]).id};
assert(ismember('amg_direct', gated) && ismember('ctau_amg', gated), ...
       'T7: amg_direct and ctau_amg must be flagged needs_sym');
assert(~arms(strcmp({arms.id}, 'defl_amg')).needs_sym, ...
       'T7: defl_amg must NOT require a symmetric M');

Vns  = amg_sketch_basis(Mns, n, m_sk, 2, 7);
tns  = amg_sketch_tau(Vns, Afun);
sns  = arms(strcmp({arms.id}, 'defl_amg')).build(ctx, Vns, tns);
[~, fl_ns] = pcg(A, b, 1e-8, 500, sns.prec);
assert(fl_ns == 0, 'T7: defl_amg on a nonsymmetric-M sketch failed to converge');

% Pure coarse-grid correction: rank(M) = coarseN, so a wider sketch cannot
% exceed that rank.  The failure mode must surface as a short basis (or a
% caught chol failure downstream), never a crash.
[Mcgc, cinfo] = make_amg_prec_ablate(A, 'maxLevels', 2, 'minCoarseSize', 100, ...
                                     'preSmooth', 0, 'postSmooth', 0, ...
                                     'coarseSolve', 'chol');
wide = min(cinfo.coarseN + 20, n);
[Vcgc, icgc] = amg_sketch_basis(Mcgc, n, wide, 2, 7);
assert(icgc.r_defl <= cinfo.coarseN, ...
       'T7: pure CGC sketch has rank %d > coarseN %d', icgc.r_defl, cinfo.coarseN);
assert(icgc.rank_limited, 'T7: rank_limited flag not set for the CGC sketch');

% ...and the same singular M must be REFUSED as a preconditioner rather than
% reported with a kappa of ~1/eps, which is a number but a meaningless one.
scgc = precond_spectrum(Mcgc, R, n, struct('m', 200, 'n_tail', 10));
assert(~scgc.ok, 'T7: singular pure-CGC M was accepted as a preconditioner');
assert(contains(scgc.err, 'singular'), ...
       'T7: expected a singularity diagnosis, got "%s"', scgc.err);
fprintf(['PASS T7: nonsym M gated (defl_amg still converges); pure CGC sketch ', ...
         'rank-limited to %d (coarseN %d, asked %d)\n'], ...
        icgc.r_defl, cinfo.coarseN, wide);

%% ------------------------------------------------------------------------
%% T8: amg_bench_snapshot -- reproducible, SPD, correct diffusivity contrast
%% ------------------------------------------------------------------------
h0t = 0.3;  contrast = 60;  dt = 1;  Tmax = 100;  t_snap = 0;
[As, Ls, mshs, bs, metas] = amg_bench_snapshot(h0t, contrast, t_snap, dt, ...
                                               Tmax, 'pdetoolbox', 20, 1, 3);
[As2, Ls2, ~, bs2] = amg_bench_snapshot(h0t, contrast, t_snap, dt, ...
                                        Tmax, 'pdetoolbox', 20, 1, 3);
assert(isequal(As, As2) && isequal(Ls, Ls2) && isequal(bs, bs2), ...
       'T8: amg_bench_snapshot is not reproducible');
assert(issymmetric(As), 'T8: A is not symmetric');
[~, cflag] = chol(As);
assert(cflag == 0, 'T8: A is not SPD');
assert(norm(bs) > 0, 'T8: RHS is zero');
assert(metas.n == mshs.numIN && metas.nnzA == nnz(As), 'T8: meta disagrees with A');

% Guard the copied kappa factory AND the copied assembly: re-derive both here,
% independently of amg_bench_snapshot's internals, and require a bit-identical
% A.  This is what protects the committed eigenvector caches, which are keyed
% only by (h0, k) and would silently go stale if the operator ever drifted.
kfield = kappa_probe(mshs, contrast, Tmax, t_snap);
kmin   = 1/sqrt(contrast);  kmax = sqrt(contrast);
assert(min(kfield) >= kmin - 1e-12 && max(kfield) <= kmax + 1e-12, ...
       'T8: kappa field leaves [%g, %g]', kmin, kmax);
assert(max(kfield)/min(kfield) > 1, 'T8: kappa field is constant');

Vscale = mshs.Vunit .* repelem(kfield, 9);
Vii    = Vscale(mshs.idxII);
Kii    = sparse(mshs.I_II, mshs.J_II, Vii, mshs.numIN, mshs.numIN);
Aref   = mshs.D_II + dt * Kii;
Aref   = 0.5 * (Aref + Aref.');
assert(isequal(As, Aref), ...
       'T8: A differs from an independent re-assembly (max diff %.3g)', ...
       full(max(max(abs(As - Aref)))));
fprintf(['PASS T8: snapshot reproducible, SPD (n=%d, nnz=%d), A matches an ', ...
         'independent re-assembly; kappa in [%.4g, %.4g]\n'], ...
        metas.n, metas.nnzA, min(kfield), max(kfield));

fprintf('\nAll test_amg_deflation_arms tests passed.\n\n');

%% =========================================================================
%% Local helpers
%% =========================================================================
function kf = kappa_probe(msh, contrast, Tmax, tcur)
%KAPPA_PROBE  Re-evaluate the banding diffusivity independently of the
%   snapshot builder, so a mis-copied kappa factory cannot pass T8 silently.
    kmin  = 1/sqrt(contrast);
    kmax  = sqrt(contrast);
    bw    = 0.25;
    freqs = [sqrt(2), sqrt(3), sqrt(5)];
    z     = msh.cent(:, 3);
    theta = acos(max(min(z, 1), -1));
    bump  = zeros(size(theta));
    base  = [pi/4, pi/2, 3*pi/4];
    for k = 1:3
        c    = base(k) + 0.5 * sin(2*pi*freqs(k)*tcur/Tmax);
        c    = max(0.1, min(pi - 0.1, c));
        bump = bump + 0.5 + 0.5 * tanh((bw - abs(theta - c)) / 0.1);
    end
    bump = min(bump/3, 1);
    kf   = kmin + (kmax - kmin) * bump;
end
