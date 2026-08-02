% TEST_BUILD_STOKES_SEQUENCE  Unit tests for the KKT time-sequence builder.
%
% The load-bearing claim of the whole study is T1: the moving immersed-rotor KKT
% family is a rank-<= 2*nC update of ONE fixed matrix.  If that fails, the
% "missing component is 2*nC dimensional" argument and the frozen-factorization
% cost model both collapse, so it is checked at every step of every case.
%
% T7 cross-checks the rebuilt sequence against the stored benchmark CSVs.  A
% silent divergence there (different mesh, different Tmax, different motion)
% would invalidate every downstream diagnosis while still "passing" T1-T6.
%
% Run:  test_build_stokes_sequence

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
p = add_recycle_paths();
rng(1);

H_TINY = 0.1;      % ~n=1500, seconds per case
NS     = 6;
npass  = 0;

fprintf('=== test_build_stokes_sequence ===\n');

%% ---- T1/T2/T3/T6: the low-rank identity, per case ----------------------
for cn = {'bar_rotating', 'disk_translating', 'disk_static'}
    cname = cn{1};
    S = build_stokes_sequence(struct('case_name', cname, 'h0', H_TINY, ...
                                     'nsteps', NS, 'use_cache', false, ...
                                     'verify', true, 'quiet', true));

    % T1: K_n = K0 + Cblk_n Sel' + Sel Cblk_n', reassembled independently.
    worst = 0;
    for n = 1:S.nsteps
        Kn = seq_K(S, n);
        % rebuild the reference form directly from the stored raw coupling
        Cu = S.Ccpl{n}';  Cu(S.veldofs, :) = 0;
        Ref = S.K0 + [Cu; sparse(S.nP + S.nC, S.nC)] * S.Sel' + ...
                     S.Sel * [Cu; sparse(S.nP + S.nC, S.nC)]';
        worst = max(worst, norm(Kn - Ref, 'fro') / norm(Kn, 'fro'));
    end
    assert(worst < 1e-12, 'T1 %s: low-rank identity rel err %.3e', cname, worst);

    % T2: rank of the step-to-step difference is at most 2*nC.
    dK = full(seq_K(S, S.nsteps) - seq_K(S, 1));
    r  = rank(dK, 1e-10 * max(norm(dK, 'fro'), eps));
    assert(r <= 2 * S.nC, 'T2 %s: rank(dK)=%d exceeds 2*nC=%d', cname, r, 2*S.nC);

    % T3: exact symmetry (MINRES requires it).
    Kn = seq_K(S, S.nsteps);
    sr = norm(Kn - Kn', 'fro') / norm(Kn, 'fro');
    assert(sr < 1e-14, 'T3 %s: symmetry residual %.3e', cname, sr);

    % T6: the stored ground truth really solves its system.
    worst_res = 0;
    for n = 1:S.nsteps
        rr = norm(seq_K(S, n) * S.xref{n} - S.b{n}) / norm(S.b{n});
        worst_res = max(worst_res, rr);
    end
    assert(worst_res < 1e-10, 'T6 %s: backslash residual %.3e', cname, worst_res);

    fprintf('  PASS T1/T2/T3/T6 [%s]: n=%d nC=%d  lowrank=%.1e rank=%d<=%d  res=%.1e\n', ...
            cname, S.n, S.nC, worst, r, 2*S.nC, worst_res);
    npass = npass + 4;
end

%% ---- T4: determinism ---------------------------------------------------
o = struct('case_name', 'bar_rotating', 'h0', H_TINY, 'nsteps', 3, ...
           'use_cache', false, 'quiet', true);
S1 = build_stokes_sequence(o);
S2 = build_stokes_sequence(o);
assert(isequal(S1.K0, S2.K0) && isequal(S1.Cblk, S2.Cblk), ...
       'T4: sequence assembly is not deterministic');
assert(norm(S1.b{3} - S2.b{3}) == 0, 'T4: RHS is not deterministic');
fprintf('  PASS T4: assembly is deterministic across calls\n');
npass = npass + 1;

%% ---- T5: disk_static is the null control -------------------------------
Ss = build_stokes_sequence(struct('case_name', 'disk_static', 'h0', H_TINY, ...
                                  'nsteps', 5, 'use_cache', false, 'quiet', true));
assert(isequal(seq_K(Ss, 1), seq_K(Ss, 5)), ...
       'T5: disk_static K is not constant across steps');
[Ustat, dCstat] = seq_dCblk(Ss, 5, 1);
assert(nnz(dCstat) == 0, 'T5: disk_static dC is not exactly zero (nnz=%d)', nnz(dCstat));
assert(rank(full(Ustat), 1e-12) == Ss.nC, ...
       'T5: disk_static generator should collapse to rank nC');
assert(norm(Ss.b{5} - Ss.b{4}) > 0, ...
       'T5: disk_static RHS should still evolve (the state advances)');
fprintf('  PASS T5: disk_static K constant, dC == 0, RHS still moving\n');
npass = npass + 1;

%% ---- T7: agreement with the stored benchmark ---------------------------
% Same h0/dt/Tstep the benchmark ran with; compare against its own CSV.
csv = fullfile(p.solvesDir, '..', 'stokes_immersed_rotor', ...
               'benchmark_final_small_krylov', 'all_results.csv');
% The decisive check is the last one: reproducing the benchmark's own
% ildl_nofill iteration count exercises mesh, assembly, BCs, RHS, motion phase
% and the ILDL smoother end to end.  nC and diffF alone would not catch a
% mismatched Tmax or an off-by-one in the motion clock.
if exist(csv, 'file') == 2
    T  = readtable(csv);
    Sb = build_stokes_sequence(struct('case_name', 'bar_rotating', 'h0', 0.05, ...
                                      'nsteps', 2, 'use_cache', false, 'quiet', true));
    rows = T(strcmp(T.case_name, 'bar_rotating'), :);

    assert(rows.nC(1) == Sb.nC, ...
           'T7: nC=%d disagrees with benchmark nC=%d', Sb.nC, rows.nC(1));
    dc_ours  = Sb.coupling_change(2);
    dc_bench = rows.diffF(2);
    assert(abs(dc_ours - dc_bench) < 1e-8 * max(abs(dc_bench), 1), ...
           'T7: coupling_change(2) %.6f vs benchmark diffF %.6f', dc_ours, dc_bench);

    K1 = seq_K(Sb, 1);
    P1 = src.precond.make_ildl_precond(K1, struct('mode', 'nofill'));
    r1 = two_level_it(K1, Sb.b{1}, P1, [], struct('tol', 1e-8, 'maxit', Sb.n));
    it_bench = rows.ildl_nofill_its(1);
    assert(r1.iters == it_bench, ...
           ['T7: ildl_nofill step-1 iterations %d disagree with the benchmark ' ...
            '(%d) — the rebuilt system is not the benchmark''s system'], ...
           r1.iters, it_bench);
    % 1e-6, matching the folder convention (test_two_level_recycle.m:114): MINRES
    % converges the SPLIT residual to 1e-8, and ||b-Kx||/||b|| is amplified by
    % cond(C) on the way back — here by ~20x.
    assert(r1.true_res < 1e-6, 'T7: ILDL solve true residual %.3e', r1.true_res);

    fprintf(['  PASS T7: matches benchmark (n=%d, nC=%d, dC_2=%.4f, ' ...
             'ildl step-1 its=%d, true_res=%.1e vs split relres=%.1e)\n'], ...
            Sb.n, Sb.nC, dc_ours, r1.iters, r1.true_res, r1.relres);
    npass = npass + 1;
else
    fprintf('  SKIP T7: %s not found\n', csv);
end

fprintf('=== test_build_stokes_sequence: %d checks passed ===\n', npass);
