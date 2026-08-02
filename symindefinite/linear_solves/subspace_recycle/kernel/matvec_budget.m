function [units, detail] = matvec_budget(spec)
%MATVEC_BUDGET  Per-step work in matvec-equivalents, so setup and iterations are
% commensurable.
%
%   [units, detail] = MATVEC_BUDGET(spec)
%
%   One UNIT = one application of the split operator Ahat = C^-1 K C^-T, i.e. one
%   sparse matvec plus two sparse triangular solves:
%
%       flops_unit = 2*nnzK + 4*nnzL
%
%   Reporting iteration counts alone hides two real costs: the coarse correction
%   is applied EVERY MINRES iteration, and E = V'Ahat^2 V is rebuilt every step
%   at 2k operator applies.  At the benchmark's k=500 the E build alone is ~546
%   units against a 312-unit ILDL-only budget, so a "5.8x fewer iterations"
%   result is still a wall-clock loss.  That is what these columns expose.
%
%   FLOP COUNTS ALONE ARE MISLEADING HERE, and the correction is large.  The
%   operator apply is dominated by two SPARSE TRIANGULAR SOLVES, which are
%   sequential and latency-bound; the coarse apply is dense BLAS-2, which is
%   threaded and near peak.  Measured on this machine (n=5840, k=500) the coarse
%   apply is 3.6x one operator apply, where a naive 8*n*k flop count predicts
%   ~68x — an 19x discrepancy, in the direction of the coarse apply being
%   CHEAPER than flops suggest.  DENSE_SPEEDUP below carries that factor; it is
%   machine-dependent, so re-measure with timeit (stage-4 run_cost_model.m)
%   before trusting a marginal verdict.  The operator and E-build terms need no
%   such correction: both are Ahat applies, counted against each other.
%
%   spec fields (all optional except n, nnzK, nnzL):
%     .n .nnzK .nnzL       system size and sparsity
%     .k                   coarse-space dimension (0 = ILDL only)
%     .iters               MINRES iterations
%     .build_E             true if E = V'Ahat^2 V was formed this step (2k units)
%     .qr_cols             columns orthonormalized this step (QR: 2*n*c^2 flops)
%     .n_backsolves        sparse backsolves against a stored factorization
%     .nnz_factor          nnz of that factorization (default 12*nnzL, a rough
%                          direct-solver fill factor; pass the real value when known)
%     .n_ldl               # ILDL factorizations built (default 0)
%     .ldl_flops           flops per ILDL build (default 20*nnzK, rough)
%     .n_eigs_k            dense k-by-k eig calls (10*k^3 flops)
%     .dense_speedup       flops/s advantage of dense BLAS-2 over the sparse
%                          triangular-solve-bound operator apply (default 19,
%                          measured; set 1 for a pure flop count)
%
%   detail breaks the total down by term so a CSV can show where the work went.
%
%   See also: two_level_it, lowrank_update_basis.

    n     = spec.n;
    nnzK  = spec.nnzK;
    nnzL  = spec.nnzL;
    k     = getdef(spec, 'k', 0);
    iters = getdef(spec, 'iters', 0);

    flops_unit = 2*nnzK + 4*nnzL;
    dsp        = getdef(spec, 'dense_speedup', 19);

    d = struct();
    d.operator = iters;                                   % one Ahat per iteration
    d.coarse_apply = iters * (8*n*k) / (flops_unit * dsp);  % Pdef, EVERY iteration
    d.coarse_build = 0;
    if getdef(spec, 'build_E', false)
        % 2k Ahat applies.  Counted as vector-equivalents; measured ~0.55 of
        % that (546 vs 1000 units at k=500) because Ahat2 is applied to the
        % whole n-by-k block at once.  Conservative in the right direction.
        d.coarse_build = 2 * k;
    end
    d.orth = 2 * n * getdef(spec, 'qr_cols', 0)^2 / flops_unit;
    d.backsolve = getdef(spec, 'n_backsolves', 0) * ...
                  (2 * getdef(spec, 'nnz_factor', 12*nnzL)) / flops_unit;
    d.ldl = getdef(spec, 'n_ldl', 0) * getdef(spec, 'ldl_flops', 20*nnzK) / flops_unit;
    d.dense_eig = getdef(spec, 'n_eigs_k', 0) * (10 * k^3) / flops_unit;

    d.setup = d.coarse_build + d.orth + d.backsolve + d.ldl + d.dense_eig;
    d.iter  = d.operator + d.coarse_apply;
    d.total = d.setup + d.iter;
    d.flops_unit = flops_unit;

    units  = d.total;
    detail = d;
end

%==========================================================================
function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
