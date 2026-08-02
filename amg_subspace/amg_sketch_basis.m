function [V, info] = amg_sketch_basis(Mfun, n, m_sk, q, seed, state)
%AMG_SKETCH_BASIS  Gaussian sketch of a preconditioner into a deflation basis.
%
%   [V, info] = amg_sketch_basis(Mfun, n, m_sk, q, seed) draws a Gaussian
%   block Omega = randn(n, m_sk) and runs q steps of subspace iteration driven
%   by the preconditioner Mfun,
%       V <- orth(Mfun(V)),   repeated q times,
%   returning an orthonormal V.  Because Mfun approximates A^{-1}, the block is
%   pushed toward the SMALLEST eigenvectors of A -- the directions deflation
%   wants.
%
%   Oversampling, no truncation.  m_sk is meant to exceed the number of
%   eigendirections actually sought (the study uses m_sk = 2*k_target); ALL
%   surviving columns are returned.  There is no Rayleigh-Ritz truncation back
%   down to k_target, so the realized deflation dimension is info.r_defl, which
%   equals m_sk unless Mfun is rank-limited (a pure coarse-grid-correction AMG
%   has rank coarseN, so a wider sketch cannot raise the rank past it).
%
%   FAIRNESS.  The basis is a function of (Mfun, n, m_sk, q, seed) and nothing
%   else.  No eigenvectors, no exact inverse, no cached spectral data enter --
%   which is what makes "AMG used as a preconditioner" and "AMG sketched into a
%   deflation basis" a comparison at matched information.
%
%   [V, info] = amg_sketch_basis(Mfun, n, m_sk, q, seed, state) continues an
%   earlier call instead of restarting: state must be the info struct of a
%   previous call (carrying .V and .q) with the same (n, m_sk, seed), and only
%   the additional q - state.q iterations are performed.  Sweeping q upward
%   this way costs max(q) applies in total rather than sum(q).
%
%   Inputs
%     Mfun  : @(X) M*X, block-capable preconditioner apply
%     n     : problem dimension
%     m_sk  : sketch width (number of Gaussian columns)
%     q     : number of preconditioner applications
%     seed  : rng seed for Omega; fixing it makes the sketch reproducible and
%             lets different preconditioners be compared on the same Omega
%     state : optional info struct from a previous call, for incremental q
%
%   Outputs
%     V    : n-by-r_defl orthonormal basis
%     info : struct with fields
%              .V              the same basis (so info can be fed back as state)
%              .q              iterations performed so far
%              .m_sk           sketch width requested
%              .r_defl         size(V, 2) -- realized rank
%              .rank_limited   true iff r_defl < m_sk
%              .seed, .n
%              .time_seconds   wall time of the applies + orth in this call
%
%   See also AMG_SKETCH_TAU, AMG_DEFLATION_ARMS, MAKE_AMG_PREC_ABLATE.

    validateattributes(n,    {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(m_sk, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(q,    {'numeric'}, {'scalar', 'integer', 'nonnegative'});
    if ~isa(Mfun, 'function_handle')
        error('amg_sketch_basis:badMfun', 'Mfun must be a function handle.');
    end
    if m_sk > n
        error('amg_sketch_basis:widthTooLarge', ...
              'm_sk = %d exceeds n = %d.', m_sk, n);
    end

    q_done = 0;
    if nargin >= 6 && ~isempty(state)
        if state.n ~= n || state.m_sk ~= m_sk || state.seed ~= seed
            error('amg_sketch_basis:stateMismatch', ...
                  ['continuation state was built with (n=%d, m_sk=%d, seed=%d) ', ...
                   'but this call asks for (n=%d, m_sk=%d, seed=%d).'], ...
                  state.n, state.m_sk, state.seed, n, m_sk, seed);
        end
        if state.q > q
            error('amg_sketch_basis:stateAhead', ...
                  'continuation state is already at q = %d > %d.', state.q, q);
        end
        V      = state.V;
        q_done = state.q;
    else
        % Reproducible Gaussian start, restoring the caller's stream so the
        % driver's own rng bookkeeping (per-config reseeds) is untouched.
        rs = rng; cleanupRng = onCleanup(@() rng(rs));   %#ok<NASGU>
        rng(seed);
        V = randn(n, m_sk);
    end

    t0 = tic;
    for s = q_done + 1 : q
        V = orth(full(Mfun(V)));
    end
    % Final orthonormalization.  Redundant when q > q_done (the loop already
    % ends in orth) but required for the q = 0 case, where V is still the raw
    % Gaussian block.
    V = orth(full(V));
    t_elapsed = toc(t0);

    info = struct('V', V, 'q', q, 'm_sk', m_sk, 'r_defl', size(V, 2), ...
                  'rank_limited', size(V, 2) < m_sk, ...
                  'seed', seed, 'n', n, 'time_seconds', t_elapsed);
end
