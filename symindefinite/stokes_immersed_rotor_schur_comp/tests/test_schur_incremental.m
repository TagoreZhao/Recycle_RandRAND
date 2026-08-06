% TEST_SCHUR_INCREMENTAL  The hoisted (p,p) block is exact and time-constant.
%
% schur_context_init hoists S_pp = D_pp + GtB'*(A^{-1}*GtB) out of the time
% loop, so each step only pays nC backsolves instead of nP+nC.  This test
% checks the shortcut is exact (vs a from-scratch S) and that S_pp really is
% invariant -- the structural fact the whole recycling story rests on.
%
% Run:  cd tests; test_schur_incremental

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(fileparts(thisFileDir));
add_schur_paths();
rng(1);

params = make_schur_params();
params.h0 = 0.1;
nsteps = 4;

[cfg, ~] = schur_make_cfg('bar_rotating', params, []);
ctx = schur_context_init(cfg, params);
nU = ctx.nU;  nP = ctx.nP;

u_prev = zeros(nU, 1);
S_prev = [];
Spp_ref = [];
npass = 0;

for n = 1:nsteps
    tcur = n * params.dt;
    st = schur_step_operator(ctx, tcur, u_prev);
    nC = st.nC;

    % --- T1: incremental S equals from-scratch S -------------------------
    GtC = st.K(1:nU, nU + nP + (1:nC));
    Gt  = [ctx.GtB, GtC];
    D   = -st.K(nU+1:end, nU+1:end);
    S_scratch = full(D) + Gt' * (ctx.dA \ full(Gt));
    S_scratch = (S_scratch + S_scratch') / 2;
    S_scratch = S_scratch(st.keep, st.keep);

    relerr = norm(st.S - S_scratch, 'fro') / norm(S_scratch, 'fro');
    assert(relerr < 1e-12, ...
        'T1 [step %d]: incremental S differs from scratch, rel err %.3e', n, relerr);
    npass = npass + 1;

    % --- T2: the (p,p) block is bit-identical across steps ---------------
    Spp_now = st.S(1:nP-1, 1:nP-1);      % pin removed from the pressure block
    if isempty(Spp_ref)
        Spp_ref = Spp_now;
    else
        assert(isequal(Spp_now, Spp_ref), ...
            'T2 [step %d]: the (p,p) block of S changed between steps', n);
        npass = npass + 1;
    end

    % --- T3: step-to-step change is confined to the multiplier border ----
    if ~isempty(S_prev) && size(S_prev, 1) == size(st.S, 1)
        Ddiff = abs(st.S - S_prev);
        block_pp = Ddiff(1:nP-1, 1:nP-1);
        assert(max(block_pp(:)) == 0, ...
            'T3 [step %d]: (p,p) block moved by %.3e (must be exactly 0)', ...
            n, max(block_pp(:)));
        npass = npass + 1;

        % rank of the update must be at most 2*nC
        r = rank(st.S - S_prev, 1e-10 * norm(st.S, 'fro'));
        assert(r <= 2 * nC, ...
            'T3b [step %d]: update rank %d exceeds 2*nC = %d', n, r, 2 * nC);
        npass = npass + 1;
        fprintf('  step %d: nC=%3d  rank(S_n - S_{n-1}) = %3d  (bound 2nC = %3d)\n', ...
                n, nC, r, 2 * nC);
    end

    S_prev = st.S;
    u_prev = (st.K \ st.b);
    u_prev = u_prev(1:nU);
end

fprintf('\n  S: %d x %d, DENSE (%.1f%% of entries above 1e-12 of the max)\n', ...
        size(st.S, 1), size(st.S, 2), ...
        100 * nnz(abs(st.S) > 1e-12 * max(abs(st.S(:)))) / numel(st.S));
fprintf('test_schur_incremental: ALL %d ASSERTIONS PASSED\n', npass);
