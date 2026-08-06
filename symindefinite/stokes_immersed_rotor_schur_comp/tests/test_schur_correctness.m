% TEST_SCHUR_CORRECTNESS  The gate: the Schur reduction must reproduce K\b.
%
% Solving S*y = rhs_S, scattering the pinned pressure DOF back, and recovering
% the velocity must return EXACTLY the full KKT solution.  Everything else in
% this study rests on this, so it runs first and on all three motions.
%
% Run:  cd tests; test_schur_correctness

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(fileparts(thisFileDir));
add_schur_paths();
assert_local_helpers();
rng(1);

params = make_schur_params();
params.h0 = 0.1;                 % coarse twin: fast, same structure
nsteps = 3;

case_names = {'bar_rotating', 'disk_translating', 'disk_static'};
msh = [];
npass = 0;

for k = 1:numel(case_names)
    cname = case_names{k};
    [cfg, msh] = schur_make_cfg(cname, params, msh);
    ctx = schur_context_init(cfg, params);

    u_prev = zeros(ctx.nU, 1);
    for n = 1:nsteps
        tcur = n * params.dt;
        st = schur_step_operator(ctx, tcur, u_prev);

        % --- ground truth ---
        x_ref = st.K \ st.b;

        % --- Schur route ---
        y = st.S \ st.rhs_S;
        x = st.recover(y);

        relerr = norm(x - x_ref) / max(norm(x_ref), eps);
        assert(relerr < 1e-10, ...
            'T1 [%s step %d]: Schur solve != K\\b, rel err %.3e', cname, n, relerr);
        npass = npass + 1;

        % --- constrained velocity DOFs come back exactly ---
        u = x(1:ctx.nU);
        assert(isequal(u(cfg.veldofs), cfg.velvals), ...
            'T2 [%s step %d]: u(veldofs) ~= velvals exactly', cname, n);
        npass = npass + 1;

        % --- S is symmetric and SPD ---
        symres = norm(st.S - st.S', 'fro') / max(norm(st.S, 'fro'), eps);
        assert(symres < 1e-14, ...
            'T3 [%s step %d]: S not symmetric, residual %.3e', cname, n, symres);
        [~, cflag] = chol(st.S);
        assert(cflag == 0, ...
            'T4 [%s step %d]: chol(S) failed -- S is not SPD', cname, n);
        npass = npass + 2;

        if n == 1
            fprintf('  [%-16s] nS=%4d  nC=%3d  relerr=%.2e  cond=%.3e\n', ...
                    cname, st.nS, st.nC, relerr, cond(st.S));
        end

        u_prev = x_ref(1:ctx.nU);   % advance with the ground truth
    end
end

fprintf('\ntest_schur_correctness: ALL %d ASSERTIONS PASSED\n', npass);
