% SCHUR_EXTRACT_EXAMPLES  Save the dense SPD Schur complement at TWO time steps.
%
% Writes, for each step in STEPS, one .mat holding the explicit Schur complement
% S(t_n) = D + G*Avel^{-1}*G' (pin dropped), its right-hand side, the ground
% truth multiplier and a meta struct:
%
%     schur_example_h0p03_step01.mat   ->  S, rhs_S, y_ref, keep, meta
%     schur_example_h0p03_step09.mat   ->  S, rhs_S, y_ref, keep, meta
%
% S is DENSE and SPD.  At h0 = 0.03 that is nS ~ 5258, i.e. ~221 MB PER FILE.
% The files are GITIGNORED and regenerable -- rerun this script.
%
% THE PIN.  S has the decoupled pressure-pin index removed (see
% schur_step_operator.m); `keep` is the nS_full x 1 logical that maps reduced
% indices back to the full (p, lambda) block, and meta.pin_node / meta.pin_val
% carry the dropped value.  Reconstruct the full multiplier with
%     y = zeros(meta.nS_full,1); y(keep) = y_ref; y(meta.pin_node) = meta.pin_val;
% The recovery handle st.recover is deliberately NOT saved: it captures a
% decomposition object and a dense nU x nS block.
%
% CROSS-CHECK against the KKT twin.  This script and
% ../stokes_immersed_rotor/extract_kkt_examples.m march the SAME sequence at the
% same h0 through two INDEPENDENT assembly implementations (schur_assemble_kkt
% here, build_stokes_sequence there).  meta carries a shared fingerprint:
%     s = load('schur_example_h0p03_step09.mat', 'meta');
%     k = load('../stokes_immersed_rotor/stokes_kkt_example_h0p03_step09.mat', 'meta');
%     [s.meta.normK_fro, k.meta.normK_fro]      % must agree to ~1e-12
%
% NO NEW ASSEMBLY CODE: everything is delegated to this folder's own
% schur_context_init / schur_step_operator pipeline.
%
% See also: schur_step_operator, schur_context_init, schur_make_cfg,
%           tests/test_schur_correctness, extract_kkt_examples.

clear; clc;

thisFileDir = fileparts(mfilename('fullpath'));
add_schur_paths();
assert_local_helpers();     % guards against the sibling folder shadowing us.
                            % Do NOT call add_recycle_paths here: it PREPENDS
                            % the sibling rotor dir, defeating the '-end'
                            % ordering add_schur_paths sets up.
rng(1);

%==========================================================================
%  Configuration
%==========================================================================
CASE_NAME = 'bar_rotating';    % the stress case
H0        = 0.03;              % set 0.1 for a seconds-long smoke run
COMPUTE_SPECTRUM = false;      % full eig on a 5258-dense S costs minutes and
                               % two more 221 MB temporaries

% WHICH TWO STEPS -- see the theta table in extract_kkt_examples.m.  Short
% version: the bar's point set is symmetric under a pi rotation, so
% C(theta+pi) = P*C(theta) exactly and separation is only meaningful mod pi.
% Step 9 is 85.6 deg from step 1 (the maximum); steps 16 and 31 are within
% 3-6 deg of a permutation of step 1 and are the WORST possible partners.
% Use [5 13] for the same separation between two developed-flow steps
% (step 1 is atypical: u_prev = 0).
STEPS = [1 9];

%==========================================================================
%  Time-constant context (the expensive part)
%==========================================================================
params    = make_schur_params();
params.h0 = H0;

% Tstep sets Tmax and hence the rotor's angular velocity -- shrinking it would
% change the geometry rather than shorten the run.
assert(params.Tstep == 61 && abs(params.dt - 0.02) < 1e-12, ...
       'unexpected params.Tstep/dt (%d, %g): the step->theta table assumes 61/0.02', ...
       params.Tstep, params.dt);
assert(max(STEPS) <= params.Tstep - 1, ...
       'step %d exceeds the sequence length %d', max(STEPS), params.Tstep - 1);

fprintf('[schur_extract_examples] meshing %s at h0=%.3g...\n', CASE_NAME, H0);
[cfg, msh] = schur_make_cfg(CASE_NAME, params, []);

% nS = nP + nC - 1; nC is only known after the first coupling assembly, so this
% is a lower bound -- printed BEFORE the expensive work so an oversized run is
% caught early.
fprintf(['[schur_extract_examples] N=%d  nU=%d  nP=%d  ->  nS = %d + nC - 1, ' ...
         'at least %.0f MB per dense S\n'], ...
        msh.N, 2*msh.N, msh.N, msh.N, msh.N^2 * 8 / 2^20);

fprintf('[schur_extract_examples] building context (chol of Avel, S_pp)...\n');
ctx = schur_context_init(cfg, params);

% Y_B is dead after schur_context_init.m:98 -- nothing downstream reads it
% (schur_step_operator uses only ctx.dA, ctx.GtB, ctx.S_pp).  Dropping it here
% frees nU*nP*8 bytes without touching shared code.
ctx.Y_B = [];

%==========================================================================
%  March the sequence, saving the requested steps
%==========================================================================
tag    = ['h' strrep(num2str(H0), '.', 'p')];   % 0.03 -> h0p03
u_prev = zeros(ctx.nU, 1);
Csave  = cell(numel(STEPS), 1);
nC_ref = [];

for n = 1:max(STEPS)
    tcur = n * params.dt;
    st   = schur_step_operator(ctx, tcur, u_prev);

    if isempty(nC_ref)
        nC_ref = st.nC;
    else
        assert(st.nC == nC_ref, ['nC changed from %d to %d at step %d -- the ' ...
               'two saved operators would have different sizes'], nC_ref, st.nC, n);
    end

    x_ref = st.K \ st.b;                      % ground truth on the FULL KKT

    k = find(STEPS == n, 1);
    if ~isempty(k)
        S     = st.S;
        rhs_S = st.rhs_S;
        keep  = st.keep;

        % --- the gate: the Schur solve must reproduce K\b -------------------
        sym_res    = norm(S - S', 'fro') / max(norm(S, 'fro'), eps);
        [R, cflag] = chol(S);                 % SPD check and the solve, one factor
        if cflag == 0
            y_ref = R \ (R' \ rhs_S);
        else
            y_ref = nan(size(rhs_S));
        end
        clear R;
        relerr = norm(st.recover(y_ref) - x_ref) / max(norm(x_ref), eps);
        assert(sym_res < 1e-14, 'step %d: S is not symmetric (%.3e)', n, sym_res);
        assert(cflag == 0, 'step %d: S is not positive definite', n);
        assert(relerr < 1e-10, ...
               'step %d: recover(S\\rhs_S) ~= K\\b (%.3e)', n, relerr);

        % --- metadata -------------------------------------------------------
        meta = struct( ...
            'nS', size(S, 1), 'nS_full', numel(keep), ...
            'nU', ctx.nU, 'nP', ctx.nP, 'nC', st.nC, 'N', ctx.N, ...
            'h0', ctx.h0, 'dt', ctx.dt, 'Tstep', params.Tstep, ...
            'Tmax', params.dt * params.Tstep, ...
            'nu', ctx.nu, 'eps_stab', ctx.eps_stab, 'case_name', CASE_NAME, ...
            'step', n, 't_snap', tcur, ...
            'theta_rad', 2*pi*2*tcur / (params.dt * params.Tstep), ...
            'theta_deg', 360*2*tcur / (params.dt * params.Tstep), ...
            'pin_node', ctx.pin_node, 'pin_val', ctx.pin_val, ...
            'sym_res', sym_res, 'chol_ok', cflag == 0, ...
            'relerr_vs_kkt', relerr, ...
            'dense_frac', nnz(abs(S) > 1e-12 * max(abs(S(:)))) / numel(S), ...
            'bytes_S', numel(S) * 8, ...
            ... % cross-artifact fingerprint -- must match the KKT twin's meta
            'normK_fro', norm(st.K, 'fro'), 'nnzK', nnz(st.K), ...
            'norm_b', norm(st.b), 'normC_fro', norm(st.C, 'fro'), ...
            'created', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
            'matlab_version', version);

        if COMPUTE_SPECTRUM
            ev = eig(S);  %#ok<UNRCH>
            meta.lambda_min = min(ev);
            meta.lambda_max = max(ev);
            meta.cond       = max(ev) / min(ev);
        end

        % --- save, then free before the next step ---------------------------
        matName = sprintf('schur_example_%s_step%02d.mat', tag, n);
        matFile = fullfile(thisFileDir, matName);
        save(matFile, 'S', 'rhs_S', 'y_ref', 'keep', 'meta', ...
             '-v7.3', '-nocompression');

        d = dir(matFile);
        fprintf(['[schur_extract_examples] step %2d  t=%.3f  theta=%6.1f deg  ' ...
                 'nS=%d  dense=%.0f%%  symres=%.1e  relerr=%.1e  -> %s (%.0f MB)\n'], ...
                n, tcur, meta.theta_deg, meta.nS, 100*meta.dense_frac, ...
                sym_res, relerr, matName, d.bytes/2^20);

        Csave{k} = st.C;
        clear S rhs_S y_ref keep;
    end

    u_prev = x_ref(1:ctx.nU);
    clear st x_ref;
end

%==========================================================================
%  The two steps must actually differ
%==========================================================================
if numel(STEPS) == 2
    dC = norm(Csave{2} - Csave{1}, 'fro') / max(norm(Csave{1}, 'fro'), eps);
    fprintf('[schur_extract_examples] ||C_%d - C_%d||_F / ||C_%d||_F = %.4f\n', ...
            STEPS(2), STEPS(1), STEPS(1), dC);
    assert(dC > 1e-3, ['steps %d and %d have near-identical coupling ' ...
           '(dC = %.2e) -- they are probably a multiple of pi apart'], ...
           STEPS(1), STEPS(2), dC);
end

fprintf('[schur_extract_examples] done. Files are gitignored; rerun to regenerate.\n');
