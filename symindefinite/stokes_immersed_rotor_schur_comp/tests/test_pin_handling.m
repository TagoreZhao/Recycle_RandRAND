% TEST_PIN_HANDLING  The pressure pin makes the raw Schur complement indefinite.
%
% apply_dirichlet_sym sets K(dofs,dofs) = I, so after the pin
% K(nU+pin, nU+pin) = +1, i.e. D(pin,pin) = -1.  Because G(pin,:) = 0 the pin
% is fully DECOUPLED and contributes exactly one eigenvalue of -1.  This test
% pins down that claim -- that the offending direction really is e_pin and that
% dropping the index is lossless -- so the reduction is justified, not assumed.
%
% Run:  cd tests; test_pin_handling

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(fileparts(thisFileDir));
add_schur_paths();
rng(1);

params = make_schur_params();
params.h0 = 0.1;

[cfg, ~] = schur_make_cfg('bar_rotating', params, []);
ctx = schur_context_init(cfg, params);

tcur   = params.dt;
u_prev = zeros(ctx.nU, 1);
st     = schur_step_operator(ctx, tcur, u_prev);

% --- Rebuild the UNREDUCED S to inspect the pin directly ------------------
nU = ctx.nU;  nP = ctx.nP;  nC = st.nC;
GtC = st.K(1:nU, nU + nP + (1:nC));
Gt  = [ctx.GtB, GtC];
D   = -st.K(nU+1:end, nU+1:end);
S_full = full(D) + Gt' * (ctx.dA \ full(Gt));
S_full = (S_full + S_full') / 2;

pin = ctx.pin_node;

% T1: the pin diagonal is exactly -1
assert(abs(S_full(pin, pin) + 1) < 1e-12, ...
    'T1: S(pin,pin) = %.6g, expected -1', S_full(pin, pin));

% T2: the pin row/column is otherwise exactly zero (fully decoupled)
row = S_full(pin, :);  row(pin) = 0;
assert(norm(row, inf) < 1e-12, ...
    'T2: S(pin, keep) not zero, max |entry| = %.3e', norm(row, inf));

% T3: the unreduced S is indefinite, and the negative direction IS e_pin
% (nS is small, so take the exact dense spectrum rather than an eigs estimate)
[Vev, Dev] = eig(S_full, 'vector');
[lmin, imin] = min(Dev);
assert(lmin < 0, 'T3a: unreduced S should be indefinite, lambda_min = %.3e', lmin);
vmin = Vev(:, imin);
overlap = abs(vmin(pin)) / norm(vmin);
assert(overlap > 1 - 1e-8, ...
    'T3b: the negative eigenvector is not e_pin (|v(pin)|/||v|| = %.6f)', overlap);

% T4: exactly ONE negative eigenvalue
nneg = sum(Dev < 0);
assert(nneg == 1, 'T4: expected exactly 1 negative eigenvalue, found %d', nneg);

% T5: after dropping the pin, S is SPD
[~, cflag] = chol(st.S);
assert(cflag == 0, 'T5: chol(S_keep) failed -- reduced S is not SPD');

% T6: dropping is lossless -- keep-set matches the by-hand deletion
S_manual = S_full;  S_manual(pin, :) = [];  S_manual(:, pin) = [];
assert(norm(st.S - S_manual, 'fro') / norm(S_manual, 'fro') < 1e-14, ...
    'T6: st.S differs from the manually pin-deleted S');

fprintf('test_pin_handling: ALL PASSED\n');
fprintf('  pin index %d, S(pin,pin) = %.3f, negative eigenvalues = %d\n', ...
        pin, S_full(pin, pin), nneg);
fprintf('  lambda_min(S_keep) = %.4e, cond(S_keep) = %.4e\n', ...
        min(eig(st.S)), cond(st.S));
