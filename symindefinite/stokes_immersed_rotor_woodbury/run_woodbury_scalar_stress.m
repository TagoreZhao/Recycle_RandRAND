%RUN_WOODBURY_SCALAR_STRESS  Make the Woodbury identity fail, on purpose, at kappa = 1.
%
%   Run:  run_woodbury_scalar_stress
%
%   The rotor benchmark reports a Woodbury forward error of ~2e-14.  That is only a
%   statement about the METHOD if the method is capable of failing, so this script
%   breaks it on two systems that are as well conditioned as systems get.  Both are
%   scalar or 1x1: kappa(A) = kappa(A + UCV) = kappa(S) = 1 throughout.  Nothing here
%   is ill posed.  What fails is the EVALUATION of an identity that is exact.
%
%   FAMILY 1 -- the final subtraction (the case in the brief).
%       A = [1], U = V = [1], C = [alpha], b = [1],   alpha = 1 ... 1e18
%   The exact solution is x = 1/(1+alpha).  Woodbury computes it as
%       z = 1,  Y = 1,  S = 1/alpha + 1,  w = alpha/(1+alpha),  x = z - Y w
%   i.e. as 1 - alpha/(1+alpha): two numbers of size 1 differenced to make one of
%   size 1e-18.  Near alpha = 1e16 the subtraction returns exactly 0 and every digit
%   is gone -- including the backward error, which reaches 1.
%
%   FAMILY 2 -- the small matrix.
%       a0 = 1/3,  A = [a0(1+eta)], U = V = [1], C = [-a0], b = [1]
%   Now S = C^{-1} + V A^{-1} U = -1/a0 + 1/A cancels two numbers of size 3 into one
%   of size 3*eta, so its one-ulp absolute error is an eps/eta RELATIVE error.  Here
%   the final subtraction is harmless (cancel_sub = 1 exactly) and the small matrix
%   is the whole story -- the two mechanisms are separated so that neither table can
%   be read as the other in disguise.
%
%   WHY THE REFERENCES ARE TRUSTWORTHY.  Family 1's solution is 1/(1+alpha), a closed
%   form accurate to eps in double.  Family 2's A and C = -a0 have the same magnitude,
%   so A + UCV = A - a0 is EXACT by Sterbenz and the solution is exactly 1/(A - a0),
%   never solved for.  Family 1 additionally runs the identical expression in ~32-digit
%   double-double arithmetic (dd_woodbury_scalar), which separates "the formula is
%   wrong" from "the precision was spent" -- it is the latter.
%
%   Writes two figures to woodbury_direct/.  See tests/test_stress_metrics.m for the
%   assertions that hold these numbers in place.
%
%   See also: woodbury_naive, dd_woodbury_scalar, run_woodbury_recursive,
%             run_woodbury_stability.

clear; clc;
paths = add_woodbury_paths();
outDir = paths.outDir;

fprintf('=== woodbury scalar stress: eps = %.4e ===\n', eps);

% =========================================================================
% FAMILY 1: alpha = 1 ... 1e18
% =========================================================================
ALPHA = logspace(0, 18, 73)';
na    = numel(ALPHA);
F1    = local_alloc(na);

for i = 1:na
    a = ALPHA(i);
    A = 1;  U = 1;  V = 1;  C = a;  b = 1;
    xex = 1 / (1 + a);

    [~, d] = woodbury_naive(A, U, C, V, b, xex);

    M   = A + U * C * V;
    xbs = M \ b;
    rbs = b - M * xbs;

    F1.fwd(i)    = d.fwd;
    F1.bwd(i)    = d.bwd;
    F1.fwd_bs(i) = abs(xbs - xex) / abs(xex);
    F1.bwd_bs(i) = norm(rbs) / (norm(M) * norm(xbs) + norm(b));
    F1.fwd_dd(i) = abs(dd_woodbury_scalar(a) - xex) / abs(xex);
    F1.kA(i)     = d.kappa_A;
    F1.kM(i)     = d.kappa_M;
    F1.kS(i)     = d.kappa_S;
    F1.cS(i)     = d.cancel_S;
    F1.cSub(i)   = d.cancel_sub;
end

fprintf('\n--- family 1: A=[1] U=V=[1] C=[alpha] b=[1], x = 1/(1+alpha) ---\n');
local_table(ALPHA, F1, 'alpha', 1:4:na);

fprintf(['\nEvery condition number is 1 and the direct solve never moves off %.1e.\n' ...
         'The Woodbury error is predicted by cancel_sub*eps, NOT by any kappa:\n' ...
         '  max err/(cancel_sub*eps) = %.3f over the whole sweep.\n'], ...
        max(F1.fwd_bs), max(F1.fwd ./ max(F1.cSub * eps, realmin)));

izero = find(F1.fwd >= 1, 1, 'first');
if ~isempty(izero)
    fprintf(['At alpha = %.2e the subtraction returns EXACTLY 0: x_hat = 0 while the\n' ...
             'true answer is %.2e.  The backward error there is %.3f -- the computed\n' ...
             'iterate does not solve any nearby system either.\n'], ...
            ALPHA(izero), 1/(1+ALPHA(izero)), F1.bwd(izero));
end
fprintf('The same expression at 32 digits returns %.2e relative error: the formula is\n', ...
        max(F1.fwd_dd));
fprintf('fine, the working precision is what the cancellation consumed.\n');

% =========================================================================
% FAMILY 2: eta = 5e-16 ... 5e-5
% =========================================================================
A0  = 1 / 3;                                  % NOT a power of two -- see below
ETA = 10 .^ -linspace(15.3, 4.3, 45)';
ne  = numel(ETA);
F2  = local_alloc(ne);

for i = 1:ne
    A = A0 * (1 + ETA(i));  U = 1;  V = 1;  C = -A0;  b = 1;
    xex = 1 / (A - A0);                       % A + UCV == A - a0 exactly

    [~, d] = woodbury_naive(A, U, C, V, b, xex);

    M   = A + U * C * V;
    xbs = M \ b;
    rbs = b - M * xbs;

    F2.fwd(i)    = d.fwd;
    F2.bwd(i)    = d.bwd;
    F2.fwd_bs(i) = abs(xbs - xex) / abs(xex);
    F2.bwd_bs(i) = norm(rbs) / (norm(M) * norm(xbs) + norm(b));
    F2.fwd_dd(i) = NaN;                       % no closed form to compensate here
    F2.kA(i)     = d.kappa_A;
    F2.kM(i)     = d.kappa_M;
    F2.kS(i)     = d.kappa_S;
    F2.cS(i)     = d.cancel_S;
    F2.cSub(i)   = d.cancel_sub;
end

fprintf('\n--- family 2: A=[a0(1+eta)] U=V=[1] C=[-a0] b=[1], a0 = 1/3 ---\n');
local_table(ETA, F2, 'eta', 1:4:ne);

r2 = F2.fwd ./ max(F2.cS * eps, realmin);
fprintf(['\nSame conclusion, other mechanism: cancel_sub is exactly 1 here, so the\n' ...
         'final subtraction is innocent, and the error tracks cancel_S*eps instead --\n' ...
         '  err/(cancel_S*eps): median %.3f, max %.3f, over 11 decades of eta.\n'], ...
        median(r2), max(r2));
fprintf(['The bound is never exceeded and is usually within an order of it.  The low\n' ...
         'outliers are eta values where fl(1/A) happens to land near-exactly on a\n' ...
         'double, so the ulp that the mechanism amplifies was not there to lose.\n']);
fprintf(['\nWHY a0 = 1/3 AND NOT 1.  The obvious A = 1+eta is degenerate: the true\n' ...
         '1/(1+eta) = 1 - eta + eta^2 sits within eta^2 of the representable 1-eta, so\n' ...
         'fl(1/A) carries a rounding of eta^2 instead of eps and the mechanism never\n' ...
         'fires (the error saturates near sqrt(eps) ~ 1e-8).  Anchoring off a power of\n' ...
         'two puts 1/A in a generic position between doubles and the full ulp is lost.\n']);

% =========================================================================
% Figures
% =========================================================================
local_figure(ALPHA, F1, '\alpha', ...
    'Woodbury on A=[1], U=V=[1], C=[\alpha], b=[1]', ...
    'cancel_{sub}\cdot\epsilon', F1.cSub * eps, ...
    fullfile(outDir, 'stress_alpha_subtraction.png'));

local_figure(ETA, F2, '\eta', ...
    'Woodbury on A=[a_0(1+\eta)], U=V=[1], C=[-a_0], a_0 = 1/3', ...
    'cancel_S\cdot\epsilon', F2.cS * eps, ...
    fullfile(outDir, 'stress_eta_small_matrix.png'));

%==========================================================================
function F = local_alloc(n)
%LOCAL_ALLOC  One row per sweep point, one field per reported quantity.
    z = nan(n, 1);
    F = struct('fwd', z, 'bwd', z, 'fwd_bs', z, 'bwd_bs', z, 'fwd_dd', z, ...
               'kA', z, 'kM', z, 'kS', z, 'cS', z, 'cSub', z);
end

%==========================================================================
function local_table(param, F, pname, rows)
%LOCAL_TABLE  Print a readable subset of the sweep.
    fprintf('%10s %10s %10s %10s %10s %8s %8s %8s %10s %10s\n', pname, ...
            'err_wood', 'err_bslash', 'err_dd', 'bwd_wood', 'k(A)', 'k(M)', ...
            'k(S)', 'cancel_S', 'cancel_sub');
    for i = rows
        fprintf('%10.2e %10.3e %10.3e %10.3e %10.3e %8.2f %8.2f %8.2f %10.3e %10.3e\n', ...
                param(i), F.fwd(i), F.fwd_bs(i), F.fwd_dd(i), F.bwd(i), ...
                F.kA(i), F.kM(i), F.kS(i), F.cS(i), F.cSub(i));
    end
end

%==========================================================================
function local_figure(param, F, pname, ttl, predname, pred, outFile)
%LOCAL_FIGURE  Four panels: what failed, what did not, and what predicted it.
    opts = woodbury_fig_defaults();
    fh = figure('Visible', 'off', 'Units', 'inches', ...
                'Position', [1 1 opts.multi_width opts.multi_height + 1]);
    tl = tiledlayout(fh, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, ttl, 'FontSize', opts.titlefontsize, 'Interpreter', 'tex');

    fl = @(v) max(v, 1e-18);          % a log axis cannot show an exact zero

    ax = nexttile(tl);
    loglog(ax, param, fl(F.fwd), '-o', 'LineWidth', 1.6, 'MarkerSize', 4); hold(ax, 'on');
    loglog(ax, param, fl(F.fwd_bs), '-s', 'LineWidth', 1.4, 'MarkerSize', 4);
    if any(isfinite(F.fwd_dd))
        loglog(ax, param, fl(F.fwd_dd), '-^', 'LineWidth', 1.4, 'MarkerSize', 4);
        lg = {'Woodbury (naive)', 'direct solve', 'Woodbury, 32 digits'};
    else
        lg = {'Woodbury (naive)', 'direct solve'};
    end
    loglog(ax, param, fl(pred), 'k--', 'LineWidth', 1.2);
    ylabel(ax, 'relative forward error');  xlabel(ax, pname);
    legend(ax, [lg, {predname}], 'Location', 'southeast', ...
           'FontSize', opts.legendfontsize);

    ax = nexttile(tl);
    loglog(ax, param, fl(F.bwd), '-o', 'LineWidth', 1.6, 'MarkerSize', 4); hold(ax, 'on');
    loglog(ax, param, fl(F.bwd_bs), '-s', 'LineWidth', 1.4, 'MarkerSize', 4);
    ylabel(ax, '||r|| / (||A+UCV|| ||x|| + ||b||)');  xlabel(ax, pname);
    legend(ax, {'Woodbury (naive)', 'direct solve'}, 'Location', 'northwest', ...
           'FontSize', opts.legendfontsize);
    title(ax, 'normwise backward error', 'FontSize', opts.subtitlefontsize);

    % The panel that makes the point: nothing is ill conditioned.
    ax = nexttile(tl);
    semilogx(ax, param, F.kA, '-o', 'LineWidth', 1.6, 'MarkerSize', 4); hold(ax, 'on');
    semilogx(ax, param, F.kM, '-s', 'LineWidth', 1.4, 'MarkerSize', 4);
    semilogx(ax, param, F.kS, '-^', 'LineWidth', 1.4, 'MarkerSize', 4);
    ylim(ax, [0 4]);
    ylabel(ax, 'condition number');  xlabel(ax, pname);
    legend(ax, {'\kappa(A)', '\kappa(A+UCV)', '\kappa(S)'}, 'Location', 'north', ...
           'FontSize', opts.legendfontsize, 'Orientation', 'horizontal');
    title(ax, 'every condition number is O(1)', 'FontSize', opts.subtitlefontsize);

    ax = nexttile(tl);
    loglog(ax, param, F.cS, '-o', 'LineWidth', 1.6, 'MarkerSize', 4); hold(ax, 'on');
    loglog(ax, param, fl(F.cSub), '-s', 'LineWidth', 1.4, 'MarkerSize', 4);
    ylabel(ax, 'cancellation factor');  xlabel(ax, pname);
    legend(ax, {'small matrix: (||C^{-1}||+||V||||A^{-1}U||)/||S||', ...
                'subtraction: (||z||+||Yw||)/||z-Yw||'}, ...
           'Location', 'northwest', 'FontSize', opts.legendfontsize);
    title(ax, 'the two cancellation sites', 'FontSize', opts.subtitlefontsize);

    save_woodbury_figure(fh, outFile, opts);
end
