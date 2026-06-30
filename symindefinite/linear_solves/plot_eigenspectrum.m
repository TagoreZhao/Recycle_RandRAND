% PLOT_EIGENSPECTRUM  Log-scale |lambda| eigenspectrum of the symmetric-indefinite
% Stokes KKT matrix, raw vs incomplete-LDL preconditioned.
%
% Loads the (A, b) pair from extract_system.m, builds the SPD incomplete-LDL
% preconditioner M = C C' (make_ildl_precond), and plots the SMALLEST-500 and
% LARGEST-500 eigenvalues by absolute value on a log axis for both the raw matrix
% A and the preconditioned operator M^-1 A (generalized eigs of (A, M)).  The
% clustering of the preconditioned spectrum is what drives the MINRES speedup.
%
% Run extract_system.m first.
%
% See also: make_ildl_precond, test_ildl_minres,
%           stokes_immersed_rotor/run_spectrum_spy (safe_eigs, palette).

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
repoRoot = fileparts(fileparts(thisFileDir));   % .../Recycle_RandRAND (for +src)
addpath(repoRoot);
import src.precond.*                             % make_ildl_precond
rng(1);

outDir = fullfile(thisFileDir, 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ---- load system ---------------------------------------------------------
matFile = fullfile(thisFileDir, 'stokes_kkt_system.mat');
assert(exist(matFile, 'file') == 2, ...
       'stokes_kkt_system.mat not found — run extract_system.m first.');
S = load(matFile);
A = S.A;
n = size(A, 1);
k = 500;
assert(k < n, 'k=%d must be < n=%d', k, n);
fprintf('[spectrum] A: n=%d  nnz=%d\n', n, nnz(A));

% ---- SPD incomplete-LDL preconditioner M = C C' --------------------------
% Rebuild C explicitly from the documented factorization C = S^-1 P^T L |D|^{1/2}
% (factors exposed by make_ildl_precond); the solver apply-handles are untouched.
P    = make_ildl_precond(A, struct('mode', 'nofill'));
Sinv = spdiags(1 ./ P.s, 0, n, n);
Pt   = sparse(P.p, (1:n)', 1, n, n);          % P^T: scatters row i -> p(i)
C    = Sinv * Pt * P.L * P.Dsqrt;
M    = C * C';
M    = (M + M') / 2;                          % SPD preconditioner
fprintf('[spectrum] preconditioner M: nnz=%d\n', nnz(M));

% ---- eigenvalues (eigs, never full eig) ----------------------------------
fprintf('[spectrum] computing smallest/largest %d |lambda| (raw + preconditioned)...\n', k);
las = real(safe_eigs(A,    k, 'smallestabs'));   % raw,  near zero
lal = real(safe_eigs(A,    k, 'largestabs'));    % raw,  extremes
lps = real(safe_eigs(A, M, k, 'smallestabs'));   % prec, near zero
lpl = real(safe_eigs(A, M, k, 'largestabs'));    % prec, extremes

report('raw  small', las);
report('raw  large', lal);
report('prec small', lps);
report('prec large', lpl);
assert(min([las; lal]) < 0 && max([las; lal]) > 0, ...
       'raw A is not indefinite — check the extracted system');

% ---- figure: |lambda| on log axis, smallest | largest -------------------
POS = [0.85 0.40 0.32];   % warm: lambda > 0
NEG = [0.20 0.45 0.70];   % cool: lambda < 0

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 1180 520]);
tl  = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile(tl);
plot_abs_panel(las, lps, POS, NEG);
title(sprintf('smallest %d  |\\lambda|', k));
ylabel('|\lambda|   (log scale)');

ax2 = nexttile(tl);
plot_abs_panel(lal, lpl, POS, NEG);
title(sprintf('largest %d  |\\lambda|', k));

% Proxy handles so the legend always shows all four categories, even where a
% panel has an empty sign-class (raw A is all-negative at the small end,
% all-positive at the large end).
hold(ax2, 'on');
hp(1) = semilogy(ax2, nan, nan, 'o', 'MarkerSize', 3, 'MarkerFaceColor', POS, 'MarkerEdgeColor', POS);
hp(2) = semilogy(ax2, nan, nan, 'o', 'MarkerSize', 3, 'MarkerFaceColor', NEG, 'MarkerEdgeColor', NEG);
hp(3) = semilogy(ax2, nan, nan, 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', POS, 'LineWidth', 0.9);
hp(4) = semilogy(ax2, nan, nan, 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', NEG, 'LineWidth', 0.9);
lg = legend(hp, {'raw  \lambda>0', 'raw  \lambda<0', ...
                 'precond  \lambda>0', 'precond  \lambda<0'}, ...
            'Orientation', 'horizontal');
lg.Layout.Tile = 'south';
title(tl, 'Stokes KKT eigenspectrum: raw vs incomplete-LDL preconditioned', ...
      'FontWeight', 'bold');
set([ax1 ax2], 'YScale', 'log');              % enforce after legend/title relayout

figPath = fullfile(outDir, 'eig_abs_spectrum.png');
exportgraphics(fig, figPath, 'Resolution', 180);
close(fig);
fprintf('[spectrum] saved %s\n', figPath);

%==========================================================================
%  Local functions
%==========================================================================
function plot_abs_panel(lraw, lprec, POS, NEG)
%PLOT_ABS_PANEL  Scatter |lambda| (log y) sorted by magnitude; raw = filled,
% preconditioned = open; color encodes sign (warm +, cool -).
    lraw  = sort_by_abs(lraw);
    lprec = sort_by_abs(lprec);

    xr = 1:numel(lraw);
    xp = 1:numel(lprec);
    rp = lraw  > 0;   rn = ~rp;
    pp = lprec > 0;   pn = ~pp;

    semilogy(xr(rp), abs(lraw(rp)),  'o', 'MarkerSize', 3, ...
             'MarkerFaceColor', POS, 'MarkerEdgeColor', POS); hold on;
    semilogy(xr(rn), abs(lraw(rn)),  'o', 'MarkerSize', 3, ...
             'MarkerFaceColor', NEG, 'MarkerEdgeColor', NEG);
    semilogy(xp(pp), abs(lprec(pp)), 'o', 'MarkerSize', 4, ...
             'MarkerFaceColor', 'none', 'MarkerEdgeColor', POS, 'LineWidth', 0.9);
    semilogy(xp(pn), abs(lprec(pn)), 'o', 'MarkerSize', 4, ...
             'MarkerFaceColor', 'none', 'MarkerEdgeColor', NEG, 'LineWidth', 0.9);
    set(gca, 'YScale', 'log');                % enforce: first series may be empty
    grid on; box on;
    xlabel('rank by |\lambda| (largest \rightarrow smallest)');
    xlim([0, max(numel(lraw), numel(lprec)) + 1]);
end

function v = sort_by_abs(v)
    [~, idx] = sort(abs(v), 'descend');   % rank 1 = largest |lambda| -> decreasing curve
    v = v(idx);
end

function report(tag, l)
    fprintf('  %-10s : #neg=%3d  #pos=%3d   |lambda| in [%.3e, %.3e]\n', ...
            tag, sum(l < 0), sum(l > 0), min(abs(l)), max(abs(l)));
end

function d = safe_eigs(varargin)
%SAFE_EIGS  eigs(...) for the smallest/largest k (copied from run_spectrum_spy).
% Standard (A,k,which) or generalized (A,B,k,which) call forms.
    if issparse(varargin{2})        % generalized: (A, B, k, which)
        A = varargin{1}; B = varargin{2}; k = varargin{3}; which = varargin{4};
        [~, D] = eigs(A, B, k, which, 'Tolerance', 1e-5, 'MaxIterations', 600);
    else                            % standard: (A, k, which)
        A = varargin{1}; k = varargin{2}; which = varargin{3};
        [~, D] = eigs(A, k, which, 'Tolerance', 1e-5, 'MaxIterations', 600);
    end
    d = diag(D);
end
