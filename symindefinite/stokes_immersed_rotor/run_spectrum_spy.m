% RUN_SPECTRUM_SPY  Matrix-property graphs for the symmetric-indefinite Stokes
% immersed-rotor KKT system — the saddle-point analog of the SPD figures in
% subspace_capture/output/spectrum_spy_paper/.
%
% The per-step KKT matrix (assembled standalone here, exactly as the benchmark
% does it in +src/+stokes/solve_stokes_immersed.m and convergence_test.m Part D)
% is
%       K = [ Avel   B'      C'  ]     Avel = M2/dt + nu*A2   (2N x 2N, SPD)
%           [ B     -eps*L   0   ]     B    = divergence       (N  x 2N)
%           [ C      0       0   ]     -eps*L pressure stab, C moving coupling
% made nonsingular by SYMMETRIC Dirichlet + pressure-pin elimination, so K stays
% SYMMETRIC INDEFINITE (negative -eps*L block + zero multiplier block => negative
% eigenvalues, SPD velocity block => positive ones).
%
% Two things differ from the SPD reference and are handled accordingly:
%   1. ichol does NOT exist for an indefinite matrix.  The benchmark's correct
%      analog is the SPD block-diagonal preconditioner P = blkdiag(Avel, Dp/nu, I)
%      (see block_precond in solve_stokes_immersed.m); the MINRES-relevant
%      spectrum is the generalized eigenproblem K x = lambda P x.
%   2. semilogy hides the negative branch.  We plot |lambda| on a log axis and
%      ENCODE SIGN BY COLOR (warm = lambda>0, cool = lambda<0).
%
% We plot the smallest 500 and largest 500 eigenvalues (via eigs, never full eig),
% so the matrix is sized to n in [5000,15000] DOFs (h0=0.05 => n ~ 12.7k).
%
% Outputs (PDF, vector) -> output/spectrum_spy_paper/ :
%   spy.pdf                            sparsity + sign pattern of K
%   eig_spectrum.pdf                   smallest 500, raw K vs preconditioned (K,P)
%   eig_spectrum_large.pdf             largest  500, raw K vs preconditioned (K,P)
%   eig_spectrum_mesh_refine.pdf       smallest 500, raw K across an h0 sweep
%   eig_spectrum_large_mesh_refine.pdf largest  500, raw + preconditioned across h0

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
addpath(repoRoot);
addpath(thisFileDir);
import src.discretization.*
import src.stokes.*
rng(1);

outDir = fullfile(thisFileDir, 'output', 'spectrum_spy_paper');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ---- parameters ----------------------------------------------------------
nu     = 1.0;
dt     = 0.02;
Tmax   = 1.2;
t_snap = 0.5 * Tmax;          % bar_rotating snapshot at mid-time
k      = 500;                 % # smallest / largest eigenvalues to plot

% Empirically N ~ 4.85/h0^2 for this channel (h0=0.05 -> N=1940, n=3N+nC=5840),
% so to keep n=3N+nC in [5000,15000]:
h0_base   = 0.04;                          % baseline mesh:  n ~ 9.1k
h0_refine = [0.05 0.044 0.039 0.034];      % sweep: n ~ {5.8k,7.6k,9.6k,12.6k}

% ---- palette (sign-coded; reused from run_subspace_capture styling) ------
POS      = [0.85 0.40 0.32];  NEG      = [0.20 0.45 0.70];   % warm +, cool -
POS_lt   = [0.97 0.78 0.74];  POS_dk   = [0.55 0.10 0.08];   % + family light->dark
NEG_lt   = [0.74 0.84 0.95];  NEG_dk   = [0.06 0.22 0.50];   % - family light->dark

%% ===================== 1. Baseline snapshot ==============================
fprintf('[spectrum_spy] assembling baseline KKT (h0=%.3f, bar_rotating t=%.3f)...\n', ...
        h0_base, t_snap);
S = assemble_kkt_snapshot(h0_base, t_snap, nu, dt, Tmax);
fprintf('  n=%d  (nU=%d, nP=%d, nC=%d)  nnz=%d\n', S.n, S.nU, S.nP, S.nC, nnz(S.K));

symres = norm(S.K - S.K', 'fro') / max(norm(S.K, 'fro'), eps);
fprintf('  symmetry residual ||K-K''||_F/||K||_F = %.2e\n', symres);
assert(symres < 1e-12, 'KKT lost symmetry during elimination (sym_res=%.2e)', symres);
assert(S.n >= 5000 && S.n <= 15000, ...
       'baseline n=%d outside target [5000,15000] — adjust h0_base', S.n);
assert(k < S.n, 'k=%d not smaller than n=%d', k, S.n);

%% ===================== 2. Baseline spectra (eigs) =========================
fprintf('[spectrum_spy] computing smallest/largest %d eigenvalues (raw + precond)...\n', k);
las = real(safe_eigs(S.K,       k, 'smallestabs'));   % raw, near zero
lal = real(safe_eigs(S.K,       k, 'largestabs'));    % raw, extremes
lps = real(safe_eigs(S.K, S.P,  k, 'smallestabs'));   % preconditioned, near zero
lpl = real(safe_eigs(S.K, S.P,  k, 'largestabs'));    % preconditioned, extremes

% Indefinite check spans BOTH sets: the negative eigenvalues are the small-
% magnitude ones (-eps*L / multiplier cluster -> captured by 'smallestabs');
% the large-magnitude ones are positive (SPD velocity block -> 'largestabs').
lam_all = [las; lal];
fprintf('  raw  : lambda_min=%.3e < 0 < lambda_max=%.3e   min|lambda|=%.3e\n', ...
        min(lam_all), max(lam_all), min(abs(las)));
fprintf('  raw  smallest-|.| set: #neg=%d  #pos=%d   largest-|.| set: #neg=%d  #pos=%d\n', ...
        sum(las < 0), sum(las > 0), sum(lal < 0), sum(lal > 0));
fprintf('  prec : lambda in [%.3e, %.3e]   min|lambda|=%.3e   (clustered)\n', ...
        min([lps; lpl]), max([lps; lpl]), min(abs(lps)));
assert(min(lam_all) < 0 && max(lam_all) > 0, 'KKT is not indefinite — check assembly');

%% ===================== 3. Figures: spy + baseline spectra =================
plot_spy(S, POS, NEG, fullfile(outDir, 'spy.pdf'));
fprintf('  saved %s\n', fullfile(outDir, 'spy.pdf'));

plot_spectrum_overlay(las, lps, S, k, 'smallest', POS, NEG, ...
                      fullfile(outDir, 'eig_spectrum.pdf'));
fprintf('  saved %s\n', fullfile(outDir, 'eig_spectrum.pdf'));

plot_spectrum_overlay(lal, lpl, S, k, 'largest', POS, NEG, ...
                      fullfile(outDir, 'eig_spectrum_large.pdf'));
fprintf('  saved %s\n', fullfile(outDir, 'eig_spectrum_large.pdf'));

%% ===================== 4. Mesh-refinement sweep ==========================
nH = numel(h0_refine);
sweep = struct('h0', cell(nH,1), 'n', [], 'las', [], 'lal', [], 'lpl', []);
for ih = 1:nH
    h0r = h0_refine(ih);
    fprintf('[spectrum_spy] sweep %d/%d  h0=%.3f ...\n', ih, nH, h0r);
    Sr = assemble_kkt_snapshot(h0r, t_snap, nu, dt, Tmax);
    assert(Sr.n >= 5000 && Sr.n <= 15000, ...
           'sweep h0=%.3f gives n=%d outside [5000,15000]', h0r, Sr.n);
    fprintf('  n=%d  (nC=%d)\n', Sr.n, Sr.nC);
    sweep(ih).h0  = h0r;
    sweep(ih).n   = Sr.n;
    sweep(ih).las = real(safe_eigs(Sr.K,      k, 'smallestabs'));
    sweep(ih).lal = real(safe_eigs(Sr.K,      k, 'largestabs'));
    sweep(ih).lpl = real(safe_eigs(Sr.K, Sr.P, k, 'largestabs'));
end

plot_meshrefine({sweep.las}, [], [sweep.h0], [sweep.n], k, 'smallest', ...
                POS_lt, POS_dk, NEG_lt, NEG_dk, ...
                fullfile(outDir, 'eig_spectrum_mesh_refine.pdf'));
fprintf('  saved %s\n', fullfile(outDir, 'eig_spectrum_mesh_refine.pdf'));

plot_meshrefine({sweep.lal}, {sweep.lpl}, [sweep.h0], [sweep.n], k, 'largest', ...
                POS_lt, POS_dk, NEG_lt, NEG_dk, ...
                fullfile(outDir, 'eig_spectrum_large_mesh_refine.pdf'));
fprintf('  saved %s\n', fullfile(outDir, 'eig_spectrum_large_mesh_refine.pdf'));

fprintf('\n[spectrum_spy] done. 5 PDFs in %s\n', outDir);

%==========================================================================
%  Local functions
%==========================================================================
function S = assemble_kkt_snapshot(h0, t_snap, nu, dt, Tmax)
%ASSEMBLE_KKT_SNAPSHOT  Standalone symmetric-indefinite KKT K and its SPD
% block-diagonal preconditioner P at the bar_rotating snapshot t_snap.
    import src.discretization.*
    import src.stokes.*

    x1 = 0; x2 = 4; y1 = 0; y2 = 1; Lyc = y2 - y1; Uin = 1.0;

    msh  = build_channel_mesh_pde(h0, x1, x2, y1, y2, {'rect_right'});
    N    = msh.N;  nU = 2*N;  nP = N;
    blk  = assemble_stokes_blocks(msh);
    Avel = blk.M2/dt + nu*blk.A2;  Avel = (Avel + Avel')/2;   % SPD velocity block
    eps_stab = h0^2 / (12*nu);

    % --- bar_rotating coupling at t_snap ---
    geo = struct('x1',x1,'x2',x2,'y1',y1,'y2',y2, ...
                 'xc',(x1+x2)/2,'yc',(y1+y2)/2,'h0',h0,'Tmax',Tmax);
    cases = define_motion_list(dt);
    sidx  = find(cellfun(@(c) strcmp(c.name,'bar_rotating'), cases), 1);
    mcase = cases{sidx}.factory(geo);
    mot   = mcase.motion_fun(t_snap);
    TR    = triangulation(msh.t, msh.p);
    [C, ~, nC] = assemble_coupling(TR, N, mot.X, mot.V);

    % --- full symmetric indefinite KKT ---
    Z = @(a,b) sparse(a,b);
    K = [ Avel ,  blk.B'       ,  C'        ; ...
          blk.B, -eps_stab*blk.L,  Z(nP,nC) ; ...
          C    ,  Z(nC,nP)     ,  Z(nC,nC)  ];
    b = zeros(size(K,1), 1);                     % rhs unused (spectrum only)

    % --- velocity Dirichlet (parabolic inflow + no-slip walls), pressure pin ---
    left  = find(msh.rect_left);
    walls = unique([find(msh.rect_top); find(msh.rect_bottom)]);
    bnodes = unique([left; walls]);
    yv  = msh.p(bnodes, 2);
    uxv = zeros(numel(bnodes), 1);
    isl = ismember(bnodes, left);
    uxv(isl) = Uin * 4 .* yv(isl) .* (Lyc - yv(isl)) / Lyc^2;
    veldofs = [bnodes; N + bnodes];
    velvals = [uxv; zeros(numel(bnodes), 1)];
    [K, b] = apply_dirichlet_sym(K, b, veldofs, velvals);
    [~, pin_node] = max(msh.p(:, 1));            % outflow corner
    [K, b] = apply_dirichlet_sym(K, b, nU + pin_node, 0);   %#ok<ASGLU>

    % --- SPD block-diagonal preconditioner P = blkdiag(Avel_bc, Dp/nu, I) ---
    % Same homogeneous Dirichlet identity rows as K so (K,P) is a consistent
    % symmetric-definite pair (block_precond, solve_stokes_immersed.m:230-244).
    Au = Avel;
    Au(veldofs, :) = 0;  Au(:, veldofs) = 0;
    Au(veldofs, veldofs) = speye(numel(veldofs));
    Au = (Au + Au')/2;
    Pp = (blk.Dp + blk.Dp')/(2*nu);              % Dp/nu, symmetric SPD
    Pp(pin_node, :) = 0;  Pp(:, pin_node) = 0;  Pp(pin_node, pin_node) = 1;
    P = blkdiag(Au, Pp, speye(nC));
    P = (P + P')/2;

    S.K = K;  S.P = P;  S.n = size(K,1);
    S.nU = nU;  S.nP = nP;  S.nC = nC;  S.N = N;  S.h0 = h0;
end

% -------------------------------------------------------------------------
function d = safe_eigs(varargin)
%SAFE_EIGS  eigs(...) for the smallest/largest k, returning the eigenvalue
% vector.  Standard (A,k,which) or generalized (A,B,k,which) call forms.
    if issparse(varargin{2})        % generalized: (A, B, k, which)
        A = varargin{1}; B = varargin{2}; k = varargin{3}; which = varargin{4};
        [~, D] = eigs(A, B, k, which, 'Tolerance', 1e-5, 'MaxIterations', 600);
    else                            % standard: (A, k, which)
        A = varargin{1}; k = varargin{2}; which = varargin{3};
        [~, D] = eigs(A, k, which, 'Tolerance', 1e-5, 'MaxIterations', 600);
    end
    d = diag(D);
end

% -------------------------------------------------------------------------
function plot_spy(S, POS, NEG, figPath)
%PLOT_SPY  Sparsity pattern of K with block-partition lines + sign pattern.
    K = S.K;  nU = S.nU;  nP = S.nP;
    b1 = nU + 0.5;  b2 = nU + nP + 0.5;

    f  = figure('Visible','off','Color','w','Position',[50 50 1200 560]);
    tl = tiledlayout(f, 1, 2, 'Padding','compact','TileSpacing','compact');

    nexttile(tl);
    spy(K);  hold on;
    xline(b1,'-','Color',[0.85 0.2 0.2]); xline(b2,'-','Color',[0.85 0.2 0.2]);
    yline(b1,'-','Color',[0.85 0.2 0.2]); yline(b2,'-','Color',[0.85 0.2 0.2]);
    title(sprintf('K   (n=%d, nnz=%d)', size(K,1), nnz(K)), 'Interpreter','none');
    xlabel(sprintf('blocks: nU=%d  nP=%d  nC=%d', nU, nP, S.nC));

    nexttile(tl);  hold on;
    [ip, jp] = find(K > 0);
    [in, jn] = find(K < 0);
    plot(jp, ip, '.', 'Color', POS, 'MarkerSize', 2);
    plot(jn, in, '.', 'Color', NEG, 'MarkerSize', 2);
    xline(b1,'-','Color',[0.6 0.6 0.6]); xline(b2,'-','Color',[0.6 0.6 0.6]);
    yline(b1,'-','Color',[0.6 0.6 0.6]); yline(b2,'-','Color',[0.6 0.6 0.6]);
    set(gca, 'YDir','reverse');  axis equal tight;  box on;
    xlim([0 size(K,1)]);  ylim([0 size(K,1)]);
    title('sign pattern (warm = +,  cool = -)');

    sgtitle(tl, sprintf('stokes\\_immersed\\_rotor KKT  h0=%.4g  (symmetric indefinite)', ...
            S.h0), 'FontWeight','bold');
    % rasterize: ~131k nonzeros as vector markers would be ~8 MB
    exportgraphics(f, figPath, 'ContentType','image', 'Resolution', 200);
    exportgraphics(f, strrep(figPath, '.pdf', '.png'), 'Resolution', 150);
    close(f);
end

% -------------------------------------------------------------------------
function plot_spectrum_overlay(lam_raw, lam_pre, S, k, mode, POS, NEG, figPath)
%PLOT_SPECTRUM_OVERLAY  |lambda| semilogy, sign-colored; raw K (solid) vs
% preconditioned (K,P) (dashed).
    f  = figure('Visible','off','Color','w','Position',[50 50 1100 700]);
    ax = axes(f);  hold(ax, 'on');  grid(ax, 'on');

    sign_semilogy(ax, lam_raw, POS, NEG, '-',  1.8);
    sign_semilogy(ax, lam_pre, POS, NEG, '--', 1.4);

    set(ax, 'YScale','log', 'FontSize', 12);
    xlabel(ax, sprintf('sorted index i (1 \\rightarrow %d)', k), 'FontSize', 12);
    ylabel(ax, '|\lambda_i|   (log scale)', 'FontSize', 12);
    title(ax, sprintf('stokes\\_immersed\\_rotor  h0=%.4g  n=%d : %s %d eigenvalues', ...
          S.h0, S.n, mode, k), 'FontSize', 13);
    subtitle(ax, sprintf(['warm = \\lambda>0,  cool = \\lambda<0   |   ', ...
          'raw \\lambda\\in[%.2e, %.2e]   |   precond \\lambda\\in[%.2e, %.2e]'], ...
          min(lam_raw), max(lam_raw), min(lam_pre), max(lam_pre)));

    % neutral proxy lines: style conveys raw vs preconditioned (color = sign)
    hL1 = plot(ax, NaN, NaN, '-',  'Color', [0.30 0.30 0.30], 'LineWidth', 1.8);
    hL2 = plot(ax, NaN, NaN, '--', 'Color', [0.30 0.30 0.30], 'LineWidth', 1.4);
    legend(ax, [hL1, hL2], {'K  (raw)', 'P^{-1}K  (preconditioned)'}, ...
           'Location','best', 'FontSize', 12, 'Box','on');
    exportgraphics(f, figPath, 'ContentType','vector');
    exportgraphics(f, strrep(figPath, '.pdf', '.png'), 'Resolution', 150);
    close(f);
end

% -------------------------------------------------------------------------
function plot_meshrefine(raw_cell, pre_cell, h0_list, n_list, k, mode, ...
                         POS_lt, POS_dk, NEG_lt, NEG_dk, figPath)
%PLOT_MESHREFINE  |lambda| semilogy across an h0 sweep, light->dark gradient
% (darker = finer).  Positive eigs warm family, negative eigs cool family.
% If pre_cell is non-empty, overlays the preconditioned spectra as dashed.
    [h0_sorted, ord] = sort(h0_list, 'descend');   % coarse -> fine
    n_sorted = n_list(ord);
    raw_cell = raw_cell(ord);
    has_pre  = ~isempty(pre_cell);
    if has_pre, pre_cell = pre_cell(ord); end
    nH = numel(h0_sorted);

    f  = figure('Visible','off','Color','w','Position',[50 50 1200 760]);
    ax = axes(f);  hold(ax, 'on');  grid(ax, 'on');
    legH = [];  legL = {};

    for ih = 1:nH
        t  = (nH == 1) * 1 + (nH > 1) * (ih - 1) / max(nH - 1, 1);
        cP = (1 - t) * POS_lt + t * POS_dk;
        cN = (1 - t) * NEG_lt + t * NEG_dk;
        hp = sign_semilogy(ax, raw_cell{ih}, cP, cN, '-', 1.6);
        legH(end+1) = hp;                                          %#ok<AGROW>
        legL{end+1} = sprintf('K  h0=%.3g, n=%d', h0_sorted(ih), n_sorted(ih)); %#ok<AGROW>
        if has_pre
            hpp = sign_semilogy(ax, pre_cell{ih}, cP, cN, '--', 1.3);
            legH(end+1) = hpp;                                     %#ok<AGROW>
            legL{end+1} = sprintf('P^{-1}K  h0=%.3g', h0_sorted(ih)); %#ok<AGROW>
        end
    end

    set(ax, 'YScale','log', 'FontSize', 12);
    xlabel(ax, sprintf('sorted index i (1 \\rightarrow %d)', k), 'FontSize', 12);
    ylabel(ax, '|\lambda_i|   (log scale)', 'FontSize', 12);
    title(ax, sprintf('stokes\\_immersed\\_rotor : %s %d eigenvalues, mesh refinement', ...
          mode, k), 'FontSize', 13);
    subtitle(ax, sprintf(['warm = \\lambda>0, cool = \\lambda<0;  darker = finer mesh', ...
          '   (h0 \\in \\{%s\\})'], ...
          strjoin(arrayfun(@(x) sprintf('%.3g', x), h0_sorted, ...
                  'UniformOutput', false), ', ')));
    legend(ax, legH, legL, 'Location','best', 'FontSize', 10, 'Box','on', ...
           'NumColumns', 1 + has_pre);
    exportgraphics(f, figPath, 'ContentType','vector');
    exportgraphics(f, strrep(figPath, '.pdf', '.png'), 'Resolution', 150);
    close(f);
end

% -------------------------------------------------------------------------
function [hrep, hpos, hneg] = sign_semilogy(ax, lam, posColor, negColor, style, lw)
%SIGN_SEMILOGY  Sort lambda ascending, plot |lambda| vs index; negatives in
% negColor, positives in posColor.  Sorted ascending => negatives (descending
% |.|) then positives (ascending |.|): a "V" whose floor is min|lambda|.
% HREP is a guaranteed-valid handle (whichever sign is present) for legends.
    lam = sort(lam(:), 'ascend');
    av  = abs(lam);
    neg = lam < 0;  pos = ~neg;
    idx = (1:numel(lam))';
    hneg = gobjects(1);  hpos = gobjects(1);                 % placeholders
    if any(neg)
        hneg = semilogy(ax, idx(neg), av(neg), style, 'Color', negColor, 'LineWidth', lw);
    end
    if any(pos)
        hpos = semilogy(ax, idx(pos), av(pos), style, 'Color', posColor, 'LineWidth', lw);
    end
    if isgraphics(hpos), hrep = hpos; else, hrep = hneg; end
end
