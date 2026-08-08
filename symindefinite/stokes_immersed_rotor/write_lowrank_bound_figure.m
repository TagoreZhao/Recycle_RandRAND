function wrote = write_lowrank_bound_figure(run_dir, stats, opts)
%WRITE_LOWRANK_BOUND_FIGURE  GMRES iterations against the low-rank k+1 bound.
%
%   WROTE = WRITE_LOWRANK_BOUND_FIGURE(RUN_DIR, STATS, OPTS)
%   Writes <RUN_DIR>/lowrank_bound.png and returns whether it did.
%
%   The one figure in the benchmark that tests a THEOREM rather than comparing
%   preconditioners.  K_n - K_1 is symmetric of rank 2*rank(dC) <= 2*nC, so
%   left-preconditioning K_n with the exact SIGNED inverse of K_1 gives
%   I + (rank-r update) and unrestarted GMRES must terminate in <= r+1
%   iterations (gmres_exact_inv_frozen; see define_solver_list>gmres_frozen_solve).
%   Plotted against 2*nC(n)+1 -- the bound with rank(dC) at its maximum nC, which
%   is what the run can state per step without a rank() of a dense n-by-n
%   difference.  Curves at or under the dashed line confirm the claim; curves
%   BELOW it mean dC is rank deficient at that step, not that the bound is wrong.
%
%   TWO PANELS, because one axis cannot carry both questions:
%     top    LINEAR, framed on the bound.  Every other iteration figure here is
%            semilogy, which is right across three decades but hides exactly the
%            few-iteration overshoot that would falsify this claim.
%     bottom LOG, adding exact_ldl_frozen -- the SAME frozen factor, SPD-ified
%            (M = |K_1|) so MINRES can use it at all.  The gap between the two is
%            the price of MINRES's SPD requirement, and it is a decade-scale gap,
%            which is why it cannot share the linear axis above.
%
%   Returns false without drawing when STATS has no GMRES arm or no nC column, so
%   results directories written before this arm existed still replot cleanly.
%
%   See also: write_case_figures, define_solver_list.

    if nargin < 3 || isempty(opts), opts = benchmark_fig_defaults(); end

    GKEY = 'gmres_exact_inv_frozen';
    MKEY = 'exact_ldl_frozen';
    GCOL = [0.84 0.37 0.00];
    MCOL = [0.00 0.45 0.70];

    wrote = false;
    if ~isfield(stats, 'solver_its') || ~isfield(stats.solver_its, GKEY) ...
            || ~isfield(stats, 'nC') || isempty(stats.nC)
        return
    end
    if ~exist(run_dir, 'dir'), mkdir(run_dir); end

    its   = stats.solver_its.(GKEY)(:);
    nC    = stats.nC(:);
    ns    = min(numel(its), numel(nC));
    its   = its(1:ns);
    bound = 2 * nC(1:ns) + 1;
    tax   = (1:ns)' * stats.dt;

    slack  = bound - its;               % >= 0 everywhere <=> claim held
    n_over = sum(slack < 0);

    mits = [];
    if isfield(stats.solver_its, MKEY)
        mits = stats.solver_its.(MKEY)(:);
        mits = mits(1:ns);
    end
    has_m = ~isempty(mits);

    fh = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', ...
                'Position', [1 1 opts.multi_width, opts.multi_height]);
    tl = tiledlayout(fh, 2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

    % --- top: the claim, on a linear axis framed by the bound ---------------
    ax1 = nexttile(tl);
    hold(ax1, 'on');
    plot(ax1, tax, bound, '--', 'Color', [0 0 0], 'LineWidth', 1.6);
    plot(ax1, tax, its, '-o', 'Color', GCOL, 'LineWidth', 1.7, ...
         'MarkerSize', 4, 'MarkerFaceColor', GCOL);
    hold(ax1, 'off');
    grid(ax1, 'on');
    ylim(ax1, [0, max(max(bound), max(its)) * 1.15 + 1]);
    xlim(ax1, [tax(1) tax(end)]);
    ylabel(ax1, 'iterations (linear)');
    ax1.XTickLabel = [];

    % --- bottom: the same curves against the SPD-ified MINRES arm -----------
    ax2 = nexttile(tl);
    hold(ax2, 'on');
    h = gobjects(0); lab = {};
    h(end+1) = plot(ax2, tax, bound, '--', 'Color', [0 0 0], 'LineWidth', 1.4);
    lab{end+1} = 'bound 2n_C + 1';
    h(end+1) = plot(ax2, tax, max(its, 1), '-', 'Color', GCOL, 'LineWidth', 1.7);
    lab{end+1} = 'GMRES, exact K_1^{-1}';
    if has_m
        h(end+1) = plot(ax2, tax, max(mits, 1), '-', 'Color', MCOL, 'LineWidth', 1.4);
        lab{end+1} = 'MINRES, |K_1| (same factor, SPD-ified)';
    end
    hold(ax2, 'off');
    set(ax2, 'YScale', 'log');
    grid(ax2, 'on');
    xlim(ax2, [tax(1) tax(end)]);
    xlabel(ax2, 't');
    ylabel(ax2, 'iterations (log)');

    % One legend, in the layout's south TILE.  Not 'Location','best' and not
    % 'southoutside': inside a tiledlayout those reserve no layout space, and
    % exportgraphics then drops the legend from the PNG entirely -- the figure
    % looks fine on screen and ships without a key.  Layout.Tile is the only
    % placement that survives the export (same rule as place_solver_legend).
    % Both panels draw the same series, so the bottom panel's handles cover both.
    place_solver_legend(tl, h, lab, opts);

    title(tl,sprintf('%s  |  low-rank finite termination', stats.case_name), ...
          'Interpreter', 'none', 'FontWeight', 'bold', ...
          'FontSize', opts.titlefontsize);
    subtitle(tl, verdict_text(its, bound, slack, n_over, ns, mits), ...
             'Interpreter', 'none', 'FontSize', opts.subtitlefontsize);

    save_benchmark_figure(fh, fullfile(run_dir, 'lowrank_bound.png'), opts);
    wrote = true;
end

%==========================================================================
function s = verdict_text(its, bound, slack, n_over, ns, mits)
%VERDICT_TEXT  Two short lines stating the outcome, kept narrow enough not to
% run off the figure edge (the reason this is not one long sentence).
    if n_over == 0
        s = {sprintf('claim HOLDS at all %d steps: max %d its vs min bound %d (slack %d)', ...
                     ns, max(its), min(bound), min(slack))};
    else
        s = {sprintf('claim VIOLATED at %d of %d steps (worst overshoot %d its)', ...
                     n_over, ns, -min(slack))};
    end
    if ~isempty(mits)
        s{end+1} = sprintf('same factor SPD-ified for MINRES: %d-%d its', ...
                           min(mits), max(mits));
    end
end
