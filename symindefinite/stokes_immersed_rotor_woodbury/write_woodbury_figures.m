function files = write_woodbury_figures(outDir, opts)
%WRITE_WOODBURY_FIGURES  Every figure for the Woodbury benchmark, from the CSVs.
%   FILES = WRITE_WOODBURY_FIGURES(OUTDIR)
%   FILES = WRITE_WOODBURY_FIGURES(OUTDIR, OPTS)   % woodbury_fig_defaults overrides
%
%   Reads ONLY <case>_results.csv and woodbury_summary.csv from OUTDIR -- never an
%   Astat -- so figures can be regenerated without re-solving anything.  That is
%   what replot_woodbury is.
%
%   Per case:
%     <case>_accuracy.png     forward error vs timestep, all arms (the headline)
%     <case>_residual.png     true relative residual vs timestep, all arms
%     <case>_capacitance.png  cond(Cap), and sigma_min/sigma_max separately
%     <case>_drift.png        how far the operator has moved from K_1
%     <case>_timing.png       per-step and cumulative wall clock, with break-even
%   Across cases:
%     all_cases_comparison.png
%
%   ZEROS ON A LOG AXIS.  The `fresh` arm's error is sometimes exactly 0 (it
%   solves the same system as the ground truth), which a log axis would silently
%   drop.  Such points are drawn at ERR_FLOOR = 1e-17, just below double
%   precision, and the axis label says so -- a dropped point would read as a gap
%   in the data rather than as "exact".
%
%   See also: write_woodbury_outputs, replot_woodbury, save_woodbury_figure.

    if nargin < 2, opts = struct(); end
    fo = woodbury_fig_defaults(opts);

    ERR_FLOOR = 1e-17;

    sumFile = fullfile(outDir, 'woodbury_summary.csv');
    if ~exist(sumFile, 'file')
        error('write_woodbury_figures:noSummary', ...
              ['No woodbury_summary.csv in %s -- run the benchmark first ' ...
               '(figures are generated from the CSVs, not from memory).'], outDir);
    end
    Tsum  = readtable(sumFile, 'TextType', 'string');
    cases = cellstr(Tsum.case_name);

    files = {};
    Tall  = cell(numel(cases), 1);

    for i = 1:numel(cases)
        cname = cases{i};
        f = fullfile(outDir, sprintf('%s_results.csv', cname));
        if ~exist(f, 'file')
            warning('write_woodbury_figures:missingCase', ...
                    'No results CSV for case "%s"; skipping its figures.', cname);
            continue;
        end
        T       = readtable(f);
        Tall{i} = T;
        [arms, labels] = local_arms(T);
        sty  = woodbury_style_table(numel(arms));
        step = T.timestep;

        % --- accuracy (the headline) --------------------------------------
        Y = local_arm_matrix(T, arms, '_err', ERR_FLOOR);
        fh = local_curve_fig(step, Y, labels, sty, fo, ...
                sprintf('%s: forward error vs timestep', cname), ...
                sprintf('relative error  ||x - x_{ref}|| / ||x_{ref}||   (0 drawn at %g)', ...
                        ERR_FLOOR));
        files{end+1} = save_woodbury_figure(fh, ...
            fullfile(outDir, sprintf('%s_accuracy.png', cname)), fo);   %#ok<AGROW>

        % --- residual -----------------------------------------------------
        Y = local_arm_matrix(T, arms, '_relres', ERR_FLOOR);
        fh = local_curve_fig(step, Y, labels, sty, fo, ...
                sprintf('%s: true relative residual vs timestep', cname), ...
                sprintf('||K_n x - b|| / ||b||   (0 drawn at %g)', ERR_FLOOR));
        files{end+1} = save_woodbury_figure(fh, ...
            fullfile(outDir, sprintf('%s_residual.png', cname)), fo);   %#ok<AGROW>

        % --- capacitance conditioning -------------------------------------
        files{end+1} = local_capacitance_fig(T, cname, outDir, fo);     %#ok<AGROW>

        % --- operator drift -----------------------------------------------
        files{end+1} = local_drift_fig(T, cname, outDir, fo);           %#ok<AGROW>

        % --- timing -------------------------------------------------------
        be = Tsum.break_even_step(i);
        files{end+1} = local_timing_fig(T, cname, outDir, fo, ...
                                        Tsum.t_setup(i), be);           %#ok<AGROW>
    end

    files{end+1} = local_comparison_fig(Tall, cases, outDir, fo, ERR_FLOOR);
    files = files(~cellfun(@isempty, files));
end

%==========================================================================
function [arms, labels] = local_arms(T)
%LOCAL_ARMS  Arms present in the table, in reporting order.
%   Order is fixed here rather than discovered, so colours stay stable across
%   cases and across reruns; anything in the CSV but not listed is appended.
    known = { ...
        'woodbury', 'Woodbury update on frozen A_1^{-1} (rank-2n_C) [METHOD]'; ...
        'frozen',   'Frozen A_1^{-1}, no correction [CONTROL]'; ...
        'fresh',    'Fresh LDL of K_n every step [REFERENCE]'};

    vn   = string(T.Properties.VariableNames);
    have = extractBefore(vn(endsWith(vn, "_err")), strlength(vn(endsWith(vn, "_err"))) - 3);

    arms = {};  labels = {};
    for i = 1:size(known, 1)
        if any(have == known{i, 1})
            arms{end+1}   = known{i, 1};                        %#ok<AGROW>
            labels{end+1} = known{i, 2};                        %#ok<AGROW>
        end
    end
    for h = have(:)'
        if ~any(strcmp(arms, h))
            arms{end+1}   = char(h);                            %#ok<AGROW>
            labels{end+1} = char(h);                            %#ok<AGROW>
        end
    end
end

%==========================================================================
function Y = local_arm_matrix(T, arms, suffix, floorval)
    Y = nan(height(T), numel(arms));
    for j = 1:numel(arms)
        v = T.(sprintf('%s%s', arms{j}, suffix));
        v(v < floorval) = floorval;         % exact zeros survive the log axis
        Y(:, j) = v;
    end
end

%==========================================================================
function fh = local_curve_fig(x, Y, labels, sty, fo, ttl, ylab)
%LOCAL_CURVE_FIG  One semilogy axis, all arms, legend below.
    fh = figure('Units', 'inches', ...
                'Position', [1 1 fo.multi_width fo.multi_height], ...
                'Visible', 'off');
    ax = axes(fh);
    hold(ax, 'on');
    h = gobjects(size(Y, 2), 1);
    for j = 1:size(Y, 2)
        h(j) = semilogy(ax, x, Y(:, j), ...
            'Color', sty(j).color, 'LineStyle', sty(j).linestyle, ...
            'LineWidth', sty(j).linewidth, 'Marker', sty(j).marker, ...
            'MarkerSize', sty(j).markersize, ...
            'MarkerIndices', local_marker_idx(numel(x), j, size(Y, 2), ...
                                              fo.marker_targets));
    end
    set(ax, 'YScale', 'log');
    hold(ax, 'off');
    xlabel(ax, 'timestep n');
    ylabel(ax, ylab);
    % Interpreter 'none': case names contain underscores, which TeX would render
    % as subscripts ("bar_rotating" -> "bar" with a subscript r).
    title(ax, ttl, 'FontSize', fo.titlefontsize, 'Interpreter', 'none');
    legend(h, labels, 'Location', 'southoutside', ...
           'NumColumns', min(fo.legend_columns, numel(labels)), ...
           'FontSize', fo.legendfontsize, 'Interpreter', 'tex');
end

%==========================================================================
function idx = local_marker_idx(npts, j, narms, target)
%LOCAL_MARKER_IDX  Staggered marker positions so coincident curves stay readable.
    if npts <= target
        idx = 1:npts;
        return;
    end
    stride = max(1, round(npts / target));
    off    = mod(j - 1, max(1, min(stride, narms)));
    idx    = (1 + off):stride:npts;
    if isempty(idx), idx = 1; end
end

%==========================================================================
function f = local_capacitance_fig(T, cname, outDir, fo)
%LOCAL_CAPACITANCE_FIG  cond(Cap), plus its two ends separately.
%   The extremes are shown on their own axis because they answer different
%   questions: sigma_min is how close the update is to breaking down, sigma_max
%   is how strongly the rank-2nC term acts.  A ratio alone hides both.
    fh = figure('Units', 'inches', ...
                'Position', [1 1 fo.multi_width fo.multi_height], ...
                'Visible', 'off');
    tl = tiledlayout(fh, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile(tl);
    semilogy(ax1, T.timestep, T.cap_cond, '-o', 'LineWidth', 1.8, ...
             'Color', [0.00 0.45 0.70], 'MarkerSize', 4);
    ylabel(ax1, '\kappa(Cap) = \sigma_{max}/\sigma_{min}');
    title(ax1, sprintf('%s: capacitance conditioning', cname), ...
          'FontSize', fo.titlefontsize, 'Interpreter', 'none');

    ax2 = nexttile(tl);
    semilogy(ax2, T.timestep, T.cap_smax, '-s', 'LineWidth', 1.6, ...
             'Color', [0.84 0.37 0.00], 'MarkerSize', 4);
    hold(ax2, 'on');
    semilogy(ax2, T.timestep, T.cap_smin, '-^', 'LineWidth', 1.6, ...
             'Color', [0.00 0.62 0.45], 'MarkerSize', 4);
    hold(ax2, 'off');
    xlabel(ax2, 'timestep n');
    ylabel(ax2, 'singular values of Cap');
    legend(ax2, {'\sigma_{max}(Cap)', '\sigma_{min}(Cap)'}, ...
           'Location', 'best', 'FontSize', fo.legendfontsize);

    f = save_woodbury_figure(fh, ...
        fullfile(outDir, sprintf('%s_capacitance.png', cname)), fo);
end

%==========================================================================
function f = local_drift_fig(T, cname, outDir, fo)
%LOCAL_DRIFT_FIG  How far the operator has moved from the frozen reference.
    fh = figure('Units', 'inches', ...
                'Position', [1 1 fo.multi_width fo.multi_height], ...
                'Visible', 'off');
    ax = axes(fh);
    hold(ax, 'on');
    plot(ax, T.timestep, T.dC_rel, '-o', 'LineWidth', 2.0, ...
         'Color', [0.00 0.00 0.00], 'MarkerSize', 4);
    plot(ax, T.timestep, T.coupling_change, '--s', 'LineWidth', 1.6, ...
         'Color', [0.84 0.37 0.00], 'MarkerSize', 4);
    plot(ax, T.timestep, T.correction_rel, '-.^', 'LineWidth', 1.6, ...
         'Color', [0.00 0.45 0.70], 'MarkerSize', 4);
    hold(ax, 'off');
    xlabel(ax, 'timestep n');
    ylabel(ax, 'relative magnitude');
    title(ax, sprintf('%s: operator drift from the frozen reference', cname), ...
          'FontSize', fo.titlefontsize, 'Interpreter', 'none');
    legend(ax, { ...
        '||dC||_F / ||Cblk_1||_F  (distance from K_1)', ...
        '||C_n - C_{n-1}||_F / ||C_{n-1}||_F  (per-step)', ...
        '||x_{wood} - K_1^{-1}b|| / ||x_{wood}||  (size of the correction)'}, ...
        'Location', 'southoutside', 'NumColumns', 1, ...
        'FontSize', fo.legendfontsize, 'Interpreter', 'tex');

    f = save_woodbury_figure(fh, ...
        fullfile(outDir, sprintf('%s_drift.png', cname)), fo);
end

%==========================================================================
function f = local_timing_fig(T, cname, outDir, fo, t_setup, break_even)
%LOCAL_TIMING_FIG  Per-step cost, and the cumulative cost that decides the question.
%   The cumulative panel is the one that matters: the method pays for one
%   factorization up front, so a per-step win only becomes a real win after
%   enough steps to amortize it.
    fh = figure('Units', 'inches', ...
                'Position', [1 1 fo.multi_width fo.multi_height], ...
                'Visible', 'off');
    tl = tiledlayout(fh, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile(tl);
    hold(ax1, 'on');
    plot(ax1, T.timestep, 1e3*T.t_woodbury_net, '-o', 'LineWidth', 2.0, ...
         'Color', [0.00 0.00 0.00], 'MarkerSize', 4);
    plot(ax1, T.timestep, 1e3*T.t_frozen, '--s', 'LineWidth', 1.6, ...
         'Color', [0.84 0.37 0.00], 'MarkerSize', 4);
    plot(ax1, T.timestep, 1e3*T.t_fresh, '-.^', 'LineWidth', 1.6, ...
         'Color', [0.00 0.45 0.70], 'MarkerSize', 4);
    hold(ax1, 'off');
    ylabel(ax1, 'per-step wall clock (ms)');
    title(ax1, sprintf('%s: cost (min over TIME_REPEATS repeats)', cname), ...
          'FontSize', fo.titlefontsize, 'Interpreter', 'none');
    legend(ax1, {'woodbury (net of diagnostics)', 'frozen apply', ...
                 'fresh (refactorize K_n each step)'}, 'Location', 'north', ...
           'NumColumns', 3, 'FontSize', fo.legendfontsize);

    ax2 = nexttile(tl);
    hold(ax2, 'on');
    plot(ax2, T.timestep, T.cum_woodbury, '-o', 'LineWidth', 2.0, ...
         'Color', [0.00 0.00 0.00], 'MarkerSize', 4);
    plot(ax2, T.timestep, T.cum_fresh, '-.^', 'LineWidth', 1.6, ...
         'Color', [0.00 0.45 0.70], 'MarkerSize', 4);
    if isfinite(break_even) && break_even >= 1
        xline(ax2, break_even, ':', ...
              sprintf('break-even n = %d', break_even), ...
              'LineWidth', 1.4, 'FontSize', fo.legendfontsize, ...
              'LabelVerticalAlignment', 'top', ...
              'LabelHorizontalAlignment', 'right', ...
              'LabelOrientation', 'horizontal');
    end
    hold(ax2, 'off');
    xlabel(ax2, 'timestep n');
    ylabel(ax2, 'cumulative wall clock (s)');
    legend(ax2, { ...
        sprintf('woodbury, incl. %.0f ms setup', 1e3*t_setup), ...
        'fresh, refactorize every step'}, ...
        'Location', 'northwest', 'FontSize', fo.legendfontsize);

    f = save_woodbury_figure(fh, ...
        fullfile(outDir, sprintf('%s_timing.png', cname)), fo);
end

%==========================================================================
function f = local_comparison_fig(Tall, cases, outDir, fo, ERR_FLOOR)
%LOCAL_COMPARISON_FIG  One accuracy panel per case, side by side.
    keep  = ~cellfun(@isempty, Tall);
    Tall  = Tall(keep);
    cases = cases(keep);
    if isempty(Tall), f = ''; return; end

    ncase = numel(Tall);
    W     = min(fo.max_width, fo.panel_width*ncase + fo.panel_margin);
    fh = figure('Units', 'inches', ...
                'Position', [1 1 W fo.multi_height], 'Visible', 'off');
    tl = tiledlayout(fh, 1, ncase, 'TileSpacing', 'compact', 'Padding', 'compact');

    % A SHARED y-range across panels.  Left to autoscale, disk_static (where every
    % arm sits at machine precision) gets a 1e-17..1e-14 axis and its rounding
    % jitter fills the panel, reading as instability instead of as "flat and
    % exact".  One range makes the three cases comparable at a glance.
    lo = ERR_FLOOR;  hi = ERR_FLOOR;
    for i = 1:ncase
        [arms_i, ~] = local_arms(Tall{i});
        Yi = local_arm_matrix(Tall{i}, arms_i, '_err', ERR_FLOOR);
        hi = max(hi, max(Yi(:), [], 'omitnan'));
    end
    ylims = [lo/2, max(hi*10, 1e-12)];

    hleg = [];  labs = {};
    for i = 1:ncase
        T = Tall{i};
        [arms, labels] = local_arms(T);
        sty = woodbury_style_table(numel(arms));
        Y   = local_arm_matrix(T, arms, '_err', ERR_FLOOR);

        ax = nexttile(tl);
        hold(ax, 'on');
        h = gobjects(numel(arms), 1);
        for j = 1:numel(arms)
            h(j) = semilogy(ax, T.timestep, Y(:, j), ...
                'Color', sty(j).color, 'LineStyle', sty(j).linestyle, ...
                'LineWidth', sty(j).linewidth, 'Marker', sty(j).marker, ...
                'MarkerSize', sty(j).markersize, ...
                'MarkerIndices', local_marker_idx(height(T), j, numel(arms), ...
                                                  fo.marker_targets));
        end
        set(ax, 'YScale', 'log');
        ylim(ax, ylims);
        hold(ax, 'off');
        xlabel(ax, 'timestep n');
        if i == 1
            ylabel(ax, sprintf('relative forward error  (0 drawn at %g)', ERR_FLOOR));
        end
        title(ax, cases{i}, 'FontSize', fo.titlefontsize, 'Interpreter', 'none');
        if isempty(hleg), hleg = h; labs = labels; end
    end

    title(tl, 'Woodbury update vs frozen and fresh, all motion cases', ...
          'FontSize', fo.titlefontsize);
    lg = legend(hleg, labs, 'FontSize', fo.legendfontsize, 'Interpreter', 'tex', ...
                'NumColumns', min(fo.legend_columns, numel(labs)));
    lg.Layout.Tile = 'south';

    f = save_woodbury_figure(fh, ...
        fullfile(outDir, 'all_cases_comparison.png'), fo);
end
