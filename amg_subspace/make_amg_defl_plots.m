function make_amg_defl_plots(rows, spectra, configs, meta, outDir)
%MAKE_AMG_DEFL_PLOTS  Panels for the deflation-vs-preconditioning study.
%
%   make_amg_defl_plots(rows, spectra, configs, meta, outDir) renders every
%   figure of run_amg_deflation_vs_precond into outDir.
%
%   The panels are ordered so the argument builds:
%     1 spectrum_by_arm          WHERE each mechanism moves the spectrum
%     2 kappa_by_config          the resulting condition numbers
%     3 lam_min_lam_max_by_config  the two ends read separately, not as a ratio
%     4 kappa_vs_msketch         how conditioning scales with sketch width
%     5 kappa_vs_q               how many V-cycle applies of sketching it takes
%     6 capture_vs_kappa         subspace quality vs conditioning, colored by
%                                how good the same V-cycle is as a preconditioner
%     7 tau_estimate             shift-estimation error, so it cannot be
%                                mistaken for subspace error, plus a direct
%                                sketch-vs-exact-vs-lam_max shift comparison
%     8 iters_by_config          the empirical companion to panel 2
%     9 aggregate_2x3.png
%
%   Panels 2, 3, 6, 7 and 8 are drawn at the largest k_target and largest q in
%   the sweep -- the best-case sketch, which is the fair setting in which to
%   ask whether deflation beats the V-cycle it came from.
%
%   CAVEAT carried in every caption: iteration counts exclude the q*m_sketch
%   V-cycle applies spent building the basis.
%
%   See also RUN_AMG_DEFLATION_VS_PRECOND, PRECOND_SPECTRUM.

    if ~isfolder(outDir), mkdir(outDir); end

    armIds    = {'amg_direct', 'defl_amg', 'defl_amg_ichol', 'ctau_amg'};
    armLabels = {'AMG as precond', 'deflation (AMG sketch)', ...
                 'ichol + AMG coarse', 'AMG + deflation'};
    armColors = [0.85 0.33 0.10;    % amg_direct   -- orange
                 0.00 0.45 0.74;    % defl_amg     -- blue
                 0.47 0.67 0.19;    % defl_amg_ichol -- green
                 0.49 0.18 0.56];   % ctau_amg     -- purple

    cfgNames  = {configs.name};
    cfgLabels = {configs.label};
    nCfg      = numel(cfgNames);

    kt_ref = max(meta.ktargets);
    q_ref  = max(meta.qs);
    m_ref  = meta.rho * kt_ref;

    ctxP = struct('rows', rows, 'spectra', spectra, 'meta', meta, ...
                  'armIds', {armIds}, 'armLabels', {armLabels}, ...
                  'armColors', armColors, 'cfgNames', {cfgNames}, ...
                  'cfgLabels', {cfgLabels}, 'nCfg', nCfg, ...
                  'kt_ref', kt_ref, 'q_ref', q_ref, 'm_ref', m_ref, ...
                  'outDir', outDir);

    panel_spectrum(ctxP);
    panel_kappa_bar(ctxP);
    panel_lam_ends(ctxP);
    panel_kappa_vs_msketch(ctxP);
    panel_kappa_vs_q(ctxP);
    panel_capture_vs_kappa(ctxP);
    panel_tau(ctxP);
    panel_iters_bar(ctxP);
    panel_aggregate(ctxP);
end

%% =========================================================================
%% 1. Spectrum: where each mechanism moves the eigenvalues
%% =========================================================================
function panel_spectrum(c)
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 11, 2.4 * c.nCfg + 1.2], 'Color', 'w');
    tl = tiledlayout(fig, c.nCfg, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    legH = gobjects(0);  legL = {};
    for ic = 1:c.nCfg
        for side = 1:2                       % 1 = bottom of spectrum, 2 = top
            ax = nexttile(tl);
            hold(ax, 'on');
            [h, lbl] = draw_spectrum_side(ax, c, c.cfgNames{ic}, side);
            hold(ax, 'off');
            set(ax, 'YScale', 'log', 'Box', 'on', 'FontSize', 8, 'LineWidth', 0.6);
            if side == 1
                ylabel(ax, sprintf('%s\n\\lambda', c.cfgLabels{ic}), ...
                       'FontSize', 8, 'FontWeight', 'bold', 'Interpreter', 'tex');
                title(ax, 'bottom of spectrum', 'FontSize', 9);
            else
                title(ax, 'top of spectrum', 'FontSize', 9);
            end
            if ic == c.nCfg
                xlabel(ax, 'Ritz index', 'FontSize', 9, 'FontWeight', 'bold');
            end
            if isempty(legH) && ~isempty(h)
                legH = h;  legL = lbl;
            end
        end
    end

    title(tl, sprintf(['Preconditioned spectrum, m_{sketch}=%d, q=%d\n', ...
                       'deflation lifts the bottom to \\tau and leaves the top ', ...
                       'untouched; AMG squeezes both ends approximately'], ...
                      c.m_ref, c.q_ref), ...
          'FontWeight', 'bold', 'FontSize', 11, 'Interpreter', 'tex');
    place_legend(tl, legH, legL, 8);
    write_fig(fig, fullfile(c.outDir, 'spectrum_by_arm.pdf'), true);
end

function [handles, labels] = draw_spectrum_side(ax, c, cfgName, side)
%DRAW_SPECTRUM_SIDE  Ritz tail for every arm of one config, plus the
%   unpreconditioned reference and the exact-deflation floor.
    handles = gobjects(0);  labels = {};

    series = [ ...
        struct('cfg', 'reference', 'arm', 'pcg_plain', 'kt', NaN, 'q', NaN, ...
               'label', 'unpreconditioned (A)', 'color', [0.4 0.4 0.4], 'style', ':'); ...
        struct('cfg', 'reference', 'arm', sprintf('defl_exact_d%d', c.m_ref), ...
               'kt', c.m_ref, 'q', NaN, ...
               'label', sprintf('deflation, exact eigvecs (d=%d)', c.m_ref), ...
               'color', [0 0 0], 'style', '--')];

    for ia = 1:numel(c.armIds)
        if strcmp(c.armIds{ia}, 'amg_direct')
            kt = NaN;  q = NaN;              % subspace-independent, measured once
        else
            kt = c.kt_ref;  q = c.q_ref;
        end
        series(end+1) = struct('cfg', cfgName, 'arm', c.armIds{ia}, ...
                               'kt', kt, 'q', q, 'label', c.armLabels{ia}, ...
                               'color', c.armColors(ia, :), 'style', '-');  %#ok<AGROW>
    end

    for is = 1:numel(series)
        s  = series(is);
        sp = find_spectrum(c.spectra, s.cfg, s.arm, s.kt, s.q);
        if isempty(sp), continue; end
        if side == 1
            y = sp.ritz_low(:);
            x = 1:numel(y);
        else
            y = sp.ritz_high(:);
            x = 1:numel(y);
        end
        if isempty(y), continue; end
        h = plot(ax, x, y, s.style, 'Color', s.color, 'LineWidth', 1.3);
        handles(end+1) = h;            %#ok<AGROW>
        labels{end+1}  = s.label;      %#ok<AGROW>
    end
end

function sp = find_spectrum(spectra, cfg, arm, kt, q)
    sel = strcmp({spectra.config}, cfg) & strcmp({spectra.arm}, arm);
    if ~isnan(kt), sel = sel & (isnan_vec([spectra.k_target]) | [spectra.k_target] == kt); end
    if ~isnan(q),  sel = sel & (isnan_vec([spectra.q]) | [spectra.q] == q); end
    idx = find(sel, 1);
    if isempty(idx), sp = []; else, sp = spectra(idx); end
end

function t = isnan_vec(v)
    t = isnan(v);
end

%% =========================================================================
%% 2 / 8. Grouped bars: kappa and iterations per config
%% =========================================================================
function panel_kappa_bar(c)
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 8.5 5.0], 'Color', 'w');
    ax = axes(fig);
    draw_bar_panel(ax, c, 'kappa', '\kappa of the preconditioned operator', true);
    title(ax, sprintf(['Condition number by AMG config (m_{sketch}=%d, q=%d)\n', ...
                       'iterations exclude the q\\cdotm_{sketch} applies spent sketching'], ...
                      c.m_ref, c.q_ref), ...
          'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'tex');
    write_fig(fig, fullfile(c.outDir, 'kappa_by_config.pdf'), true);
end

function panel_iters_bar(c)
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 8.5 5.0], 'Color', 'w');
    ax = axes(fig);
    draw_bar_panel(ax, c, 'iters', 'PCG iterations to 1e-8', false);
    title(ax, sprintf(['PCG iterations by AMG config (m_{sketch}=%d, q=%d)\n', ...
                       'NOT a cost comparison: one AMG iteration costs far more ', ...
                       'than one deflation-only iteration'], c.m_ref, c.q_ref), ...
          'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'tex');
    write_fig(fig, fullfile(c.outDir, 'iters_by_config.pdf'), true);
end

function hb = draw_bar_panel(ax, c, field, ylab, logy)
%DRAW_BAR_PANEL  One bar per (config, arm), with the reference levels drawn as
%   horizontal lines.  Skipped arms (nonsymmetric M) leave a gap and are
%   annotated, never silently omitted.
    Y = nan(c.nCfg, numel(c.armIds));
    for ic = 1:c.nCfg
        for ia = 1:numel(c.armIds)
            if strcmp(c.armIds{ia}, 'amg_direct')
                r = find_row(c.rows, c.cfgNames{ic}, c.armIds{ia}, NaN, NaN);
            else
                r = find_row(c.rows, c.cfgNames{ic}, c.armIds{ia}, c.kt_ref, c.q_ref);
            end
            if ~isempty(r) && r.ok, Y(ic, ia) = r.(field); end
        end
    end

    hb = bar(ax, Y, 'grouped');
    for ia = 1:numel(hb)
        hb(ia).FaceColor = c.armColors(ia, :);
        hb(ia).EdgeColor = 'none';
    end
    hold(ax, 'on');

    refs = { sprintf('defl_exact_d%d', c.m_ref), [0 0 0],       '--', ...
             sprintf('exact eigvec floor (d=%d)', c.m_ref); ...
             'pcg_ichol',                        [0.4 0.4 0.4], '-.', 'ichol only'; ...
             'pcg_plain',                        [0.6 0.6 0.6], ':',  'unpreconditioned' };
    for ir = 1:size(refs, 1)
        rr = find_row(c.rows, 'reference', refs{ir,1}, NaN, NaN);
        if isempty(rr) || ~rr.ok, continue; end
        yline(ax, rr.(field), refs{ir,3}, refs{ir,4}, 'Color', refs{ir,2}, ...
              'LineWidth', 1.1, 'FontSize', 7, 'LabelHorizontalAlignment', 'left', ...
              'Interpreter', 'none');
    end

    % Annotate the bars that are absent by construction rather than by failure.
    for ic = 1:c.nCfg
        for ia = 1:numel(c.armIds)
            if ~isnan(Y(ic, ia)), continue; end
            r = find_row(c.rows, c.cfgNames{ic}, c.armIds{ia}, c.kt_ref, c.q_ref);
            if isempty(r), r = find_row(c.rows, c.cfgNames{ic}, c.armIds{ia}, NaN, NaN); end
            if isempty(r), continue; end
            txt = 'skip';
            if contains(lower(r.skip_reason), 'nonsym'), txt = 'nonsym'; end
            xoff = (ia - (numel(c.armIds)+1)/2) * 0.8/numel(c.armIds);
            text(ax, ic + xoff, min(Y(:), [], 'omitnan'), txt, ...
                 'Rotation', 90, 'FontSize', 6, 'Color', c.armColors(ia,:), ...
                 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
        end
    end
    hold(ax, 'off');

    if logy
        set(ax, 'YScale', 'log');
        % On a log axis a bar is drawn from the axis floor, so a value near
        % that floor renders as no bar at all.  Give the smallest value a
        % decade of headroom below it -- lam_max_ratio for the AMG arms is
        % exactly such a small value, and it is the point of the panel.
        yp = Y(isfinite(Y) & Y > 0);
        if ~isempty(yp)
            ylim(ax, [min(yp) / 3, max(yp) * 2]);
        end
    end
    set(ax, 'XTick', 1:c.nCfg, 'XTickLabel', c.cfgLabels, 'Box', 'on', ...
            'FontSize', 8, 'LineWidth', 0.6, 'TickLabelInterpreter', 'none');
    ax.XTickLabelRotation = 25;
    ylabel(ax, ylab, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'tex');
    legend(ax, hb, c.armLabels, 'Location', 'southoutside', 'NumColumns', 2, ...
           'FontSize', 7.5, 'Box', 'on', 'Interpreter', 'none');
end

%% =========================================================================
%% 3. The two spectral ends, read separately
%% =========================================================================
function panel_lam_ends(c)
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 13 5.2], 'Color', 'w');
    tl = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    ax1 = nexttile(tl);
    draw_bar_panel(ax1, c, 'lam_min_ratio', '\lambda_{min} / \lambda_{min}(A)', true);
    title(ax1, 'bottom: how far each mechanism lifts \lambda_{min}', ...
          'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'tex');
    legend(ax1, 'off');

    ax2 = nexttile(tl);
    draw_bar_panel(ax2, c, 'lam_max_ratio', '\lambda_{max} / \lambda_{max}(A)', true);
    title(ax2, 'top: \approx1 means the mechanism never touched \lambda_{max}', ...
          'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'tex');

    title(tl, sprintf(['Which END of the spectrum each mechanism moves ', ...
                       '(m_{sketch}=%d, q=%d)'], c.m_ref, c.q_ref), ...
          'FontWeight', 'bold', 'FontSize', 11, 'Interpreter', 'tex');
    write_fig(fig, fullfile(c.outDir, 'lam_min_lam_max_by_config.pdf'), true);
end

%% =========================================================================
%% 4 / 5. kappa vs sketch width and vs subspace iterations
%% =========================================================================
function panel_kappa_vs_msketch(c)
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 4.2 * min(c.nCfg, 4) + 0.5, ...
                              4.0 * ceil(c.nCfg / 4) + 1.0], 'Color', 'w');
    tl = tiledlayout(fig, ceil(c.nCfg / 4), min(c.nCfg, 4), ...
                     'Padding', 'compact', 'TileSpacing', 'compact');
    legH = gobjects(0);  legL = {};

    msk = c.meta.rho * c.meta.ktargets;
    for ic = 1:c.nCfg
        ax = nexttile(tl);
        hold(ax, 'on');
        for ia = 1:numel(c.armIds)
            if strcmp(c.armIds{ia}, 'amg_direct')
                r = find_row(c.rows, c.cfgNames{ic}, 'amg_direct', NaN, NaN);
                if isempty(r) || ~r.ok, continue; end
                h = plot(ax, msk, repmat(r.kappa, size(msk)), '--', ...
                         'Color', c.armColors(ia,:), 'LineWidth', 1.4);
            else
                ys = arrayfun(@(kt) kappa_at(c.rows, c.cfgNames{ic}, ...
                                             c.armIds{ia}, kt, c.q_ref), ...
                              c.meta.ktargets);
                h = plot(ax, msk, ys, '-o', 'Color', c.armColors(ia,:), ...
                         'MarkerFaceColor', c.armColors(ia,:), 'MarkerSize', 3.5, ...
                         'LineWidth', 1.3);
            end
            if ic == 1, legH(end+1) = h; legL{end+1} = c.armLabels{ia}; end %#ok<AGROW>
        end
        ye = arrayfun(@(d) kappa_at(c.rows, 'reference', ...
                                    sprintf('defl_exact_d%d', d), NaN, NaN), msk);
        he = plot(ax, msk, ye, 'k--', 'LineWidth', 1.2);
        if ic == 1, legH(end+1) = he; legL{end+1} = 'exact eigvec floor'; end
        hold(ax, 'off');

        set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'on', 'FontSize', 8, ...
                'XTick', msk, 'XTickLabel', string(msk));
        title(ax, c.cfgLabels{ic}, 'FontSize', 9, 'Interpreter', 'none');
        xlabel(ax, 'm_{sketch}', 'FontSize', 9, 'Interpreter', 'tex');
        ylabel(ax, '\kappa', 'FontSize', 9, 'Interpreter', 'tex');
    end
    title(tl, sprintf(['\\kappa vs sketch width (q=%d).  The flat dashed line is ', ...
                       'the same V-cycle used as a preconditioner --\nwhere the ', ...
                       'solid curve crosses it, the sketched subspace has overtaken ', ...
                       'the V-cycle it came from.'], c.q_ref), ...
          'FontWeight', 'bold', 'FontSize', 10, 'Interpreter', 'tex');
    place_legend(tl, legH, legL, 8);
    write_fig(fig, fullfile(c.outDir, 'kappa_vs_msketch.pdf'), true);
end

function panel_kappa_vs_q(c)
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 6.6 5.0], 'Color', 'w');
    ax = axes(fig);
    hold(ax, 'on');
    markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p'};
    cmap    = lines(c.nCfg);
    handles = gobjects(0);  labels = {};

    for ic = 1:c.nCfg
        ys = arrayfun(@(q) kappa_at(c.rows, c.cfgNames{ic}, 'defl_amg', ...
                                    c.kt_ref, q), c.meta.qs);
        mk = markers{mod(ic-1, numel(markers)) + 1};
        h = plot(ax, c.meta.qs, ys, ['-' mk], 'Color', cmap(ic,:), ...
                 'MarkerFaceColor', cmap(ic,:), 'MarkerSize', 4, 'LineWidth', 1.3);
        handles(end+1) = h;  labels{end+1} = c.cfgLabels{ic};   %#ok<AGROW>

        rd = find_row(c.rows, c.cfgNames{ic}, 'amg_direct', NaN, NaN);
        if ~isempty(rd) && rd.ok
            plot(ax, c.meta.qs, repmat(rd.kappa, size(c.meta.qs)), ':', ...
                 'Color', cmap(ic,:), 'LineWidth', 1.1, 'HandleVisibility', 'off');
        end
    end
    hold(ax, 'off');

    set(ax, 'YScale', 'log', 'Box', 'on', 'FontSize', 9, 'XTick', c.meta.qs);
    xlim(ax, [min(c.meta.qs) - 0.4, max(c.meta.qs) + 0.4]);
    xlabel(ax, 'subspace iterations q (V-cycle applies per sketch column)', ...
           'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax, '\kappa of the deflated operator', 'FontSize', 10, ...
           'FontWeight', 'bold', 'Interpreter', 'tex');
    title(ax, sprintf(['\\kappa(deflation) vs sketching effort (m_{sketch}=%d)\n', ...
                       'dotted = same V-cycle used directly as a preconditioner'], ...
                      c.m_ref), 'FontSize', 10, 'FontWeight', 'bold', ...
          'Interpreter', 'tex');
    legend(ax, handles, labels, 'Location', 'southoutside', 'NumColumns', 2, ...
           'FontSize', 7.5, 'Box', 'on', 'Interpreter', 'none');
    write_fig(fig, fullfile(c.outDir, 'kappa_vs_q.pdf'), true);
end

%% =========================================================================
%% 6. The money plot: subspace quality vs conditioning
%% =========================================================================
function panel_capture_vs_kappa(c)
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 7.2 5.4], 'Color', 'w');
    ax = axes(fig);
    hold(ax, 'on');

    xs = nan(1, c.nCfg);  ys = nan(1, c.nCfg);  cs = nan(1, c.nCfg);
    for ic = 1:c.nCfg
        r = find_row(c.rows, c.cfgNames{ic}, 'defl_amg', c.kt_ref, c.q_ref);
        if isempty(r) || ~r.ok, continue; end
        xs(ic) = r.eigspace_err_2;
        ys(ic) = r.kappa;
        rd = find_row(c.rows, c.cfgNames{ic}, 'amg_direct', NaN, NaN);
        if ~isempty(rd) && rd.ok, cs(ic) = rd.kappa; end
    end

    good = isfinite(xs) & isfinite(ys);
    % Colour encodes kappa of the SAME V-cycle used as a preconditioner, mapped
    % by hand rather than through colorbar(): the label is attached to each
    % point instead, which is more readable at this handful of configs and
    % sidesteps a headless-MATLAB colorbar rendering warning.
    cmapK = hot(256);
    finC  = good & isfinite(cs);
    lo = 0;  hi = 1;
    if any(finC)
        lg = log10(cs(finC));
        lo = min(lg);  hi = max(lg);
        if hi - lo < eps, hi = lo + 1; end
    end

    for ic = find(good)
        if isfinite(cs(ic))
            t   = (log10(cs(ic)) - lo) / (hi - lo);
            col = cmapK(max(1, min(256, 1 + round(t * 200))), :);
            plot(ax, xs(ic), ys(ic), 'o', 'MarkerSize', 10, ...
                 'MarkerFaceColor', col, 'MarkerEdgeColor', [0.2 0.2 0.2], ...
                 'LineWidth', 1.0);
            lbl = sprintf('  %s  (\\kappa_{AMG}=%.3g)', c.cfgLabels{ic}, cs(ic));
        else
            % Nonsymmetric V-cycle: no kappa as a preconditioner exists, which
            % is exactly the "subspace only" case.  Open marker, never dropped.
            plot(ax, xs(ic), ys(ic), 'o', 'MarkerSize', 10, ...
                 'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0.35 0.35 0.35], ...
                 'LineWidth', 1.4);
            lbl = sprintf('  %s  (\\kappa_{AMG}: n/a)', c.cfgLabels{ic});
        end
        text(ax, xs(ic), ys(ic), lbl, 'FontSize', 7, 'Interpreter', 'tex', ...
             'VerticalAlignment', 'middle');
    end

    rr = find_row(c.rows, 'reference', sprintf('defl_exact_d%d', c.m_ref), NaN, NaN);
    if ~isempty(rr) && rr.ok
        yline(ax, rr.kappa, '--', 'exact eigvec floor', 'Color', [0 0 0], ...
              'FontSize', 7, 'Interpreter', 'none');
    end
    ri = find_row(c.rows, 'reference', 'pcg_ichol', NaN, NaN);
    if ~isempty(ri) && ri.ok
        yline(ax, ri.kappa, '-.', 'ichol only', 'Color', [0.4 0.4 0.4], ...
              'FontSize', 7, 'Interpreter', 'none');
    end
    hold(ax, 'off');

    set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'on', 'FontSize', 9);
    % The per-point labels extend to the right of their marker; widen the
    % x-range so they are not clipped by the axes box.
    xf = xs(isfinite(xs) & xs > 0);
    if ~isempty(xf), xlim(ax, [min(xf) * 0.7, max(xf) * 3.0]); end
    xlabel(ax, 'subspace capture error  ||(I-P_V)Q_{true}||_2', ...
           'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'tex');
    ylabel(ax, '\kappa with that sketched basis as deflation', ...
           'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'tex');
    title(ax, sprintf(['Cheap-AMG test (m_{sketch}=%d, q=%d)\n', ...
                       'marker brightness = \\kappa of the SAME V-cycle used as a ', ...
                       'preconditioner;\nlower-left AND bright = weak ', ...
                       'preconditioner, strong sketched subspace'], ...
                      c.m_ref, c.q_ref), ...
          'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'tex');
    write_fig(fig, fullfile(c.outDir, 'capture_vs_kappa.pdf'), true);
end

%% =========================================================================
%% 7. Shift estimation, kept separable from subspace quality
%% =========================================================================
function panel_tau(c)
    panel_tau_gap(c);
    panel_tau_modes(c);
end

function panel_tau_modes(c)
%PANEL_TAU_MODES  kappa under the sketch-only shift vs two reference shifts.
%   The sketch-only tau = lam_max(V'AV) is the only fair choice; 'exact'
%   (tau = lam_{r+1}, from the cache) is optimal ONLY for an exact eigenbasis,
%   so this panel is what shows whether the fairness constraint costs the
%   deflation arm anything.  If the sketch curve sits BELOW exact, it does not.
    modes  = {'defl_amg', 'defl_amg_tau_exact', 'defl_amg_tau_lam_max'};
    labels = {'\tau = \lambda_{max}(V''AV)  (sketch only)', ...
              '\tau = \lambda_{r+1}(A)  (cached, reference)', ...
              '\tau = \lambda_{max}(A)  (reference)'};
    styles = {'-o', '--s', ':^'};

    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 4.2 * min(c.nCfg, 4) + 0.5, ...
                              3.8 * ceil(c.nCfg / 4) + 1.0], 'Color', 'w');
    tl = tiledlayout(fig, ceil(c.nCfg / 4), min(c.nCfg, 4), ...
                     'Padding', 'compact', 'TileSpacing', 'compact');
    legH = gobjects(0);  legL = {};

    for ic = 1:c.nCfg
        ax = nexttile(tl);
        hold(ax, 'on');
        for im = 1:numel(modes)
            ys = arrayfun(@(q) kappa_at(c.rows, c.cfgNames{ic}, modes{im}, ...
                                        c.kt_ref, q), c.meta.qs);
            if all(isnan(ys)), continue; end
            h = plot(ax, c.meta.qs, ys, styles{im}, 'MarkerSize', 3.5, ...
                     'LineWidth', 1.3);
            if isempty(legH) || numel(legH) < im
                legH(end+1) = h;  legL{end+1} = labels{im};  %#ok<AGROW>
            end
        end
        hold(ax, 'off');
        set(ax, 'YScale', 'log', 'Box', 'on', 'FontSize', 8, 'XTick', c.meta.qs);
        title(ax, c.cfgLabels{ic}, 'FontSize', 9, 'Interpreter', 'none');
        xlabel(ax, 'q', 'FontSize', 9);
        ylabel(ax, '\kappa', 'FontSize', 9, 'Interpreter', 'tex');
    end
    title(tl, sprintf(['Does the sketch-only shift cost anything? ', ...
                       '(m_{sketch}=%d)\n\\lambda_{r+1} is optimal only for an ', ...
                       'EXACT eigenbasis, so for an unconverged sketch the ', ...
                       'larger Ritz-based \\tau can be the better choice'], ...
                      c.m_ref), ...
          'FontWeight', 'bold', 'FontSize', 10, 'Interpreter', 'tex');
    place_legend(tl, legH, legL, 8);
    write_fig(fig, fullfile(c.outDir, 'tau_modes.pdf'), true);
end

function panel_tau_gap(c)
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 6.6 4.8], 'Color', 'w');
    ax = axes(fig);
    hold(ax, 'on');
    markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p'};
    cmap    = lines(c.nCfg);
    handles = gobjects(0);  labels = {};

    for ic = 1:c.nCfg
        ys = arrayfun(@(q) field_at(c.rows, c.cfgNames{ic}, 'defl_amg', ...
                                    c.kt_ref, q, 'tau_ritz_gap'), c.meta.qs);
        mk = markers{mod(ic-1, numel(markers)) + 1};
        h  = plot(ax, c.meta.qs, ys, ['-' mk], 'Color', cmap(ic,:), ...
                  'MarkerFaceColor', cmap(ic,:), 'MarkerSize', 4, 'LineWidth', 1.3);
        handles(end+1) = h;  labels{end+1} = c.cfgLabels{ic};   %#ok<AGROW>
    end
    yline(ax, 1, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    hold(ax, 'off');

    set(ax, 'Box', 'on', 'FontSize', 9, 'XTick', c.meta.qs);
    xlim(ax, [min(c.meta.qs) - 0.4, max(c.meta.qs) + 0.4]);
    xlabel(ax, 'subspace iterations q', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax, '\tau_{sketch} / \lambda_{r+1}(A)', 'FontSize', 10, ...
           'FontWeight', 'bold', 'Interpreter', 'tex');
    title(ax, sprintf(['Shift estimated from the sketch alone (m_{sketch}=%d)\n', ...
                       'a \\kappa that is bad only because this ratio is off is a ', ...
                       'shift problem, not a subspace problem'], c.m_ref), ...
          'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'tex');
    legend(ax, handles, labels, 'Location', 'southoutside', 'NumColumns', 2, ...
           'FontSize', 7.5, 'Box', 'on', 'Interpreter', 'none');
    write_fig(fig, fullfile(c.outDir, 'tau_estimate.pdf'), true);
end

%% =========================================================================
%% 9. Aggregate
%% =========================================================================
function panel_aggregate(c)
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 16.5 9.5], 'Color', 'w');
    tl = tiledlayout(fig, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

    ax = nexttile(tl);
    hold(ax, 'on');
    draw_spectrum_side(ax, c, c.cfgNames{1}, 1);
    hold(ax, 'off');
    set(ax, 'YScale', 'log', 'Box', 'on', 'FontSize', 8);
    title(ax, sprintf('spectrum bottom (%s)', c.cfgLabels{1}), 'FontSize', 9, ...
          'Interpreter', 'none');

    ax = nexttile(tl);
    hbLeg = draw_bar_panel(ax, c, 'kappa', '\kappa', true);
    legend(ax, 'off');
    title(ax, '\kappa by config', 'FontSize', 9, 'Interpreter', 'tex');

    ax = nexttile(tl);
    draw_bar_panel(ax, c, 'lam_max_ratio', '\lambda_{max}/\lambda_{max}(A)', true);
    legend(ax, 'off');
    title(ax, 'top of spectrum', 'FontSize', 9, 'Interpreter', 'tex');

    ax = nexttile(tl);
    draw_bar_panel(ax, c, 'lam_min_ratio', '\lambda_{min}/\lambda_{min}(A)', true);
    legend(ax, 'off');
    title(ax, 'bottom of spectrum', 'FontSize', 9, 'Interpreter', 'tex');

    ax = nexttile(tl);
    hold(ax, 'on');
    for ic = 1:c.nCfg
        ys = arrayfun(@(q) kappa_at(c.rows, c.cfgNames{ic}, 'defl_amg', ...
                                    c.kt_ref, q), c.meta.qs);
        plot(ax, c.meta.qs, ys, '-o', 'MarkerSize', 3.5, 'LineWidth', 1.2);
    end
    hold(ax, 'off');
    set(ax, 'YScale', 'log', 'Box', 'on', 'FontSize', 8, 'XTick', c.meta.qs);
    title(ax, '\kappa(deflation) vs q', 'FontSize', 9, 'Interpreter', 'tex');
    xlabel(ax, 'q', 'FontSize', 9);

    ax = nexttile(tl);
    hold(ax, 'on');
    xsAgg = [];
    for ic = 1:c.nCfg
        r = find_row(c.rows, c.cfgNames{ic}, 'defl_amg', c.kt_ref, c.q_ref);
        if isempty(r) || ~r.ok, continue; end
        plot(ax, r.eigspace_err_2, r.kappa, 'o', 'MarkerSize', 7, ...
             'MarkerFaceColor', [0 0.45 0.74], 'MarkerEdgeColor', 'k');
        text(ax, r.eigspace_err_2, r.kappa, ['  ' c.cfgLabels{ic}], ...
             'FontSize', 6, 'Interpreter', 'none');
        xsAgg(end+1) = r.eigspace_err_2;                       %#ok<AGROW>
    end
    hold(ax, 'off');
    set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'on', 'FontSize', 8);
    % Room for the point labels, which run to the right of their marker.
    if ~isempty(xsAgg), xlim(ax, [min(xsAgg) * 0.7, max(xsAgg) * 3.5]); end
    xlabel(ax, 'capture error', 'FontSize', 9);
    ylabel(ax, '\kappa(deflation)', 'FontSize', 9, 'Interpreter', 'tex');
    title(ax, 'capture vs conditioning', 'FontSize', 9);

    title(tl, sprintf(['AMG as preconditioner vs AMG-sketched deflation ', ...
                       '(n=%d, m_{sketch}=%d, q=%d)'], c.meta.n, c.m_ref, c.q_ref), ...
          'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'tex');
    % Without this the aggregate's bar colours are unexplained -- the per-panel
    % legends are suppressed to keep the tiles readable.
    place_legend(tl, hbLeg, c.armLabels, 9);
    write_fig(fig, fullfile(c.outDir, 'aggregate_2x3.png'), false);
end

%% =========================================================================
%% Shared helpers
%% =========================================================================
function r = find_row(rows, cfg, arm, kt, q)
%FIND_ROW  First row matching (config, arm) and, when given, (k_target, q).
    sel = strcmp({rows.config}, cfg) & strcmp({rows.arm}, arm);
    if ~isnan(kt), sel = sel & [rows.k_target] == kt; end
    if ~isnan(q),  sel = sel & [rows.q] == q;         end
    idx = find(sel, 1);
    if isempty(idx), r = []; else, r = rows(idx); end
end

function v = kappa_at(rows, cfg, arm, kt, q)
    v = field_at(rows, cfg, arm, kt, q, 'kappa');
end

function v = field_at(rows, cfg, arm, kt, q, field)
    r = find_row(rows, cfg, arm, kt, q);
    if isempty(r) || ~r.ok, v = NaN; else, v = r.(field); end
end

function place_legend(tl, handles, labels, fontSize)
    if isempty(handles), return; end
    lgd = legend(handles, labels, 'Box', 'on', 'EdgeColor', [0.65 0.65 0.65], ...
                 'Color', 'white', 'FontSize', fontSize, 'Interpreter', 'none', ...
                 'NumColumns', min(numel(labels), 3));
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [18, 8];
end

function write_fig(fig, outfile, vector)
    if vector
        exportgraphics(fig, outfile, 'ContentType', 'vector');
    else
        exportgraphics(fig, outfile, 'Resolution', 200);
    end
    close(fig);
    fprintf('Wrote %s\n', outfile);
end
