function write_schur_case_outputs(run_dir, Astat, opts)
%WRITE_SCHUR_CASE_OUTPUTS  Per-case CSVs and figures.
%   WRITE_SCHUR_CASE_OUTPUTS(RUN_DIR, ASTAT, OPTS)
%
%   Mirrors the output tree of the SPD reference experiment, plus the
%   condition-number panel this study needs (kappa is the headline metric
%   here, not just iteration counts).
%
%   Writes:
%     <arm>_solver_iterations.csv / .png     one per arm
%     all_solvers_comparison.png             the key figure
%     kappa_vs_timestep.png                  lambda_min/lambda_max/kappa
%     relative_step_to_step_change.png       ||S_n - S_{n-1}||_F (normalized)
%     diff_from_initial.png                  ||S_n - S_1||_F (normalized)
%     relative_inverse_difference.png        staleness of the frozen chol
%     low_rank_inverse_difference.png        low-rank complement error
%
%   See also: solve_schur_sequence, save_benchmark_figure, solver_style_table.

    if nargin < 3 || isempty(opts), opts = benchmark_fig_defaults(); end
    if ~exist(run_dir, 'dir'), mkdir(run_dir); end

    keys   = Astat.solver_keys;
    labels = Astat.solver_labels;
    nk     = numel(keys);
    steps  = (1:Astat.nsteps)';
    sty    = solver_style_table(nk);

    % --- per-arm CSV + figure ----------------------------------------------
    for i = 1:nk
        its = Astat.solver_its.(keys{i});
        writematrix(its, fullfile(run_dir, [keys{i} '_solver_iterations.csv']));

        fh = new_figure(opts.single_width, opts.single_height);
        ax = axes(fh);
        plot(ax, steps, its, 'LineWidth', 1.6, 'Color', sty(i).color, ...
             'Marker', sty(i).marker, 'MarkerSize', 5, ...
             'MarkerIndices', marker_idx(numel(steps), opts.marker_targets));
        grid(ax, 'on');
        xlabel(ax, 'time step', 'FontSize', opts.labelfontsize);
        ylabel(ax, opts.iter_label, 'FontSize', opts.labelfontsize);
        title(ax, labels{i}, 'FontSize', opts.titlefontsize, 'Interpreter', 'none');
        ax.FontSize = opts.fontsize;
        save_benchmark_figure(fh, ...
            fullfile(run_dir, [keys{i} '_solver_iterations.png']), opts);
    end

    % --- all-arms comparison (always kept) ---------------------------------
    fh = new_figure(opts.multi_width, opts.multi_height);
    tl = tiledlayout(fh, 1, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    ax = nexttile(tl);
    h = gobjects(nk, 1);
    for i = 1:nk
        h(i) = semilogy(ax, steps, max(Astat.solver_its.(keys{i}), 1), ...
            'LineWidth', sty(i).linewidth, 'Color', sty(i).color, ...
            'LineStyle', sty(i).linestyle, 'Marker', sty(i).marker, ...
            'MarkerSize', sty(i).markersize, ...
            'MarkerIndices', marker_idx(numel(steps), opts.marker_targets));
        hold(ax, 'on');
    end
    grid(ax, 'on');
    xlabel(ax, 'time step', 'FontSize', opts.labelfontsize);
    ylabel(ax, opts.iter_label, 'FontSize', opts.labelfontsize);
    title(tl, sprintf('%s: %s, all arms', Astat.case_name, opts.iter_label), ...
          'FontSize', opts.titlefontsize, 'Interpreter', 'none');
    ax.FontSize = opts.fontsize;
    lgd = legend(ax, h, labels, 'Interpreter', 'none', ...
                 'FontSize', opts.legendfontsize, 'NumColumns', 2);
    lgd.Layout.Tile = 'south';
    save_benchmark_figure(fh, fullfile(run_dir, 'all_solvers_comparison.png'), opts);

    % --- conditioning -------------------------------------------------------
    if any(~isnan(Astat.kappa))
        fh = new_figure(opts.accuracy_width, opts.accuracy_height);
        tl = tiledlayout(fh, 2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
        ax1 = nexttile(tl);
        semilogy(ax1, steps, Astat.kappa, 'LineWidth', 1.6, 'Color', [0 0.45 0.70]);
        grid(ax1, 'on'); ylabel(ax1, '\kappa(S)', 'FontSize', opts.labelfontsize);
        ax1.FontSize = opts.fontsize;
        ax2 = nexttile(tl);
        semilogy(ax2, steps, Astat.lambda_max, 'LineWidth', 1.6, ...
                 'Color', [0.84 0.37 0], 'DisplayName', '\lambda_{max}');
        hold(ax2, 'on');
        semilogy(ax2, steps, Astat.lambda_min, 'LineWidth', 1.6, ...
                 'Color', [0 0.62 0.45], 'DisplayName', '\lambda_{min}');
        grid(ax2, 'on');
        xlabel(ax2, 'time step', 'FontSize', opts.labelfontsize);
        ylabel(ax2, 'eigenvalue', 'FontSize', opts.labelfontsize);
        legend(ax2, 'Location', 'best', 'FontSize', opts.legendfontsize);
        ax2.FontSize = opts.fontsize;
        title(tl, sprintf('%s: conditioning of S(t_n)', Astat.case_name), ...
              'FontSize', opts.titlefontsize, 'Interpreter', 'none');
        save_benchmark_figure(fh, fullfile(run_dir, 'kappa_vs_timestep.png'), opts);
    end

    % --- drift diagnostics --------------------------------------------------
    single_curve(run_dir, 'relative_step_to_step_change.png', steps, ...
        Astat.ReldiffF, '||S_n - S_{n-1}||_F (normalized)', ...
        sprintf('%s: relative step-to-step change', Astat.case_name), opts);
    single_curve(run_dir, 'diff_from_initial.png', steps, ...
        Astat.RelInitdiffF, '||S_n - S_1||_F (normalized)', ...
        sprintf('%s: drift from the initial operator', Astat.case_name), opts);
    single_curve(run_dir, 'relative_inverse_difference.png', steps, ...
        Astat.InvRelDiff, 'relative inverse error', ...
        sprintf('%s: staleness of the frozen exact inverse', Astat.case_name), opts);
    if any(~isnan(Astat.LowRankInvRelDiff))
        single_curve(run_dir, 'low_rank_inverse_difference.png', steps, ...
            Astat.LowRankInvRelDiff, 'relative low-rank complement error', ...
            sprintf('%s: low-rank inverse difference', Astat.case_name), opts);
    end
    if any(~isnan(Astat.coupling_change))
        single_curve(run_dir, 'coupling_change.png', steps, ...
            Astat.coupling_change, '||\DeltaC||_F / ||C||_F', ...
            sprintf('%s: coupling-block change', Astat.case_name), opts);
    end
end

%==========================================================================
function single_curve(run_dir, fname, steps, y, ylab, ttl, opts)
    if all(isnan(y)), return; end
    fh = new_figure(opts.single_width, opts.single_height);
    ax = axes(fh);
    semilogy(ax, steps, max(y, eps), 'LineWidth', 1.6, 'Color', [0.35 0.35 0.35], ...
             'Marker', 'o', 'MarkerSize', 4, ...
             'MarkerIndices', marker_idx(numel(steps), opts.marker_targets));
    grid(ax, 'on');
    xlabel(ax, 'time step', 'FontSize', opts.labelfontsize);
    ylabel(ax, ylab, 'FontSize', opts.labelfontsize);
    title(ax, ttl, 'FontSize', opts.titlefontsize, 'Interpreter', 'none');
    ax.FontSize = opts.fontsize;
    save_benchmark_figure(fh, fullfile(run_dir, fname), opts);
end

%==========================================================================
function idx = marker_idx(npts, ntarget)
    step = max(1, round(npts / max(ntarget, 1)));
    idx  = 1:step:npts;
end

%==========================================================================
function fh = new_figure(w, h)
    fh = figure('Visible', 'off', 'Units', 'inches', ...
                'Position', [1 1 w h], 'Color', 'w');
end
