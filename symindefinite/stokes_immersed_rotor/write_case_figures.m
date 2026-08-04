function write_case_figures(run_dir, stats, opts)
%WRITE_CASE_FIGURES  The four per-case figure kinds for one motion case.
%
%   WRITE_CASE_FIGURES(RUN_DIR, STATS, OPTS)
%
%   Writes, into RUN_DIR:
%       <key>_solver_iterations.png     one per solver
%       all_solvers_comparison.png      every solver overlaid
%       relative_step_to_step_change.png
%       accuracy.png
%
%   STATS follows the contract in load_benchmark_stats, so run_benchmark (live)
%   and replot_benchmark (from CSV) produce byte-comparable figures.
%
%   See also: plot_solver_curves, place_solver_legend, save_benchmark_figure.

    if nargin < 3 || isempty(opts), opts = benchmark_fig_defaults(); end
    if ~exist(run_dir, 'dir'), mkdir(run_dir); end

    keys   = stats.solver_keys;
    labels = stats.solver_labels;
    ns     = numel(stats.solver_its.(keys{1}));
    tax    = (1:ns)' * stats.dt;

    write_per_solver_figures(run_dir, stats, tax, keys, labels, opts);
    write_comparison_figure(run_dir, stats, tax, opts);
    write_coupling_change_figure(run_dir, stats, tax, opts);
    write_accuracy_figure(run_dir, stats, tax, opts);
end

%==========================================================================
function write_per_solver_figures(run_dir, stats, tax, keys, labels, opts)
% Title carries the short name only; the 57-character registry label goes into
% the subtitle at 9 pt, where it fits instead of running off the top edge.
    for s = 1:numel(keys)
        itv = max(stats.solver_its.(keys{s})(:), 1);
        fh  = new_figure(opts.single_width, opts.single_height);
        ax  = axes(fh); %#ok<LAXES>
        sty = solver_style_table(numel(keys));
        semilogy(ax, tax, itv, 'Color', sty(s).color, 'LineWidth', 1.6, ...
                 'Marker', sty(s).marker, 'MarkerSize', 4);
        grid(ax, 'on');
        % MATLAB fits the y axis tight to the data, which clips the marker at
        % each extremum against the frame. Pad multiplicatively (log axis).
        yl = ylim(ax);
        ylim(ax, [yl(1) * 0.98, yl(2) * 1.02]);
        xlabel(ax, 't');
        ylabel(ax, 'MINRES iterations');
        title(ax, sprintf('%s  |  %s', stats.case_name, solver_short_label(keys{s})), ...
              'Interpreter', 'none', 'FontSize', opts.titlefontsize, ...
              'FontWeight', 'bold');
        subtitle(ax, labels{s}, 'Interpreter', 'tex', ...
                 'FontSize', opts.subtitlefontsize);
        save_benchmark_figure(fh, ...
            fullfile(run_dir, [keys{s} '_solver_iterations.png']), opts);
    end
end

%==========================================================================
function write_comparison_figure(run_dir, stats, tax, opts)
    fh = new_figure(opts.multi_width, opts.multi_height);
    tl = tiledlayout(fh, 1, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    ax = nexttile(tl);
    [h, legLabels] = plot_solver_curves(ax, tax, stats, 't', opts);
    title(tl, sprintf('%s: MINRES iterations, all solvers', stats.case_name), ...
          'Interpreter', 'none', 'FontWeight', 'bold', ...
          'FontSize', opts.titlefontsize);
    place_solver_legend(tl, h, legLabels, opts);
    save_benchmark_figure(fh, fullfile(run_dir, 'all_solvers_comparison.png'), opts);
end

%==========================================================================
function write_coupling_change_figure(run_dir, stats, tax, opts)
    fh = new_figure(opts.single_width, opts.single_height);
    ax = axes(fh);
    plot(ax, tax, stats.coupling_change(:), '.-', 'LineWidth', 1.4, ...
         'Color', [0.00 0.45 0.70]);
    grid(ax, 'on');
    xlabel(ax, 't');
    ylabel(ax, '||\DeltaC||_F / ||C||_F');
    title(ax, sprintf('%s  |  per-step coupling change', stats.case_name), ...
          'Interpreter', 'none', 'FontSize', opts.titlefontsize, ...
          'FontWeight', 'bold');
    save_benchmark_figure(fh, ...
        fullfile(run_dir, 'relative_step_to_step_change.png'), opts);
end

%==========================================================================
function write_accuracy_figure(run_dir, stats, tax, opts)
%WRITE_ACCURACY_FIGURE  Whichever accuracy measures this results set actually has.
%
%   The error-of-the-last-solver-vs-backslash curve is only available when the
%   run recorded it (stats.solver_err, written to all_results.csv as
%   solver_err_last from this revision on).  Older results directories only
%   carry the MINRES relative residual, so we plot that and say so in the
%   subtitle rather than relabelling a residual as an error.
    keys = stats.solver_keys;
    lastShort = solver_short_label(keys{end});

    y = {}; lab = {}; sty = {};
    if isfield(stats, 'solver_err') && isfield(stats.solver_err, keys{end})
        y{end+1}   = stats.solver_err.(keys{end})(:);
        lab{end+1} = sprintf('%s: error vs backslash', lastShort);
        sty{end+1} = '-';
        note = '';
    else
        note = 'error vs backslash not recorded in this results directory';
    end
    % The two drivers name this differently -- solve_stokes_immersed keeps a
    % per-solver struct (solver_relres), all_results.csv flattens the last
    % solver's column to relres -- so accept either, or the live figure loses a
    % curve the replotted one shows.
    rr = [];
    if isfield(stats, 'solver_relres') && isfield(stats.solver_relres, keys{end})
        rr = stats.solver_relres.(keys{end})(:);
    elseif isfield(stats, 'relres')
        rr = stats.relres(:);
    end
    if ~isempty(rr)
        y{end+1}   = rr;
        lab{end+1} = sprintf('%s: MINRES relative residual', lastShort);
        sty{end+1} = '-';
    end
    if isfield(stats, 'backslash_relres')
        y{end+1}   = stats.backslash_relres(:);
        lab{end+1} = 'backslash relative residual';
        sty{end+1} = ':';
    end
    y{end+1}   = stats.constraint_res(:);
    lab{end+1} = 'constraint ||Cu-g||/||g||';
    sty{end+1} = '--';

    colors = [0.00 0.45 0.70; 0.84 0.37 0.00; 0.00 0.62 0.45; 0.60 0.60 0.60];

    fh = new_figure(opts.accuracy_width, opts.accuracy_height);
    ax = axes(fh);
    hold(ax, 'on');
    h = gobjects(numel(y), 1);
    for i = 1:numel(y)
        h(i) = plot(ax, tax, max(y{i}, 1e-16), sty{i}, 'LineWidth', 1.4, ...
                    'Color', colors(mod(i-1, size(colors,1)) + 1, :));
    end
    hold(ax, 'off');
    set(ax, 'YScale', 'log');
    grid(ax, 'on');
    xlabel(ax, 't');
    ylabel(ax, 'relative');
    title(ax, sprintf('%s  |  accuracy', stats.case_name), ...
          'Interpreter', 'none', 'FontSize', opts.titlefontsize, ...
          'FontWeight', 'bold');
    if ~isempty(note)
        subtitle(ax, note, 'Interpreter', 'none', ...
                 'FontSize', opts.subtitlefontsize);
    end
    lgd = legend(ax, h, lab, 'Interpreter', 'none', ...
                 'FontSize', opts.legendfontsize, 'NumColumns', 1, ...
                 'Location', 'southoutside', 'Box', 'on', ...
                 'EdgeColor', [0.65 0.65 0.65]);
    lgd.ItemTokenSize = [16 8];
    save_benchmark_figure(fh, fullfile(run_dir, 'accuracy.png'), opts);
end

%==========================================================================
function fh = new_figure(w, h)
    fh = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', ...
                'Position', [1 1 w h]);
end
