function plot_pcg_results(resultsTable, outDir)
%PLOT_PCG_RESULTS  Plot PCG benchmark results (no-precond vs ichol).
%   PLOT_PCG_RESULTS(RESULTSTABLE, OUTDIR) generates, for each subset size n
%   in RESULTSTABLE, a 2-by-2 figure versus the system index j:
%     (1) PCG iterations: none vs ichol
%     (2) PCG solve time: none vs ichol
%     (3) total time: none vs (ichol solve + ichol build)
%     (4) iteration speedup: iters_none ./ iters_ichol
%   Each figure is saved as a vector PDF in OUTDIR.
%
%   Inputs
%     resultsTable : table from run_kernel_pcg_benchmark with columns
%                    n, j, iters_none, time_none, iters_ichol, time_ichol,
%                    ichol_build_time, ichol_failed.
%     outDir       : (optional, default 'results/figures') output directory.
%
%   Example
%     plot_pcg_results(T, 'results/figures');

    if nargin < 2 || isempty(outDir)
        outDir = fullfile('results', 'figures');
    end
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    sizes = unique(resultsTable.n, 'stable');
    for s = 1:numel(sizes)
        n = sizes(s);
        T = sortrows(resultsTable(resultsTable.n == n, :), 'j');
        fig = figure('Name', sprintf('PCG benchmark n=%d', n), ...
                     'Color', 'w', 'Position', [100, 100, 1000, 750]);
        tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        j = T.j;
        total_ichol = T.time_ichol + T.ichol_build_time;
        speedup = T.iters_none ./ T.iters_ichol;

        % (1) iterations
        nexttile;
        plot(j, T.iters_none, '-o', 'LineWidth', 1.2); hold on;
        plot(j, T.iters_ichol, '-s', 'LineWidth', 1.2);
        grid on; xlabel('system index j'); ylabel('PCG iterations');
        title('Iterations vs system index');
        legend({'no precond', 'ichol'}, 'Location', 'best');

        % (2) solve time
        nexttile;
        plot(j, T.time_none, '-o', 'LineWidth', 1.2); hold on;
        plot(j, T.time_ichol, '-s', 'LineWidth', 1.2);
        grid on; xlabel('system index j'); ylabel('solve time [s]');
        title('Solve time vs system index');
        legend({'no precond', 'ichol solve'}, 'Location', 'best');

        % (3) total time incl. ichol build
        nexttile;
        plot(j, T.time_none, '-o', 'LineWidth', 1.2); hold on;
        plot(j, total_ichol, '-s', 'LineWidth', 1.2);
        grid on; xlabel('system index j'); ylabel('total time [s]');
        title('Total time (ichol incl. build)');
        legend({'no precond', 'ichol build+solve'}, 'Location', 'best');

        % (4) iteration speedup
        nexttile;
        plot(j, speedup, '-d', 'LineWidth', 1.2); hold on;
        yline(1, '--', 'Color', [0.5 0.5 0.5]);
        grid on; xlabel('system index j'); ylabel('iters_{none} / iters_{ichol}');
        title('Iteration speedup of ichol');

        sgtitle(sprintf('Kernel-ridge PCG benchmark, n = %d', n), ...
                'FontWeight', 'bold');

        outFile = fullfile(outDir, sprintf('pcg_benchmark_n%d.pdf', n));
        exportgraphics(fig, outFile, 'ContentType', 'vector');
        fprintf('  saved %s\n', outFile);
        close(fig);
    end
end
