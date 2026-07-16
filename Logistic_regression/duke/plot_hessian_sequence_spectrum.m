function plot_hessian_sequence_spectrum(specByIter, lambda, outDir, dsName)
%PLOT_HESSIAN_SEQUENCE_SPECTRUM  Hessian spectra along one Newton sequence.
%   PLOT_HESSIAN_SEQUENCE_SPECTRUM(specByIter, lambda, outDir, dsName) renders
%   a two-tile figure in the style of plot_hessian_spectrum, but for the
%   SEQUENCE of linear systems H_k = Xa'diag(w_k)Xa + lambda*I arising from
%   the Newton iterations at a single fixed lambda (w_k = IRLS weights used at
%   Newton step k; k = 1 is the first system, with beta = 0 and w = 0.25):
%     Tile 1 (smallest) : semilogy of the smallest-K eigenvalues of each H_k,
%                         one curve per Newton iteration (light -> dark as k
%                         grows), with a dashed reference line at lambda (the
%                         spectrum floor). The x-axis is reversed (k -> 1):
%                         eigenvalue index 1 sits at the right edge, so the
%                         spike piling onto the lambda floor sits at the right.
%     Tile 2 (largest)  : semilogy of the largest-K eigenvalues, natural axis.
%   The figure is saved as results/figures/hessian_eig_spectrum_sequence.pdf.
%
%   specByIter(k) must provide eigs_small (ascending) and eigs_large
%   (descending); see hessian_spectrum.
%
%   See also hessian_spectrum, plot_hessian_spectrum, logreg_newton,
%   run_logistic_benchmark.

    if nargin < 4 || isempty(dsName), dsName = 'dataset'; end
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    mode = specByIter(1).mode;

    fig = figure('Color', 'w', 'Position', [50, 50, 1300, 560]);
    tl  = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    draw_overlay(nexttile(tl), specByIter, lambda, 'smallest');
    draw_overlay(nexttile(tl), specByIter, lambda, 'largest');

    title(tl, sprintf(['%s: Newton sequence H_k = X^TW_kX + \\lambda I, ', ...
          '\\lambda=%.3g  (%s mode)'], dsName, lambda, mode), ...
          'FontWeight', 'bold', 'FontSize', 13, 'Interpreter', 'tex');

    outFile = fullfile(outDir, 'hessian_eig_spectrum_sequence.pdf');
    exportgraphics(fig, outFile, 'ContentType', 'vector');
    fprintf('Saved %s\n', outFile);
    close(fig);
end

%% --------- local helpers ---------
function draw_overlay(ax, specByIter, lambda, mode)
%DRAW_OVERLAY  Overlay one eigenvalue end across the Newton iterations.
%   mode = 'smallest' : eigs_small (ascending), x-axis reversed (k -> 1),
%                       so the lambda-floor spike sits at the right.
%   mode = 'largest'  : eigs_large (descending), natural x-axis (1 -> k).
%   Darker shade = later Newton iteration. A dashed yline marks the lambda
%   floor (identical for every system in the sequence).
    nIt = numel(specByIter);

    % Light -> dark blue ramp; darker = later Newton iteration.
    C_light = [0.74 0.84 0.95];
    C_dark  = [0.06 0.22 0.50];

    hold(ax, 'on'); grid(ax, 'on');
    legH = gobjects(nIt, 1);
    legL = cell(nIt, 1);
    kMax = 0;

    for k = 1:nIt
        t  = ramp_t(k, nIt);
        cc = (1 - t) * C_light + t * C_dark;
        if strcmp(mode, 'smallest')
            e = specByIter(k).eigs_small(:);
        else
            e = specByIter(k).eigs_large(:);
        end
        e(e <= 0) = NaN;
        kMax = max(kMax, numel(e));

        legH(k) = semilogy(ax, 1:numel(e), e, '-', 'LineWidth', 1.5, 'Color', cc);
        legL{k} = sprintf('it=%d', k);
    end
    yline(ax, lambda, ':k', 'LineWidth', 0.75, 'HandleVisibility', 'off');

    set(ax, 'YScale', 'log', 'FontSize', 12, 'Box', 'on');
    ylabel(ax, '\lambda_i(H_k)  (log scale)', 'FontSize', 12);
    if strcmp(mode, 'smallest')
        set(ax, 'XDir', 'reverse');
        xlabel(ax, sprintf('index i (%d \\rightarrow 1)', kMax), 'FontSize', 12);
        title(ax, sprintf('smallest %d eigenvalues', kMax), 'FontSize', 12);
    else
        set(ax, 'XDir', 'normal');
        xlabel(ax, sprintf('index i (1 \\rightarrow %d)', kMax), 'FontSize', 12);
        title(ax, sprintf('largest %d eigenvalues', kMax), 'FontSize', 12);
    end
    legend(ax, legH, legL, 'Location', 'best', 'FontSize', 9, 'Box', 'on', ...
           'NumColumns', 2);
    hold(ax, 'off');
end

function t = ramp_t(k, nIt)
    if nIt == 1
        t = 1;
    else
        t = (k - 1) / (nIt - 1);
    end
end
