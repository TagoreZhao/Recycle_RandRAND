function plot_hessian_spectrum(specByLambda, lambdaList, outDir, dsName)
%PLOT_HESSIAN_SPECTRUM  Smallest|largest Newton-Hessian spectra across lambda.
%   PLOT_HESSIAN_SPECTRUM(specByLambda, lambdaList, outDir, dsName) renders a
%   two-tile figure mirroring GP_train/plot_kernel_spectrum.m:
%     Tile 1 (smallest) : semilogy of the smallest-K Hessian eigenvalues, one
%                         curve per lambda (light -> dark as lambda grows), with
%                         a dashed reference line at each lambda (the spectrum
%                         floor). The x-axis is reversed (k -> 1) so the spike
%                         piling onto the lambda floor sits at the left.
%     Tile 2 (largest)  : semilogy of the largest-K eigenvalues, natural axis.
%   The figure is saved as results/figures/hessian_eig_spectrum.pdf.
%
%   specByLambda(il) must provide eigs_small (ascending) and eigs_large
%   (descending) plus lambda_floor; see hessian_spectrum.

    if nargin < 4 || isempty(dsName), dsName = 'dataset'; end
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    mode = specByLambda(1).mode;

    fig = figure('Color', 'w', 'Position', [50, 50, 1300, 560]);
    tl  = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    draw_overlay(nexttile(tl), specByLambda, lambdaList, 'smallest');
    draw_overlay(nexttile(tl), specByLambda, lambdaList, 'largest');

    title(tl, sprintf('%s: logistic Newton Hessian H = X^TWX + \\lambda I  (%s mode)', ...
          dsName, mode), 'FontWeight', 'bold', 'FontSize', 13, 'Interpreter', 'tex');

    outFile = fullfile(outDir, 'hessian_eig_spectrum.pdf');
    exportgraphics(fig, outFile, 'ContentType', 'vector');
    fprintf('Saved %s\n', outFile);
    close(fig);
end

%% --------- local helpers ---------
function draw_overlay(ax, specByLambda, lambdaList, mode)
%DRAW_OVERLAY  Overlay one eigenvalue end across all lambda values.
%   mode = 'smallest' : eigs_small (ascending), x-axis reversed (k -> 1).
%   mode = 'largest'  : eigs_large (descending), natural x-axis (1 -> k).
%   Darker shade = larger lambda. A dashed yline marks each lambda floor.
    nL = numel(specByLambda);

    % Light -> dark blue ramp; darker = larger lambda.
    C_light = [0.74 0.84 0.95];
    C_dark  = [0.06 0.22 0.50];

    hold(ax, 'on'); grid(ax, 'on');
    legH = gobjects(nL, 1);
    legL = cell(nL, 1);
    kMax = 0;

    for il = 1:nL
        t  = ramp_t(il, nL);
        cc = (1 - t) * C_light + t * C_dark;
        if strcmp(mode, 'smallest')
            e = specByLambda(il).eigs_small(:);
        else
            e = specByLambda(il).eigs_large(:);
        end
        e(e <= 0) = NaN;
        kMax = max(kMax, numel(e));

        legH(il) = semilogy(ax, 1:numel(e), e, '-', 'LineWidth', 1.5, 'Color', cc);
        legL{il} = sprintf('\\lambda=%.3g', lambdaList(il));
        yline(ax, specByLambda(il).lambda_floor, ':', 'Color', cc, ...
              'LineWidth', 0.75, 'HandleVisibility', 'off');
    end

    set(ax, 'YScale', 'log', 'FontSize', 12, 'Box', 'on');
    ylabel(ax, '\lambda_i(H)  (log scale)', 'FontSize', 12);
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

function t = ramp_t(il, nL)
    if nL == 1
        t = 1;
    else
        t = (il - 1) / (nL - 1);
    end
end
