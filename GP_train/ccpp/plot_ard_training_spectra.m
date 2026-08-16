function output = plot_ard_training_spectra(X, trainingResults, params)
%PLOT_ARD_TRAINING_SPECTRA  Both absolute spectral tails along ARD training.
%   The canonical trajectory is exact fresh deflation.  Snapshots are the
%   initial state, maximum off-diagonal kernel drift, and final state.
%   Spectrum work is diagnostic and is not included in solver timings.

    canonical = "defl_exact_fresh_oracle";
    C = trainingResults(trainingResults.method == canonical & trainingResults.completed, :);
    if isempty(C)
        successful = trainingResults(trainingResults.completed, :);
        if isempty(successful)
            error('plot_ard_training_spectra:noTrajectory', ...
                  'No successful optimizer state is available for spectrum diagnostics.');
        end
        counts = groupsummary(successful, 'method');
        [~, idx] = max(counts.GroupCount);
        canonical = counts.method(idx);
        C = successful(successful.method == canonical, :);
        warning('plot_ard_training_spectra:fallbackTrajectory', ...
                'Fresh-exact trajectory unavailable; using %s.', canonical);
    end
    C = sortrows(C, 'step');

    drift = C.offdiag_kernel_relative_change;
    drift(~isfinite(drift)) = -Inf;
    [~, maxIdx] = max(drift);
    if all(drift == -Inf), maxIdx = min(2, height(C)); end
    candidate = [1, maxIdx, height(C)];
    selected = unique(candidate, 'stable');
    labels = strings(numel(selected), 1);
    for i = 1:numel(selected)
        if selected(i) == 1
            labels(i) = "initial";
        elseif selected(i) == height(C)
            labels(i) = "final";
        else
            labels(i) = "maximum drift";
        end
    end

    n = size(X, 1);
    tailCount = min(params.SpectrumCount, floor((n - 1) / 2));
    coarseRank = min(params.Rank, floor((n - 1) / 2));
    if tailCount < params.SpectrumCount
        warning('plot_ard_training_spectra:tailCountClamped', ...
                'Spectrum tail count reduced from %d to %d for n=%d.', ...
                params.SpectrumCount, tailCount, n);
    end
    fprintf('\n[Spectrum] canonical=%s snapshots=%d eigenvalues/tail=%d\n', ...
            canonical, numel(selected), tailCount);

    % Build one dense exact factor from the canonical initial state.  This
    % same L0 is used at every snapshot; it is not refreshed as theta moves.
    thetaInitial = table_theta(C(1, :));
    K0 = ard_rbf_kernel(X, exp(thetaInitial(1:4)));
    A0 = exp(thetaInitial(5)) * K0 + exp(thetaInitial(6)) * eye(n);
    L0 = chol((A0 + A0') / 2, 'lower');

    fullSpectra = repmat(struct('label', "", 'step', NaN, 'theta', [], ...
        'lambda_A', [], 'lambda_recycled_exact_split', [], 'lambda_defl_none', [], ...
        'lambda_defl_large', [], 'lambda_defl_small', [], 'lambda_defl_both', []), ...
        numel(selected), 1);
    valueRows = struct([]); summaryRows = struct([]);
    initialLarge = [];

    for isnap = 1:numel(selected)
        row = C(selected(isnap), :);
        theta = table_theta(row);
        ell = exp(theta(1:4)); sf2 = exp(theta(5)); sn2 = exp(theta(6));
        K = ard_rbf_kernel(X, ell);
        A = sf2 * K + sn2 * eye(n); A = (A + A') / 2;

        fprintf('  %s (step %d): dense eig(A) ...\n', labels(isnap), row.step);
        [Q, lambdaA] = eig(A, 'vector'); lambdaA = real(lambdaA);
        warn_negative(lambdaA, sprintf('A/%s', labels(isnap)));
        [~, absOrder] = sort(abs(lambdaA), 'ascend');
        Vsmall = Q(:, absOrder(1:coarseRank));
        Vlarge = Q(:, absOrder(end-coarseRank+1:end));
        if isempty(initialLarge), initialLarge = Vlarge; end
        overlapLarge = norm(initialLarge' * Vlarge, 'fro')^2 / coarseRank;
        overlapSmall = norm(initialLarge' * Vsmall, 'fro')^2 / coarseRank;

        fprintf('  %s (step %d): dense eig(L0^{-1} A L0^{-T}) ...\n', labels(isnap), row.step);
        M = L0 \ A / L0'; M = (M + M') / 2;
        lambdaM = real(eig(M, 'vector'));
        warn_negative(lambdaM, sprintf('recycled exact split/%s', labels(isnap)));

        lambdaNone = ideal_deflated_spectrum(lambdaA, 'none', coarseRank, params.Tau);
        lambdaLarge = ideal_deflated_spectrum(lambdaA, 'large', coarseRank, params.Tau);
        lambdaSmall = ideal_deflated_spectrum(lambdaA, 'small', coarseRank, params.Tau);
        lambdaBoth = ideal_deflated_spectrum(lambdaA, 'both', coarseRank, params.Tau);

        fullSpectra(isnap).label = labels(isnap);
        fullSpectra(isnap).step = row.step;
        fullSpectra(isnap).theta = theta;
        fullSpectra(isnap).lambda_A = lambdaA;
        fullSpectra(isnap).lambda_recycled_exact_split = lambdaM;
        fullSpectra(isnap).lambda_defl_none = lambdaNone;
        fullSpectra(isnap).lambda_defl_large = lambdaLarge;
        fullSpectra(isnap).lambda_defl_small = lambdaSmall;
        fullSpectra(isnap).lambda_defl_both = lambdaBoth;

        spectra = {lambdaA, lambdaM, lambdaNone, lambdaLarge, lambdaSmall, lambdaBoth};
        names = {"A", "recycled_exact_split", "defl_none", "defl_large", "defl_small", "defl_both"};
        for io = 1:numel(spectra)
            tails = extract_spectrum_tails(spectra{io}, tailCount);
            newValues = spectrum_value_rows(labels(isnap), row.step, names{io}, ...
                                            tails, sn2);
            if isempty(valueRows), valueRows = newValues; else, valueRows = [valueRows; newValues]; end %#ok<AGROW>
            if names{io} == "A"
                opOverlapLarge = overlapLarge; opOverlapSmall = overlapSmall;
            else
                opOverlapLarge = NaN; opOverlapSmall = NaN;
            end
            srow = spectrum_summary_row(labels(isnap), row.step, names{io}, spectra{io}, ...
                sn2, coarseRank, opOverlapLarge, opOverlapSmall);
            if isempty(summaryRows), summaryRows = srow; else, summaryRows(end+1, 1) = srow; end %#ok<AGROW>
        end
    end

    spectrumValues = struct2table(valueRows);
    spectrumSummary = struct2table(summaryRows);
    writetable(spectrumValues, fullfile(params.OutDir, 'spectrum_values.csv'));
    writetable(spectrumSummary, fullfile(params.OutDir, 'spectrum_summary.csv'));
    save(fullfile(params.OutDir, 'spectrum_full.mat'), 'fullSpectra', 'canonical', ...
         'selected', 'labels', 'tailCount', 'coarseRank', '-v7.3');
    render_raw_split(fullSpectra, tailCount, params.OutDir);
    render_tail_ablation(fullSpectra, tailCount, coarseRank, params.OutDir);

    output = struct('values', spectrumValues, 'summary', spectrumSummary, ...
                    'full', fullSpectra, 'canonical_method', canonical, ...
                    'selected_rows', selected, 'labels', labels, ...
                    'tail_count', tailCount, 'coarse_rank', coarseRank);
end

% ========================================================================
function theta = table_theta(row)
    theta = [row.log_ell1; row.log_ell2; row.log_ell3; row.log_ell4; ...
             row.log_signal_variance; row.log_noise_variance];
end

function rows = spectrum_value_rows(label, step, operator, tails, noiseFloor)
    count = numel(tails.small);
    template = struct('snapshot', "", 'step', NaN, 'operator', "", 'tail', "", ...
                      'absolute_rank', NaN, 'signed_eigenvalue', NaN, ...
                      'absolute_eigenvalue', NaN, 'noise_floor', NaN, ...
                      'distance_from_noise_floor', NaN);
    rows = repmat(template, 2 * count, 1);
    for i = 1:count
        rows(i) = fill_value(template, label, step, operator, "smallest_abs", i, ...
                             tails.small(i), noiseFloor);
        rows(count+i) = fill_value(template, label, step, operator, "largest_abs", i, ...
                                   tails.large(i), noiseFloor);
    end
end

function row = fill_value(row, label, step, operator, tail, rank, lambda, noiseFloor)
    row.snapshot = label; row.step = step; row.operator = operator; row.tail = tail;
    row.absolute_rank = rank; row.signed_eigenvalue = lambda;
    row.absolute_eigenvalue = abs(lambda); row.noise_floor = noiseFloor;
    row.distance_from_noise_floor = lambda - noiseFloor;
end

function row = spectrum_summary_row(label, step, operator, lambda, noiseFloor, ...
        coarseRank, overlapLarge, overlapSmall)
    lambda = real(lambda(:)); absAsc = sort(abs(lambda), 'ascend');
    absDesc = absAsc(end:-1:1); k = min(coarseRank, numel(lambda) - 1);
    if all(lambda > 0)
        conditionNumber = max(lambda) / min(lambda);
        rho = (sqrt(conditionNumber) - 1) / (sqrt(conditionNumber) + 1);
    else
        conditionNumber = NaN; rho = NaN;
    end
    smallGap = absAsc(k+1) / max(absAsc(k), realmin);
    largeGap = absDesc(k) / max(absDesc(k+1), realmin);
    smallNoise = median(abs(absAsc(1:k) - noiseFloor)) / max(noiseFloor, realmin);
    row = struct('snapshot', label, 'step', step, 'operator', operator, ...
        'minimum_signed_eigenvalue', min(lambda), 'maximum_signed_eigenvalue', max(lambda), ...
        'minimum_absolute_eigenvalue', min(abs(lambda)), ...
        'maximum_absolute_eigenvalue', max(abs(lambda)), ...
        'condition_number', conditionNumber, 'predicted_cg_factor', rho, ...
        'small_tail_gap_at_coarse_rank', smallGap, ...
        'large_tail_gap_at_coarse_rank', largeGap, ...
        'noise_floor', noiseFloor, 'small_tail_median_relative_noise_distance', smallNoise, ...
        'initial_large_to_fresh_large_overlap', overlapLarge, ...
        'initial_large_to_fresh_small_overlap', overlapSmall);
end

function render_raw_split(S, tailCount, outDir)
    nS = numel(S);
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [20 20 520*nS 850]);
    tl = tiledlayout(fig, 2, nS, 'Padding', 'compact', 'TileSpacing', 'compact');
    for i = 1:nS
        ta = extract_spectrum_tails(S(i).lambda_A, tailCount);
        tm = extract_spectrum_tails(S(i).lambda_recycled_exact_split, tailCount);
        ax = nexttile(tl, i); hold(ax, 'on');
        semilogy(ax, 1:tailCount, ta.small_abs, '-', 'LineWidth', 1.4, 'DisplayName', 'A');
        semilogy(ax, 1:tailCount, tm.small_abs, '-', 'LineWidth', 1.4, 'DisplayName', 'L_0^{-1}AL_0^{-T}');
        format_tail_axes(ax, sprintf('%s: %d smallest |\\lambda|', S(i).label, tailCount));
        ax = nexttile(tl, nS+i); hold(ax, 'on');
        semilogy(ax, 1:tailCount, ta.large_abs, '-', 'LineWidth', 1.4, 'DisplayName', 'A');
        semilogy(ax, 1:tailCount, tm.large_abs, '-', 'LineWidth', 1.4, 'DisplayName', 'L_0^{-1}AL_0^{-T}');
        format_tail_axes(ax, sprintf('%s: %d largest |\\lambda|', S(i).label, tailCount));
    end
    lg = legend(nexttile(tl, 1), 'Location', 'best'); lg.Layout.Tile = 'south';
    title(tl, 'ARD GP spectrum: raw and recycled exact-factor operators');
    exportgraphics(fig, fullfile(outDir, 'spectrum_raw_and_recycled_exact.pdf'), ...
                   'ContentType', 'vector');
    close(fig);
    obsolete = fullfile(outDir, 'spectrum_raw_and_ichol.pdf');
    if exist(obsolete, 'file') == 2, delete(obsolete); end
end

function render_tail_ablation(S, tailCount, coarseRank, outDir)
    nS = numel(S);
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [20 20 520*nS 850]);
    tl = tiledlayout(fig, 2, nS, 'Padding', 'compact', 'TileSpacing', 'compact');
    fields = {'lambda_defl_none','lambda_defl_large','lambda_defl_small','lambda_defl_both'};
    names = {'none','largest only','smallest only','both tails'};
    for i = 1:nS
        axS = nexttile(tl, i); hold(axS, 'on');
        axL = nexttile(tl, nS+i); hold(axL, 'on');
        for j = 1:numel(fields)
            tails = extract_spectrum_tails(S(i).(fields{j}), tailCount);
            semilogy(axS, 1:tailCount, tails.small_abs, '-', 'LineWidth', 1.2, ...
                     'DisplayName', names{j});
            semilogy(axL, 1:tailCount, tails.large_abs, '-', 'LineWidth', 1.2, ...
                     'DisplayName', names{j});
        end
        format_tail_axes(axS, sprintf('%s: %d smallest |\\lambda|', S(i).label, tailCount));
        format_tail_axes(axL, sprintf('%s: %d largest |\\lambda|', S(i).label, tailCount));
    end
    lg = legend(nexttile(tl, 1), 'Location', 'best'); lg.Layout.Tile = 'south';
    title(tl, sprintf('Ideal exact deflation tail ablation (total rank %d)', coarseRank));
    exportgraphics(fig, fullfile(outDir, 'spectrum_tail_ablation.pdf'), 'ContentType', 'vector');
    close(fig);
end

function format_tail_axes(ax, titleText)
    grid(ax, 'on'); box(ax, 'on'); xlabel(ax, 'absolute-tail rank');
    ylabel(ax, '|\lambda|'); title(ax, titleText, 'Interpreter', 'tex');
end

function warn_negative(lambda, label)
    scale = max(abs(lambda));
    bad = lambda < -1e-10 * max(scale, 1);
    if any(bad)
        warning('plot_ard_training_spectra:negativeEigenvalue', ...
                '%s has %d materially negative eigenvalues (minimum %.3e).', ...
                label, sum(bad), min(lambda));
    end
end
