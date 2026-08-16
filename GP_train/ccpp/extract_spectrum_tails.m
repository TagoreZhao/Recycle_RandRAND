function tails = extract_spectrum_tails(lambda, count)
%EXTRACT_SPECTRUM_TAILS  Rank signed eigenvalues by absolute magnitude.

    lambda = real(lambda(:));
    if any(~isfinite(lambda))
        error('extract_spectrum_tails:nonFinite', 'Eigenvalues must be finite.');
    end
    if ~(isscalar(count) && count >= 1 && count == floor(count))
        error('extract_spectrum_tails:badCount', 'count must be a positive integer.');
    end
    count = min(count, floor(numel(lambda) / 2));
    [~, order] = sort(abs(lambda), 'ascend');
    smallIdx = order(1:count);
    largeIdx = order(end:-1:end-count+1);

    tails.small = lambda(smallIdx);
    tails.large = lambda(largeIdx);
    tails.small_abs = abs(tails.small);
    tails.large_abs = abs(tails.large);
    tails.small_indices = smallIdx;
    tails.large_indices = largeIdx;
end
