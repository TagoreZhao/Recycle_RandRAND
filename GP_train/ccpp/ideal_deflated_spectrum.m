function lambdaOut = ideal_deflated_spectrum(lambda, mode, rankTotal, tau)
%IDEAL_DEFLATED_SPECTRUM  Spectrum for exact tail deflation on an SPD matrix.
%   Selected eigenvalues of P^(1/2) A P^(1/2) become tau.

    lambdaOut = real(lambda(:));
    if any(lambdaOut <= 0) || any(~isfinite(lambdaOut))
        error('ideal_deflated_spectrum:notSPD', ...
              'The idealized PCG diagnostic requires positive finite eigenvalues.');
    end
    if rankTotal < 0 || rankTotal ~= floor(rankTotal) || rankTotal > numel(lambdaOut)
        error('ideal_deflated_spectrum:badRank', 'rankTotal is invalid.');
    end
    if ~(isscalar(tau) && tau > 0 && isfinite(tau))
        error('ideal_deflated_spectrum:badTau', 'tau must be positive and finite.');
    end

    [~, asc] = sort(abs(lambdaOut), 'ascend');
    switch mode
        case 'none'
            idx = zeros(0, 1);
        case 'small'
            idx = asc(1:rankTotal);
        case 'large'
            idx = asc(end-rankTotal+1:end);
        case 'both'
            nSmall = floor(rankTotal / 2);
            nLarge = rankTotal - nSmall;
            idx = [asc(1:nSmall); asc(end-nLarge+1:end)];
        otherwise
            error('ideal_deflated_spectrum:badMode', 'Unknown mode: %s', mode);
    end
    lambdaOut(idx) = tau;
end
