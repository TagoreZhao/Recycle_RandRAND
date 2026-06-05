function [X, y, info] = load_libsvm(filePath, dExpected)
%LOAD_LIBSVM  Load a sparse binary dataset in LIBSVM text format.
%   [X, y, info] = LOAD_LIBSVM(FILEPATH) parses a LIBSVM-format file whose
%   lines look like
%       <label> <index>:<value> <index>:<value> ...
%   and returns a sparse feature matrix X (n-by-d, 1-based indices) and a
%   binary target y in {0,1}. The two distinct labels found in the file are
%   mapped so that the LARGER raw label becomes 1 and the smaller becomes 0
%   (handles +1/-1, 1/2, 0/1, ... uniformly).
%
%   [...] = LOAD_LIBSVM(FILEPATH, DEXPECTED) pads X to DEXPECTED columns when
%   the maximum feature index observed is smaller (useful for train/test
%   files that must share a feature dimension).
%
%   Outputs
%     X    : sparse n-by-d feature matrix.
%     y    : n-by-1 vector in {0,1}.
%     info : struct with fields orig_labels (the two raw label values),
%            n, d, and nnz.
%
%   Implementation notes
%     - Pure base MATLAB (no LIBSVM mex dependency).
%     - Blank lines are skipped; values are read as double.
%
%   See also standardize_features, run_logistic_benchmark.

    if nargin < 1 || isempty(filePath)
        error('load_libsvm:noPath', 'A file path is required.');
    end
    if nargin < 2
        dExpected = [];
    end
    filePath = char(filePath);
    if exist(filePath, 'file') ~= 2
        error('load_libsvm:fileNotFound', 'File not found: %s', filePath);
    end

    lines = readlines(filePath);
    lines = strip(lines);
    lines = lines(strlength(lines) > 0);
    n = numel(lines);
    if n == 0
        error('load_libsvm:empty', 'No data lines found in %s', filePath);
    end

    rawLabels = zeros(n, 1);
    rowCell = cell(n, 1);
    colCell = cell(n, 1);
    valCell = cell(n, 1);

    for i = 1:n
        parts = split(lines(i));            % whitespace-delimited tokens
        rawLabels(i) = str2double(parts(1));
        rest = parts(2:end);
        if isempty(rest)
            continue;                        % all-zero feature row
        end
        idx = double(extractBefore(rest, ":"));
        val = double(extractAfter(rest, ":"));
        rowCell{i} = repmat(i, numel(idx), 1);
        colCell{i} = idx(:);
        valCell{i} = val(:);
    end

    rows = cell2mat(rowCell);
    cols = cell2mat(colCell);
    vals = cell2mat(valCell);
    if any(~isfinite([rows; cols; vals]))
        error('load_libsvm:nonFinite', 'Parsed NaN/Inf in %s', filePath);
    end

    d = max(cols);
    if ~isempty(dExpected)
        d = max(d, dExpected);
    end
    X = sparse(rows, cols, vals, n, d);

    uy = unique(rawLabels);
    if numel(uy) ~= 2
        error('load_libsvm:notBinary', ...
              'Expected exactly 2 labels, found %d in %s.', numel(uy), filePath);
    end
    y = double(rawLabels == max(uy));        % larger raw label -> 1

    info = struct('orig_labels', uy(:).', 'n', n, 'd', d, 'nnz', nnz(X));
end
