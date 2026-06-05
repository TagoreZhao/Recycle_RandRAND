function [X, y] = load_dataset_csv_or_mat(filePath)
%LOAD_DATASET_CSV_OR_MAT  Load a regression dataset from a .csv or .mat file.
%   [X, y] = LOAD_DATASET_CSV_OR_MAT(FILEPATH) reads a single numeric data
%   matrix and splits it into features X and target y, assuming the LAST
%   column is the target and all preceding columns are features.
%
%   Inputs
%     filePath : char/string. Path to a .csv or .mat file. A .mat file must
%                contain exactly one numeric matrix variable.
%
%   Outputs
%     X        : n-by-d matrix of features (all but the last column).
%     y        : n-by-1 vector of targets (the last column).
%
%   Example
%     [X, y] = load_dataset_csv_or_mat('data/elevators.csv');
%
%   Implementation notes
%     - .csv is read with readmatrix (no header assumed).
%     - .mat is loaded and the sole numeric matrix variable is used; an
%       error is raised if there are zero or multiple candidate variables.

    if nargin < 1 || isempty(filePath)
        error('load_dataset_csv_or_mat:noPath', 'A file path is required.');
    end
    filePath = char(filePath);
    if exist(filePath, 'file') ~= 2
        error('load_dataset_csv_or_mat:fileNotFound', ...
              'File not found: %s', filePath);
    end

    [~, ~, ext] = fileparts(filePath);
    switch lower(ext)
        case '.csv'
            M = readmatrix(filePath);
        case '.mat'
            M = pick_matrix_from_mat(filePath);
        otherwise
            error('load_dataset_csv_or_mat:badExt', ...
                  'Unsupported extension "%s" (expected .csv or .mat).', ext);
    end

    if ~isnumeric(M) || ~ismatrix(M) || size(M, 2) < 2
        error('load_dataset_csv_or_mat:badMatrix', ...
              'Expected a numeric matrix with at least 2 columns.');
    end
    if any(~isfinite(M(:)))
        error('load_dataset_csv_or_mat:nonFinite', ...
              'Data contains NaN or Inf values.');
    end

    X = M(:, 1:end-1);
    y = M(:, end);
end

%% --------- local helpers ---------
function M = pick_matrix_from_mat(filePath)
%PICK_MATRIX_FROM_MAT  Return the single numeric matrix stored in a .mat file.
    s = load(filePath);
    names = fieldnames(s);
    isCandidate = false(numel(names), 1);
    for k = 1:numel(names)
        v = s.(names{k});
        isCandidate(k) = isnumeric(v) && ismatrix(v) && size(v, 2) >= 2;
    end
    candidates = names(isCandidate);
    if isempty(candidates)
        error('load_dataset_csv_or_mat:noMatrix', ...
              'No numeric matrix (>=2 columns) found in %s', filePath);
    end
    if numel(candidates) > 1
        error('load_dataset_csv_or_mat:ambiguous', ...
              'Multiple numeric matrices found in %s: %s', ...
              filePath, strjoin(candidates, ', '));
    end
    M = s.(candidates{1});
end
