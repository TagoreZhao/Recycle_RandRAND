function download_kin40k()
%DOWNLOAD_KIN40K  Fetch the kin40k GP-benchmark data -> data/kin40k.csv.
%   Downloads the four ASCII files (train/test data + labels) from the
%   trungngv/fgp repository, stacks train and test, and writes a header-less
%   numeric CSV [X y] with 8 robot-arm input features and the target as the
%   LAST column, matching load_dataset_csv_or_mat's convention.
%
%   No-ops if data/kin40k.csv already exists. Uses only base MATLAB.
%
%   See also load_dataset_csv_or_mat, plot_kernel_spectrum.

    thisDir = fileparts(mfilename('fullpath'));
    dataDir = fullfile(thisDir, 'data');
    outFile = fullfile(dataDir, 'kin40k.csv');
    if exist(outFile, 'file') == 2
        fprintf('kin40k.csv already present: %s\n', outFile);
        return;
    end

    base  = 'https://raw.githubusercontent.com/trungngv/fgp/master/data/kin40k/';
    names = {'kin40k_train_data.asc', 'kin40k_train_labels.asc', ...
             'kin40k_test_data.asc',  'kin40k_test_labels.asc'};
    tmp = tempname;
    mkdir(tmp);
    cleaner = onCleanup(@() rmdir(tmp, 's'));

    for i = 1:numel(names)
        fprintf('Downloading %s\n', names{i});
        websave(fullfile(tmp, names{i}), [base, names{i}]);
    end

    Xtr = load(fullfile(tmp, 'kin40k_train_data.asc'));    % whitespace-delimited ASCII
    ytr = load(fullfile(tmp, 'kin40k_train_labels.asc'));
    Xte = load(fullfile(tmp, 'kin40k_test_data.asc'));
    yte = load(fullfile(tmp, 'kin40k_test_labels.asc'));

    X = [Xtr; Xte];
    y = [ytr(:); yte(:)];
    M = [X, y];
    M = M(all(isfinite(M), 2), :);

    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end
    writematrix(M, outFile);
    fprintf('Wrote %s  (%d rows x %d cols, last col = target)\n', ...
            outFile, size(M, 1), size(M, 2));
end
