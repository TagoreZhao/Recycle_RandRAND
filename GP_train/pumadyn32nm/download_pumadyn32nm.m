function download_pumadyn32nm()
%DOWNLOAD_PUMADYN32NM  Fetch the pumadyn-32nm GP-benchmark data -> data/pumadyn32nm.csv.
%   Downloads the four ASCII files (train/test data + labels) from the
%   trungngv/fgp repository, stacks train and test, and writes a header-less
%   numeric CSV [X y] with 32 robot-arm input features and the target as the
%   LAST column, matching load_dataset_csv_or_mat's convention.
%
%   No-ops if data/pumadyn32nm.csv already exists. Uses only base MATLAB.
%
%   See also load_dataset_csv_or_mat, plot_kernel_spectrum.

    thisDir = fileparts(mfilename('fullpath'));
    dataDir = fullfile(thisDir, 'data');
    outFile = fullfile(dataDir, 'pumadyn32nm.csv');
    if exist(outFile, 'file') == 2
        fprintf('pumadyn32nm.csv already present: %s\n', outFile);
        return;
    end

    base  = 'https://raw.githubusercontent.com/trungngv/fgp/master/data/pumadyn32nm/';
    names = {'pumadyn32nm_train_data.asc', 'pumadyn32nm_train_labels.asc', ...
             'pumadyn32nm_test_data.asc',  'pumadyn32nm_test_labels.asc'};
    tmp = tempname;
    mkdir(tmp);
    cleaner = onCleanup(@() rmdir(tmp, 's'));

    for i = 1:numel(names)
        fprintf('Downloading %s\n', names{i});
        websave(fullfile(tmp, names{i}), [base, names{i}]);
    end

    Xtr = load(fullfile(tmp, 'pumadyn32nm_train_data.asc'));    % whitespace-delimited ASCII
    ytr = load(fullfile(tmp, 'pumadyn32nm_train_labels.asc'));
    Xte = load(fullfile(tmp, 'pumadyn32nm_test_data.asc'));
    yte = load(fullfile(tmp, 'pumadyn32nm_test_labels.asc'));

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
