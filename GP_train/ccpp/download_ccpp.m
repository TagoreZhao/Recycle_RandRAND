function download_ccpp()
%DOWNLOAD_CCPP  Fetch the UCI Combined Cycle Power Plant data -> data/ccpp.csv.
%   Downloads the UCI dataset 294 zip, reads the first sheet of
%   Folds5x2_pp.xlsx (columns AT, V, AP, RH, PE), and writes a header-less
%   numeric CSV. The net hourly energy output PE is the regression target and
%   is already the LAST column, matching load_dataset_csv_or_mat's convention.
%
%   No-ops if data/ccpp.csv already exists. Uses only base MATLAB.
%
%   See also load_dataset_csv_or_mat, plot_kernel_spectrum.

    thisDir = fileparts(mfilename('fullpath'));
    dataDir = fullfile(thisDir, 'data');
    outFile = fullfile(dataDir, 'ccpp.csv');
    if exist(outFile, 'file') == 2
        fprintf('ccpp.csv already present: %s\n', outFile);
        return;
    end

    url = 'https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip';
    tmp = tempname;
    mkdir(tmp);
    cleaner = onCleanup(@() rmdir(tmp, 's'));

    zipFile = fullfile(tmp, 'ccpp.zip');
    fprintf('Downloading %s\n', url);
    websave(zipFile, url);
    unzip(zipFile, tmp);

    xlsx = find_file(tmp, 'Folds5x2_pp.xlsx');
    M = readmatrix(xlsx, 'Sheet', 1);       % text header row -> NaN, dropped below
    M = M(all(isfinite(M), 2), :);

    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end
    writematrix(M, outFile);
    fprintf('Wrote %s  (%d rows x %d cols, last col = PE target)\n', ...
            outFile, size(M, 1), size(M, 2));
end

function f = find_file(root, name)
%FIND_FILE  Recursively locate NAME under ROOT; error if missing.
    d = dir(fullfile(root, '**', name));
    if isempty(d)
        error('download_ccpp:fileNotFound', '%s not found under %s', name, root);
    end
    f = fullfile(d(1).folder, d(1).name);
end
