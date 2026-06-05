function download_protein()
%DOWNLOAD_PROTEIN  Fetch the UCI Protein Tertiary Structure data -> data/protein.csv.
%   Downloads the UCI dataset 265 zip, reads CASP.csv (header row auto-skipped)
%   whose first column is the RMSD target and columns 2..10 are features
%   F1..F9. Reorders the target to the LAST column and writes a header-less
%   numeric CSV matching load_dataset_csv_or_mat's convention.
%
%   No-ops if data/protein.csv already exists. Uses only base MATLAB.
%
%   See also load_dataset_csv_or_mat, plot_kernel_spectrum.

    thisDir = fileparts(mfilename('fullpath'));
    dataDir = fullfile(thisDir, 'data');
    outFile = fullfile(dataDir, 'protein.csv');
    if exist(outFile, 'file') == 2
        fprintf('protein.csv already present: %s\n', outFile);
        return;
    end

    url = ['https://archive.ics.uci.edu/static/public/265/' ...
           'physicochemical+properties+of+protein+tertiary+structure.zip'];
    tmp = tempname;
    mkdir(tmp);
    cleaner = onCleanup(@() rmdir(tmp, 's'));

    zipFile = fullfile(tmp, 'protein.zip');
    fprintf('Downloading %s\n', url);
    websave(zipFile, url);
    unzip(zipFile, tmp);

    csv = find_file(tmp, 'CASP.csv');
    M = readmatrix(csv);                    % header auto-skipped: col1 = RMSD target
    M = M(all(isfinite(M), 2), :);
    M = [M(:, 2:end), M(:, 1)];             % move RMSD target to last column

    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end
    writematrix(M, outFile);
    fprintf('Wrote %s  (%d rows x %d cols, last col = RMSD target)\n', ...
            outFile, size(M, 1), size(M, 2));
end

function f = find_file(root, name)
%FIND_FILE  Recursively locate NAME under ROOT; error if missing.
    d = dir(fullfile(root, '**', name));
    if isempty(d)
        error('download_protein:fileNotFound', '%s not found under %s', name, root);
    end
    f = fullfile(d(1).folder, d(1).name);
end
