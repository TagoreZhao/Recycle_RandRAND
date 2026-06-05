function download_libsvm_bz2(fileName, url)
%DOWNLOAD_LIBSVM_BZ2  Download and decompress a .bz2 LIBSVM file into data/.
%   DOWNLOAD_LIBSVM_BZ2(FILENAME, URL) saves URL (a .bz2 file) to
%   data/FILENAME.bz2 relative to the calling script's folder, then runs
%   'bzip2 -df' to produce data/FILENAME. No-ops if data/FILENAME already
%   exists. Requires the system 'bzip2' utility.
%
%   Shared helper for the per-dataset download_*.m scripts.
%
%   See also load_libsvm, run_logistic_benchmark.

    thisDir = fileparts(mfilename('fullpath'));
    dataDir = fullfile(thisDir, 'data');
    outFile = fullfile(dataDir, fileName);

    if exist(outFile, 'file') == 2
        fprintf('%s already present: %s\n', fileName, outFile);
        return;
    end
    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end

    bzFile = [outFile, '.bz2'];
    fprintf('Downloading %s\n  -> %s\n', url, bzFile);
    websave(bzFile, url, weboptions('Timeout', 300));

    fprintf('Decompressing %s\n', bzFile);
    [status, out] = system(sprintf('bzip2 -df "%s"', bzFile));
    if status ~= 0
        error('download_libsvm_bz2:bzip2', ...
              'bzip2 failed (status %d): %s', status, out);
    end
    if exist(outFile, 'file') ~= 2
        error('download_libsvm_bz2:missing', ...
              'Expected %s after decompression but it is missing.', outFile);
    end
    fprintf('Wrote %s\n', outFile);
end
