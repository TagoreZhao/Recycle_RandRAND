function download_realsim()
%DOWNLOAD_REALSIM  Fetch the LIBSVM 'real-sim' dataset -> data/real-sim.
%   Downloads real-sim.bz2 from the LIBSVM binary datasets page and
%   decompresses it with bzip2 (20958 sparse text features, n=72309).
%   No-ops if data/real-sim already exists. Requires system 'bzip2'.
%
%   See also load_libsvm, run_logistic_benchmark.

    fileName = 'real-sim';
    url = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/', ...
           fileName, '.bz2'];
    download_libsvm_bz2(fileName, url);
end
