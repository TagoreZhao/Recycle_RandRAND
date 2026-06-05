function download_gisette()
%DOWNLOAD_GISETTE  Fetch the LIBSVM 'gisette_scale' dataset -> data/gisette_scale.
%   Downloads gisette_scale.bz2 from the LIBSVM binary datasets page and
%   decompresses it with bzip2 (5000 features, n=6000, values in [-1,1]).
%   No-ops if data/gisette_scale already exists. Requires system 'bzip2'.
%
%   See also load_libsvm, run_logistic_benchmark.

    fileName = 'gisette_scale';
    url = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/', ...
           fileName, '.bz2'];
    download_libsvm_bz2(fileName, url);
end
