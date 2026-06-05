function download_duke()
%DOWNLOAD_DUKE  Fetch the LIBSVM 'duke.tr' dataset -> data/duke.tr.
%   Downloads duke.tr.bz2 (Duke breast-cancer training split) from the LIBSVM
%   binary datasets page and decompresses it with bzip2 (7129 gene-expression
%   features, n=38). No-ops if data/duke.tr already exists. Requires 'bzip2'.
%
%   See also load_libsvm, run_logistic_benchmark.

    fileName = 'duke.tr';
    url = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/', ...
           fileName, '.bz2'];
    download_libsvm_bz2(fileName, url);
end
