function download_rcv1()
%DOWNLOAD_RCV1  Fetch the LIBSVM 'rcv1_train.binary' dataset -> data/rcv1_train.binary.
%   Downloads rcv1_train.binary.bz2 from the LIBSVM binary datasets page and
%   decompresses it with bzip2 (47236 sparse text features, n=20242).
%   No-ops if data/rcv1_train.binary already exists. Requires system 'bzip2'.
%
%   See also load_libsvm, run_logistic_benchmark.

    fileName = 'rcv1_train.binary';
    url = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/', ...
           fileName, '.bz2'];
    download_libsvm_bz2(fileName, url);
end
