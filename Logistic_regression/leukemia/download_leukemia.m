function download_leukemia()
%DOWNLOAD_LEUKEMIA  Fetch the LIBSVM 'leu' dataset -> data/leu.
%   Downloads leu.bz2 from the LIBSVM binary datasets page and decompresses it
%   with bzip2. The result is a LIBSVM-format text file with 7129 gene-
%   expression features (n=38). No-ops if data/leu already exists.
%
%   Requires the system 'bzip2' utility (present on Linux/WSL/macOS).
%
%   See also load_libsvm, run_logistic_benchmark.

    fileName = 'leu';
    url = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/', ...
           fileName, '.bz2'];
    download_libsvm_bz2(fileName, url);
end
