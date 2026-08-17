% TEST_VARVISC_SCHUR_EXTRACT  Saved system, spectrum, metadata, and PNG agree.

thisDir = fileparts(mfilename('fullpath'));
addpath(fileparts(thisDir));
add_varvisc_schur_paths();

EXTRACT_CASE_NAME = 'bar_rotating_nu_orbiting';
EXTRACT_H0 = 0.1;
EXTRACT_STEP = 2;
EXTRACT_OUTPUT_DIR = tempname;
mkdir(EXTRACT_OUTPUT_DIR);
cleanupOutput = onCleanup(@() rmdir(EXTRACT_OUTPUT_DIR, 's'));

varvisc_schur_extract_examples;

assert(exist(matFile, 'file') == 2, 'Extractor did not create the MAT file.');
assert(exist(plotFile, 'file') == 2, 'Extractor did not create the spectrum PNG.');
plotInfo = dir(plotFile);
assert(plotInfo.bytes > 0, 'Spectrum PNG is empty.');

saved = load(matFile);
required = {'S','rhs_S','y_ref','eigenvalues','keep','meta'};
assert(all(isfield(saved, required)), 'Saved artifact is missing required fields.');
assert(isequal(size(saved.S), [saved.meta.nS saved.meta.nS]), ...
    'Saved S dimensions disagree with metadata.');
assert(numel(saved.rhs_S) == saved.meta.nS, 'rhs_S has the wrong length.');
assert(numel(saved.y_ref) == saved.meta.nS, 'y_ref has the wrong length.');
assert(numel(saved.keep) == saved.meta.nS_full, 'keep has the wrong length.');

relres = norm(saved.S*saved.y_ref - saved.rhs_S) / ...
    max(norm(saved.rhs_S), eps);
assert(relres < 1e-10, 'Reloaded Schur residual is %.3e.', relres);
assert(isreal(saved.eigenvalues), 'Saved eigenvalues are not real.');
assert(all(diff(saved.eigenvalues) >= 0), 'Saved eigenvalues are not ordered.');
assert(saved.eigenvalues(1) > 0, 'Saved spectrum is not positive.');

expected = sort(real(eig(saved.S)), 'ascend');
specerr = norm(saved.eigenvalues - expected) / max(norm(expected), eps);
assert(specerr < 1e-13, 'Saved spectrum differs from eig(S) by %.3e.', specerr);
assert(abs(saved.meta.lambda_min - saved.eigenvalues(1)) <= ...
    10*eps(saved.eigenvalues(1)), 'meta.lambda_min is inconsistent.');
assert(abs(saved.meta.lambda_max - saved.eigenvalues(end)) <= ...
    10*eps(saved.eigenvalues(end)), 'meta.lambda_max is inconsistent.');
expectedCond = saved.eigenvalues(end) / saved.eigenvalues(1);
assert(abs(saved.meta.condition_number - expectedCond) <= ...
    10*eps(expectedCond), 'meta.condition_number is inconsistent.');

fprintf('test_varvisc_schur_extract: artifact and exact spectrum passed\n');
