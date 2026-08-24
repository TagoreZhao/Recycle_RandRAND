% TEST_VARVISC_SCHUR_EXTREME_EIGENVALUES  Per-system spectra and plots.
clear; clc;
thisDir = fileparts(mfilename('fullpath')); addpath(fileparts(thisDir));
add_varvisc_schur_paths(); rng(11);

p = make_varvisc_schur_params();
assert(~p.PLOT_EXTREME_EIGENVALUES, ...
       'Extreme-eigenvalue plotting must be opt-in by default.');
p.PLOT_EXTREME_EIGENVALUES = true;
p.h0 = 0.2;
p.max_steps = 2;
p.sm_eig = 2;
p.lg_eig = 2;
p.sketch_oversampling = 1;
p.SPECTRAL_RITZ_TOL = 1e-9;
p.standalone_variants = struct( ...
    'name','deflate_shared_small','design','shared_small');

cfg = varvisc_schur_make_cfg('disk_static_nu_const',p,[]);
A = solve_varvisc_schur_sequence(cfg,p,'');
assert(isequal(A.dense_materialized_step,1), ...
       'Extreme plots materialized a dense operator after the frozen factor.');
for keyIndex = 1:numel(A.solver_keys)
    key = A.solver_keys{keyIndex};
    lambdaMin = A.system_lambda_min.(key);
    lambdaMax = A.system_lambda_max.(key);
    assert(all(isfinite(lambdaMin)) && all(isfinite(lambdaMax)), ...
           'Missing extreme eigenvalues for %s.',key);
    assert(all(lambdaMin > 0) && all(lambdaMin <= lambdaMax), ...
           'Invalid extreme eigenvalues for %s.',key);
    assert(max(abs(A.system_kappa.(key)-lambdaMax./lambdaMin)) < 1e-12, ...
           'Preconditioned condition numbers are inconsistent for %s.',key);
    assert(all(A.system_spectrum_flag.(key) == 0), ...
           'The iterative eigensolver did not converge for %s.',key);
    if ~strcmp(key,'pcg_unprec')
        assert(~any(A.system_spectrum_is_exact.(key)), ...
               'A preconditioned system unexpectedly used a dense eigensolve.');
    end
    if ~strcmp(key,'pcg_unprec')
        assert(all(isfinite(A.system_spectrum_residual.(key))), ...
               'Missing Ritz residuals for %s.',key);
        assert(max(A.system_spectrum_residual.(key)) < 1e-6, ...
               'Extreme Ritz residual is too large for %s.',key);
    end
end
assert(max(abs(A.system_lambda_min.chol-1)) < 1e-7, ...
       'Frozen Cholesky smallest eigenvalues are not one.');
assert(max(abs(A.system_lambda_max.chol-1)) < 1e-7, ...
       'Frozen Cholesky largest eigenvalues are not one.');

ctx = varvisc_schur_context_init(cfg,p);
st = varvisc_schur_step_operator(ctx,p.dt,zeros(ctx.nU,1));
exactValues = sort(real(eig(st.to_dense())),'ascend');
assert(abs(A.system_lambda_min.pcg_unprec(1)-exactValues(1)) < ...
       1e-6*exactValues(1), 'Raw smallest eigenvalue estimate is inaccurate.');
assert(abs(A.system_lambda_max.pcg_unprec(1)-exactValues(end)) < ...
       1e-6*exactValues(end), 'Raw largest eigenvalue estimate is inaccurate.');

outputDir = tempname; mkdir(outputDir);
cleanupOutput = onCleanup(@() rmdir(outputDir,'s'));
opts = varvisc_schur_fig_defaults();
write_varvisc_schur_case_outputs(outputDir,A,opts);
smallPlot = fullfile(outputDir,'plot_smallest_eigenvalues.png');
largePlot = fullfile(outputDir,'plot_largest_eigenvalues.png');
kappaPlot = fullfile(outputDir,'plot_preconditioned_kappa.png');
assert(exist(smallPlot,'file') == 2 && dir(smallPlot).bytes > 0, ...
       'Smallest-eigenvalue plot was not written.');
assert(exist(largePlot,'file') == 2 && dir(largePlot).bytes > 0, ...
       'Largest-eigenvalue plot was not written.');
assert(exist(kappaPlot,'file') == 2 && dir(kappaPlot).bytes > 0, ...
       'Preconditioned-condition-number plot was not written.');

write_varvisc_schur_all_results(outputDir,{A});
T = readtable(fullfile(outputDir,'all_results.csv'));
for keyIndex = 1:numel(A.solver_keys)
    key = A.solver_keys{keyIndex};
    requiredColumns = {[key '_lambda_min'],[key '_lambda_max'], ...
        [key '_kappa_prec'],[key '_spectrum_flag'], ...
        [key '_spectrum_residual'],[key '_spectrum_is_exact']};
    assert(all(ismember(requiredColumns,T.Properties.VariableNames)), ...
           'CSV is missing extreme-eigenvalue columns for %s.',key);
end
replot_varvisc_schur(outputDir);
caseDir = fullfile(outputDir,A.case_name);
assert(exist(fullfile(caseDir,'plot_smallest_eigenvalues.png'),'file') == 2, ...
       'Replot did not recreate the smallest-eigenvalue figure.');
assert(exist(fullfile(caseDir,'plot_largest_eigenvalues.png'),'file') == 2, ...
       'Replot did not recreate the largest-eigenvalue figure.');
assert(exist(fullfile(caseDir,'plot_preconditioned_kappa.png'),'file') == 2, ...
       'Replot did not recreate the condition-number figure.');

fprintf('test_varvisc_schur_extreme_eigenvalues: ALL ASSERTIONS PASSED\n');
