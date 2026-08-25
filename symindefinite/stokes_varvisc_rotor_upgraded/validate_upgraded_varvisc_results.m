function validate_upgraded_varvisc_results(results_root)
%VALIDATE_UPGRADED_VARVISC_RESULTS Check the retained comparison artifact set.
    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(results_root)
        results_root = fullfile(here, 'benchmark_varvisc_upgraded');
    elseif ~isfolder(results_root)
        results_root = fullfile(here, results_root);
    end

    data_file = fullfile(results_root, 'all_results.csv');
    config_file = fullfile(results_root, 'run_config.mat');
    assert(isfile(data_file) && isfile(config_file), ...
        'Missing all_results.csv or run_config.mat in %s.', results_root);
    T = readtable(data_file);
    S = load(config_file, 'cfg_out');
    cfg = S.cfg_out;
    keys = cellstr(string(cfg.solver_keys(:)));
    cases = string(cfg.case_names(:));
    expected_steps = cfg.params.Tstep - 1;

    assert(numel(keys) == 11, 'Expected the complete 11-solver registry.');
    assert(height(T) == numel(cases) * expected_steps, ...
        'Expected %d result rows, found %d.', ...
        numel(cases) * expected_steps, height(T));
    assert(isequal(unique(string(T.case_name), 'stable'), cases), ...
        'Result cases or order do not match run_config.');

    iteration_columns = strcat(keys, '_its');
    flag_columns = strcat(keys, '_flag');
    assert(all(ismember([iteration_columns; flag_columns], ...
        T.Properties.VariableNames)), 'Solver iteration/flag columns are incomplete.');
    assert(all(isfinite(T{:, iteration_columns}), 'all'), ...
        'Non-finite iteration count found.');
    assert(all(T{:, iteration_columns} >= 0, 'all'), ...
        'Negative iteration count found.');
    assert(all(isfinite(T.backslash_relres)) && ...
        max(T.backslash_relres) < 1e-8, 'Direct residual acceptance failed.');
    assert(all(isfinite(T.constraint_res)) && ...
        max(T.constraint_res) < 1e-8, 'Constraint residual acceptance failed.');

    expected_nC = containers.Map( ...
        {'current_channel_ar4', 'mixer_circle_four_blade'}, {6, 18});
    for k = 1:numel(cases)
        cname = char(cases(k));
        rows = string(T.case_name) == cases(k);
        assert(all(T.nC(rows) == expected_nC(cname)), ...
            'Unexpected constraint count for %s.', cname);
        assert(isfile(fullfile(results_root, 'iteration_vs_timestep', ...
            [cname '.png'])), 'Missing iteration plot for %s.', cname);
        assert(isfile(fullfile(results_root, 'iteration_vs_timestep', ...
            [cname '_linear.png'])), ...
            'Missing linear-scale iteration plot for %s.', cname);
    end
    assert(isfile(fullfile(results_root, 'summary_plots', ...
        'all_cases_comparison.png')), 'Missing combined comparison plot.');
    assert(isfile(fullfile(results_root, 'summary_plots', ...
        'all_cases_comparison_linear.png')), ...
        'Missing combined linear-scale comparison plot.');

    if ~contains(results_root, 'smoke')
        assert(cfg.params.h0 == 0.05 && cfg.params.dt == 0.02 && ...
            cfg.params.Tstep == 61 && cfg.physical_Tmax == 1.2, ...
            'Full-run mesh/time parameters do not match the promoted values.');
    end
    fprintf(['validate_upgraded_varvisc_results: PASS (%d rows, %d cases, ' ...
        '%d solvers)\n'], height(T), numel(cases), numel(keys));
end
