function run_all_tests()
%RUN_ALL_TESTS  Run every assertion script in this folder, in dependency order.
%
%   Ordered so the cheapest, most fundamental gate fails first: if the Schur
%   reduction does not reproduce K\b, nothing downstream is meaningful.
%
%   This is a FUNCTION, not a script, on purpose: each test script begins with
%   `clear`, which would wipe the runner's own bookkeeping if they shared a
%   workspace.  run_one() gives each test its own scope to clear.
%
%   Run:  cd tests; run_all_tests

    thisFileDir = fileparts(mfilename('fullpath'));

    tests = { ...
        'test_schur_correctness', ...   % the gate
        'test_pin_handling', ...
        'test_schur_incremental', ...
        'test_sketch_basis', ...
        'test_baseline_drift', ...
        'test_registry_smoke'};

    nfail   = 0;
    results = cell(numel(tests), 1);
    t_all   = tic;

    for i = 1:numel(tests)
        fprintf('\n================ %s ================\n', tests{i});
        t0 = tic;
        [ok, msg] = run_one(fullfile(thisFileDir, [tests{i} '.m']));
        if ok
            results{i} = sprintf('PASS  %-26s (%.1f s)', tests{i}, toc(t0));
        else
            nfail = nfail + 1;
            results{i} = sprintf('FAIL  %-26s %s', tests{i}, msg);
            fprintf(2, 'FAILED: %s\n', msg);
        end
    end

    fprintf('\n\n======================= SUMMARY =======================\n');
    for i = 1:numel(results)
        fprintf('  %s\n', results{i});
    end
    fprintf('------------------------------------------------------\n');
    fprintf('  %d/%d passed in %.1f s\n', ...
            numel(tests) - nfail, numel(tests), toc(t_all));

    if nfail > 0
        error('run_all_tests:failures', '%d test script(s) failed.', nfail);
    end
end

%==========================================================================
function [ok, msg] = run_one(script_path)
%RUN_ONE  Execute one test script in a throwaway workspace.
    try
        run(script_path);
        ok  = true;
        msg = '';
    catch ME
        ok  = false;
        msg = ME.message;
    end
end
