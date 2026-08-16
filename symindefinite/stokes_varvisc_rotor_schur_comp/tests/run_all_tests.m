function run_all_tests()
%RUN_ALL_TESTS  Run all variable-viscosity Schur assertion scripts.
    thisDir=fileparts(mfilename('fullpath'));
    tests={'test_varvisc_schur_correctness','test_varvisc_schur_pin', ...
        'test_varvisc_schur_structure','test_varvisc_schur_projector', ...
        'test_varvisc_schur_drift','test_varvisc_schur_registry', ...
        'test_varvisc_schur_hard_case'};
    failed={}; t=tic;
    for i=1:numel(tests)
        fprintf('\n================ %s ================\n',tests{i});
        [ok,msg,report] = run_one(fullfile(thisDir,[tests{i} '.m']));
        if ~ok
            failed{end+1}=sprintf('%s: %s',tests{i},msg); %#ok<AGROW>
            fprintf(2,'FAILED: %s\n',report);
        end
    end
    fprintf('\n%d/%d passed in %.1f s\n',numel(tests)-numel(failed),numel(tests),toc(t));
    if ~isempty(failed), error('run_all_tests:failures','%s',strjoin(failed,newline)); end
end

function [ok,msg,report] = run_one(scriptPath)
    try
        run(scriptPath); ok=true; msg=''; report='';
    catch ME
        ok=false; msg=ME.message;
        report=ME.getReport('extended','hyperlinks','off');
    end
end
