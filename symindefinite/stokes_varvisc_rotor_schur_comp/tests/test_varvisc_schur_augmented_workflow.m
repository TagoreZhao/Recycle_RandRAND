function test_varvisc_schur_augmented_workflow()
%TEST_VARVISC_SCHUR_AUGMENTED_WORKFLOW Checkpoint resume and config rejection.
    here=fileparts(mfilename('fullpath')); addpath(fileparts(here));
    root=tempname; mkdir(root); cleanup=onCleanup(@()rmdir(root,'s'));
    run_varvisc_schur_augmented_benchmark('smoke',1,1,root);
    checkpoint=fullfile(root,'seed_1','bar_rotating_nu_orbiting','solver_stats.mat');
    before=load(checkpoint);
    run_varvisc_schur_augmented_benchmark('smoke',1,1,root);
    after=load(checkpoint); assert(isequaln(before,after),'Resume changed a completed result.');
    caught=false;
    try
        run_varvisc_schur_augmented_benchmark('finalize',[],[],root);
    catch exception
        caught=strcmp(exception.identifier,'varviscSchur:incompleteExperiment');
    end
    assert(caught,'Partial experiment was finalized without rejecting missing jobs.');
    file=fullfile(root,'experiment_config.mat'); d=load(file); config=d.config;
    config.params.AUGMENT_M=21; save(file,'config');
    caught=false;
    try
        run_varvisc_schur_augmented_benchmark('smoke',1,1,root);
    catch exception
        caught=strcmp(exception.identifier,'varviscSchur:configMismatch');
    end
    assert(caught,'Configuration mismatch was not rejected.');
    fprintf('PASS: augmented experiment checkpoint resume, partial finalization and config mismatch.\n');
end
