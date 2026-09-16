function run_varvisc_schur_augmented_benchmark(mode,seed,case_index,output_root)
%RUN_VARVISC_SCHUR_AUGMENTED_BENCHMARK Paired frozen Gaussian PCG, m=0 vs m=20.
% ('full',seed,case_index) resumes one job. ('finalize') rebuilds full reports.
% Optional OUTPUT_ROOT supports isolated experiments and smoke finalization.
    if nargin<1, mode='full'; end
    mode=validatestring(mode,{'full','smoke','finalize'});
    paths=add_varvisc_schur_paths(); assert_varvisc_schur_helpers();
    maxNumCompThreads(4);
    root=fullfile(paths.thisDir,'varvisc_schur_augmented');
    if strcmp(mode,'smoke'), root=[root '_smoke']; end
    if nargin>=4 && ~isempty(output_root), root=output_root; end
    file=fullfile(root,'experiment_config.mat');
    if strcmp(mode,'finalize')
        assert(isfile(file),'No completed experiment to finalize.');
        varvisc_schur_analyze_augmentation(root); return;
    end
    params=struct('h0',.05,'dt',.02,'Tstep',61,'sm_eig',500, ...
        'sketch_oversampling',2,'SOLVER_TOL',1e-8,'SOLVER_MAXIT',4000,'AUGMENT_M',20);
    if strcmp(mode,'smoke'), params.h0=.16; params.sm_eig=8; params.max_steps=3; end
    seeds=1:3;
    cases={'bar_rotating_nu_orbiting','disk_translating_nu_wake','disk_static_nu_const'};
    ss=seeds; cc=1:3;
    if nargin>=2 && ~isempty(seed)
        validateattributes(seed,{'numeric'},{'scalar','integer','>=',1,'<=',3}); ss=seed;
    end
    if nargin>=3 && ~isempty(case_index)
        validateattributes(case_index,{'numeric'},{'scalar','integer','>=',1,'<=',3}); cc=case_index;
    end
    config=struct('schema_version',1,'params',params,'cases',{cases},'seeds',seeds, ...
        'mesh_seed',1,'power_count',2,'basis_refresh',Inf,'augmentation',[0 20], ...
        'tau_policy','fixed_lambda_max_S1','initial_guess','zero', ...
        'reference','direct_KKT','solution_error_tol',1e-5, ...
        'matlab_version',version,'compute_threads',maxNumCompThreads);
    if isfile(file)
        old=load(file,'config');
        assert(isequaln(old.config,config),'varviscSchur:configMismatch', ...
            'Experiment configuration differs; use a new output directory.');
    else
        if ~isfolder(root), mkdir(root); end
        save(file,'config');
        fid=fopen(fullfile(root,'experiment_config.json'),'w'); assert(fid>=0);
        cleanup=onCleanup(@()fclose(fid));
        json=config; json.basis_refresh='Inf'; fwrite(fid,jsonencode(json)); clear cleanup;
    end
    meshfile=fullfile(root,'mesh.mat');
    if isfile(meshfile)
        data=load(meshfile,'msh'); msh=data.msh;
    else
        rng(config.mesh_seed); [~,msh]=varvisc_schur_make_cfg(cases{1},params,[]);
        save(meshfile,'msh');
    end
    for s=ss
        for j=cc
            folder=fullfile(root,sprintf('seed_%d',s),cases{j});
            checkpoint=fullfile(folder,'solver_stats.mat');
            identity=struct('config',config,'seed',s,'case_name',cases{j});
            if isfile(checkpoint)
                old=load(checkpoint,'identity');
                assert(isequaln(old.identity,identity),'Checkpoint identity differs.');
                fprintf('[resume] seed %d, %s\n',s,cases{j}); continue;
            end
            if ~isfolder(folder), mkdir(folder); end
            rng(1); cfg=varvisc_schur_make_cfg(cases{j},params,msh);
            pp=params; pp.PAIRED_SEED=s;
            fprintf('\n[Schur augmented] seed %d, %s, width=%d, m=%d\n', ...
                s,cases{j},ceil(params.sm_eig*params.sketch_oversampling),params.AUGMENT_M);
            result=solve_varvisc_schur_augmented(cfg,pp,folder);
            pending=fullfile(folder,'solver_stats.pending.mat');
            save(pending,'result','identity','-v7'); movefile(pending,checkpoint,'f');
        end
    end
    if numel(ss)==3 && numel(cc)==3, varvisc_schur_analyze_augmentation(root); end
    fprintf('[Schur augmented] output: %s\n',root);
end
