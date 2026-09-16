function run_varvisc_gaussian_refresh_benchmark(mode, seed, case_index)
%RUN_VARVISC_GAUSSIAN_REFRESH_BENCHMARK Paired q=2 recycle / q=1 rebuild.
% ('full') runs 3 seeds x 3 cases x 60 timesteps; ('smoke') uses 3 steps.
% ('full',seed,case_index) checkpoints one job; ('finalize') only renders.
    if nargin<1, mode='full'; end
    mode = validatestring(mode,{'full','smoke','finalize'});
    here = fileparts(mfilename('fullpath'));
    addpath(here,fileparts(fileparts(here)));
    params = varvisc_default_benchmark_params();
    params.h0=.05; params.SOLVER_PROFILE='gaussian_refresh';
    params.AUGMENT_ENABLED=false;
    seeds=1:3;
    cases={'bar_rotating_nu_orbiting','disk_translating_nu_wake','disk_static_nu_const'};
    root=fullfile(here,'benchmark_varvisc_gaussian_refresh');
    if strcmp(mode,'smoke')
        params.h0=.16; params.Tstep=4; params.DEFLAT_SM_EIG=8;
        root=[root '_smoke'];
    end
    selected_seeds=seeds; selected_cases=1:3;
    if nargin>=2
        validateattributes(seed,{'numeric'},{'scalar','integer','>=',1,'<=',3});
        selected_seeds=seed;
    end
    if nargin>=3
        validateattributes(case_index,{'numeric'},{'scalar','integer','>=',1,'<=',3});
        selected_cases=case_index;
    end
    config=struct('schema_version',1,'params',params,'seeds',seeds, ...
        'cases',{cases},'mesh_seed',1,'physical_Tmax',1.2, ...
        'matlab_version',version,'compute_threads',maxNumCompThreads);
    if ~isfolder(root), mkdir(root); end
    configfile=fullfile(root,'experiment_config.mat');
    if isfile(configfile)
        prior=load(configfile,'config');
        assert(isequaln(prior.config,config),'varvisc:configMismatch', ...
            'Saved experiment configuration differs; use a fresh output directory.');
    else
        assert(~strcmp(mode,'finalize'),'No saved experiment to finalize.');
        save(configfile,'config');
        fid=fopen(fullfile(root,'experiment_config.json'),'w');
        assert(fid>=0); cleanup=onCleanup(@()fclose(fid));
        fwrite(fid,jsonencode(config)); clear cleanup;
    end
    if strcmp(mode,'finalize')
        varvisc_analyze_gaussian_refresh(root); return;
    end

    meshfile=fullfile(root,'mesh.mat');
    if isfile(meshfile)
        data=load(meshfile,'msh'); msh=data.msh;
    else
        rng(config.mesh_seed);
        msh=src.discretization.build_channel_mesh_pde(params.h0,0,4,0,1,{'rect_right'});
        save(meshfile,'msh');
    end
    N=msh.N;
    left=find(msh.rect_left);
    bnodes=unique([left;find(msh.rect_top);find(msh.rect_bottom)]);
    y=msh.p(bnodes,2); ux=zeros(numel(bnodes),1); inlet=ismember(bnodes,left);
    ux(inlet)=4*y(inlet).*(1-y(inlet));
    dofs=[bnodes;N+bnodes]; vals=[ux;zeros(numel(bnodes),1)];
    [~,pin]=max(msh.p(:,1));
    geo=struct('x1',0,'x2',4,'y1',0,'y2',1,'xc',2,'yc',.5, ...
        'h0',params.h0,'Tmax',config.physical_Tmax);
    definitions=varvisc_define_case_list(params.dt);
    names=cellfun(@(c)c.name,definitions,'UniformOutput',false);
    for s=selected_seeds
        for j=selected_cases
            folder=fullfile(root,sprintf('seed_%d',s),cases{j});
            checkpoint=fullfile(folder,'solver_stats.mat');
            identity=struct('config',config,'seed',s,'case_name',cases{j});
            if isfile(checkpoint)
                saved=load(checkpoint,'identity');
                assert(isequaln(saved.identity,identity),'Checkpoint identity mismatch.');
                fprintf('[resume] seed %d, %s\n',s,cases{j}); continue;
            end
            if ~isfolder(folder), mkdir(folder); end
            % Same initial RNG/mesh/physics for every seed; only Omega varies.
            rng(1);
            definition=definitions{strcmp(names,cases{j})};
            mc=definition.factory(geo);
            cfg=struct('mesh',msh,'nu_fun',mc.nu_fun,'h0',params.h0, ...
                'velbc_fun',@(t)struct('dofs',dofs,'vals',vals), ...
                'motion_fun',mc.motion_fun,'pin_node',pin,'pin_val',0, ...
                'case_name',cases{j},'geometry','stokes_varvisc_rotor');
            pp=params; pp.PAIRED_SEED=s; pp.solvers=varvisc_define_solver_list(pp);
            fprintf('\n[paired] seed %d, %s, h=%.3f, width=%d\n', ...
                s,cases{j},params.h0,params.DEFLAT_SM_EIG*params.SKETCH_OVERSAMPLE);
            st=src.stokes.solve_stokes_varvisc(cfg,pp,folder);
            st.case_name=cases{j}; st.dt=params.dt;
            pending=fullfile(folder,'solver_stats.pending.mat');
            save(pending,'st','identity','-v7'); movefile(pending,checkpoint,'f');
        end
    end
    if nargin<2, varvisc_analyze_gaussian_refresh(root); end
    fprintf('[paired] output: %s\n',root);
end
