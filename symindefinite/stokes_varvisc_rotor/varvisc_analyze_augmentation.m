function varvisc_analyze_augmentation(results_root, compute_spectra)
%VARVISC_ANALYZE_AUGMENTATION Build tidy paired/sweep tables and spectral probes.
% Expensive probes are separate from benchmark timings. Replotting reads the
% resulting artifacts and never calls this analysis routine.
    if nargin < 2, compute_spectra = true; end
    [~,cfg] = varvisc_load_benchmark_stats(results_root);
    rows = [];
    for c = 1:numel(cfg.case_names)
        cname = cfg.case_names{c};
        case_dir = fullfile(results_root,cname);
        files = dir(fullfile(case_dir,'diagnostics','m*_step_*.mat'));
        for j = 1:numel(files)
            data = load(fullfile(files(j).folder,files(j).name),'record');
            r = data.record;
            rows = [rows; result_row(cname,r.step,r)]; %#ok<AGROW>
            for s = 1:numel(r.sweep)
                rows = [rows; result_row(cname,r.step,r.sweep(s))]; %#ok<AGROW>
            end
        end
        if compute_spectra
            snaps = dir(fullfile(case_dir,'snapshots','step_*.mat'));
            for j = 1:numel(snaps)
                data = load(fullfile(snaps(j).folder,snaps(j).name),'snapshot');
                snap = data.snapshot;
                fprintf('[spectra] %s step %d, %d Arnoldi probe vectors\n', ...
                    cname,snap.step,cfg.params.AUGMENT_PROBE_DIM);
                timer = tic;
                dc = decomposition(snap.C,'lu');
                Afun = @(x) dc \ (snap.K * (dc' \ x));
                stream = RandStream('mt19937ar','Seed',19073+snap.step);
                probe = randn(stream,size(snap.K,1),1);
                [W0,a0] = varvisc_build_projected_arnoldi( ...
                    Afun,snap.V,probe,cfg.params.AUGMENT_PROBE_DIM);
                [h0,r0] = varvisc_arnoldi_metrics(snap.V,W0,a0);
                S = [snap.V,snap.W];
                [W1,a1] = varvisc_build_projected_arnoldi( ...
                    Afun,S,probe,cfg.params.AUGMENT_PROBE_DIM);
                [h1,r1] = varvisc_arnoldi_metrics(S,W1,a1);
                spectrum = struct('step',snap.step,'before',r0,'after',r1, ...
                    'method',snap.ritz,'before_health',h0,'after_health',h1, ...
                    'forced_zeros_before',size(snap.V,2), ...
                    'forced_zeros_after',size(S,2),'seconds',toc(timer), ...
                    'requested',cfg.params.AUGMENT_PROBE_DIM, ...
                    'actual_before',size(W0,2),'actual_after',size(W1,2));
                save(fullfile(case_dir,'diagnostics',sprintf('spectrum_%03d.mat',snap.step)), ...
                    'spectrum','-v7');
                write_spectrum_csv(case_dir,spectrum);
            end
        end
    end
    T = sortrows(struct2table(rows),{'case_name','timestep','m'});
    writetable(T,fullfile(results_root,'augmentation_results.csv'));
    summaries = [];
    for c = 1:numel(cfg.case_names)
        cname = cfg.case_names{c};
        Tc = T(strcmp(T.case_name,cname),:);
        B = sortrows(Tc(Tc.m==0,:),'timestep');
        for m = unique(Tc.m)'
            R = sortrows(Tc(Tc.m==m,:),'timestep');
            assert(isequal(R.timestep,B.timestep),'Unpaired sweep rows.');
            valid = R.flag==0 & R.true_relres<=1e-6 & R.err<=1e-5;
            row = struct('case_name',cname,'m',m,'steps',height(R), ...
                'mean_iterations',mean(R.iters),'max_iterations',max(R.iters), ...
                'iteration_saving_percent',100*(1-sum(R.iters)/sum(B.iters)), ...
                'min_total_rank',min(R.total_rank),'max_total_rank',max(R.total_rank), ...
                'max_true_relres',max(R.true_relres),'max_solution_error',max(R.err), ...
                'valid_steps',sum(valid),'short_augmentation_steps',sum(R.augment_rank<m));
            summaries = [summaries;row]; %#ok<AGROW>
        end
    end
    writetable(struct2table(summaries),fullfile(results_root,'augmentation_summary.csv'));
    varvisc_write_augmentation_report(results_root,cfg,T,struct2table(summaries));
end

function row = result_row(cname,step,r)
    row = struct('case_name',cname,'timestep',step,'m',r.info.augment_requested, ...
        'iters',r.iters,'flag',r.flag,'relres',r.relres, ...
        'true_relres',r.true_relres,'err',r.err);
    names = {'base_rank','augment_rank','total_rank','arnoldi_status', ...
        'factor_setup_s','base_setup_s','arnoldi_s','coarse_setup_s','minres_s', ...
        'algorithm_s','arnoldi_operator_columns','coarse_operator_columns', ...
        'minres_operator_columns','operator_columns','coarse_min_eig', ...
        'coarse_max_eig','coarse_condition','vw_orth','ww_orth','arnoldi_relation', ...
        'arnoldi_coupling','vv_orth','leakage_before','leakage_after','basis_built_step'};
    for j=1:numel(names)
        if isfield(r.info,names{j}), row.(names{j})=r.info.(names{j});
        else, row.(names{j})=NaN; end
    end
end

function write_spectrum_csv(case_dir,s)
    T = table();
    for name = {'before','after','method'}
        r = s.(name{1}); n = numel(r.values);
        R = table(repmat(s.step,n,1),repmat(string(name{1}),n,1), ...
            r.values,r.relative_residual,r.converged, ...
            'VariableNames',{'timestep','source','ritz_value','relative_residual','converged'});
        T = [T;R]; %#ok<AGROW>
    end
    writetable(T,fullfile(case_dir,'diagnostics',sprintf('spectrum_%03d.csv',s.step)));
end
