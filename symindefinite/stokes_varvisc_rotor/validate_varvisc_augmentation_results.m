function report = validate_varvisc_augmentation_results(results_root)
%VALIDATE_VARVISC_AUGMENTATION_RESULTS Structural checks and measured accuracy.
% Numerical failures remain in the saved evidence; they do not get filtered.
    [stats,cfg] = varvisc_load_benchmark_stats(results_root);
    T=readtable(fullfile(results_root,'augmentation_results.csv'));
    expected=cfg.params.DEFLAT_SM_EIG*cfg.params.SKETCH_OVERSAMPLE;
    assert(cfg.params.DEFLAT_SM_EIG==500 || contains(results_root,'smoke'));
    assert(cfg.params.SKETCH_OVERSAMPLE==2 && cfg.params.DEFLAT_Q==2);
    assert(all(T.base_rank==expected),'Base Gaussian dimension changed.');
    assert(all(T.total_rank==T.base_rank+T.augment_rank));
    assert(all(T.augment_rank<=T.m) && all(T.augment_rank>=0));
    assert(all(T.basis_built_step==1),'Gaussian physical basis was rebuilt.');
    assert(max(T.vw_orth)<1e-10 && max(T.ww_orth)<1e-10);
    assert(max(T.arnoldi_relation)<1e-10);
    assert(all(T.coarse_min_eig>0));
    assert(all(isfinite(T.algorithm_s)) && all(T.algorithm_s>=0));
    main=T(T.m==0 | T.m==cfg.params.AUGMENT_M,:);
    assert(height(main)==2*numel(stats)*(cfg.params.Tstep-1));
    for c=1:numel(stats)
        B=sortrows(main(strcmp(main.case_name,stats{c}.case_name)&main.m==0,:),'timestep');
        A=sortrows(main(strcmp(main.case_name,stats{c}.case_name)&main.m==cfg.params.AUGMENT_M,:),'timestep');
        assert(isequal(B.factor_setup_s,A.factor_setup_s) && ...
            isequal(B.base_setup_s,A.base_setup_s),'Shared setup attribution differs.');
        for m=[0,cfg.params.AUGMENT_M]
            rows=strcmp(main.case_name,stats{c}.case_name) & main.m==m;
            assert(isequal(sort(main.timestep(rows)),(1:cfg.params.Tstep-1)'));
        end
    end
    sweep=T(strcmp(T.case_name,'bar_rotating_nu_orbiting'),:);
    expected_m=unique([0,cfg.params.AUGMENT_M,cfg.params.AUGMENT_SWEEP]);
    assert(isequal(unique(sweep.m)',expected_m),'Missing dimension-sweep variant.');
    for m=expected_m
        assert(isequal(sort(sweep.timestep(sweep.m==m)),(1:cfg.params.Tstep-1)'));
    end
    assert(all(T.coarse_operator_columns==2*T.total_rank));
    assert(all(T.arnoldi_operator_columns==T.augment_rank));
    expected_time=T.factor_setup_s+T.base_setup_s+T.arnoldi_s+T.coarse_setup_s+T.minres_s;
    assert(max(abs(T.algorithm_s-expected_time))<1e-9,'Algorithm timing components disagree.');
    report=table(sum(T.flag~=0),sum(T.true_relres>1e-6 | ~isfinite(T.true_relres)), ...
        sum(T.err>1e-5 | ~isfinite(T.err)),sum(T.augment_rank<T.m), ...
        'VariableNames',{'nonconverged','physical_residual_failures','solution_error_failures','short_augmentations'});
    writetable(report,fullfile(results_root,'augmentation_validation.csv'));
    fprintf('Augmentation structural checks PASS (%d rows). Accuracy / shortfall counts:\n',height(T));
    disp(report);
end
