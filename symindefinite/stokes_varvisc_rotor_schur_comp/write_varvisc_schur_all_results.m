function write_varvisc_schur_all_results(results_root, all_stats)
%WRITE_VARVISC_SCHUR_ALL_RESULTS  Master CSV, one row per case and timestep.
    rows = {};
    diagnostics = {'kappa','lambda_min','lambda_max','spectrum_is_exact', ...
        'ReldiffF','RelInitdiffF','ReldiffProbe','RelInitdiffProbe', ...
        'InvRelDiff','coupling_change','A_change','D_change', ...
        'pressure_schur_change','pressure_schur_change_probe','nu_contrast', ...
        'backslash_relres','vel_recovery_err','schur_ref_relres', ...
        'symmetry_res','symmetry_probe_res','chol_flag','nC','nS'};
    for ci = 1:numel(all_stats)
        A = all_stats{ci};
        for n = 1:A.nsteps
            r = struct('case_name',string(A.case_name), ...
                       'geometry',string(A.geometry),'timestep',n);
            for i = 1:numel(A.solver_keys)
                key = A.solver_keys{i};
                r.([key '_its']) = A.solver_its.(key)(n);
                r.([key '_flag']) = A.solver_flag.(key)(n);
                r.([key '_err']) = A.solver_err.(key)(n);
            end
            for i = 1:numel(diagnostics)
                f = diagnostics{i}; r.(f) = A.(f)(n);
            end
            rows{end+1} = r; %#ok<AGROW>
        end
    end
    writetable(struct2table([rows{:}]),fullfile(results_root,'all_results.csv'));
end
