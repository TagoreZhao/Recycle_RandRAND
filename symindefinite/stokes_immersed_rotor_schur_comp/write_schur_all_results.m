function write_schur_all_results(results_root, all_stats)
%WRITE_SCHUR_ALL_RESULTS  Master long-format CSV, one row per (case, timestep).
%   WRITE_SCHUR_ALL_RESULTS(RESULTS_ROOT, ALL_STATS)
%
%   Columns: case_name, geometry, timestep, <arm>_its / _flag for every arm,
%   then the diagnostics (kappa, lambda_min/max, drift, inverse staleness,
%   recovery error, nC, nS).
%
%   See also: solve_schur_sequence, write_schur_speedup.

    rows = {};
    for k = 1:numel(all_stats)
        A    = all_stats{k};
        keys = A.solver_keys;
        for n = 1:A.nsteps
            r = struct();
            r.case_name = string(A.case_name);
            r.geometry  = string(A.geometry);
            r.timestep  = n;
            for i = 1:numel(keys)
                r.([keys{i} '_its'])  = A.solver_its.(keys{i})(n);
                r.([keys{i} '_flag']) = A.solver_flag.(keys{i})(n);
                r.([keys{i} '_err'])  = A.solver_err.(keys{i})(n);
            end
            r.kappa              = A.kappa(n);
            r.lambda_min         = A.lambda_min(n);
            r.lambda_max         = A.lambda_max(n);
            r.ReldiffF           = A.ReldiffF(n);
            r.RelInitdiffF       = A.RelInitdiffF(n);
            r.InvRelDiff         = A.InvRelDiff(n);
            r.LowRankInvRelDiff  = A.LowRankInvRelDiff(n);
            r.coupling_change    = A.coupling_change(n);
            r.backslash_relres   = A.backslash_relres(n);
            r.vel_recovery_err   = A.vel_recovery_err(n);
            r.nC                 = A.nC(n);
            r.nS                 = A.nS(n);
            rows{end+1} = r; %#ok<AGROW>
        end
    end

    T = struct2table([rows{:}]);
    writetable(T, fullfile(results_root, 'all_results.csv'));
end
