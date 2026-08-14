function write_varvisc_schur_speedup(results_root, all_stats)
%WRITE_VARVISC_SCHUR_SPEEDUP  Iteration comparisons against frozen chol(S_1).
    rows = {};
    for ci = 1:numel(all_stats)
        A = all_stats{ci}; base = A.solver_its.chol;
        for i = 1:numel(A.solver_keys)
            key = A.solver_keys{i}; if strcmp(key,'chol'), continue; end
            v = A.solver_its.(key);
            rows{end+1} = struct('geometry',string(A.geometry), ... %#ok<AGROW>
                'case_name',string(A.case_name),'variant',string(key), ...
                'max_diff_vs_chol',max(base-v), ...
                'max_factor_vs_chol',max(base./max(v,1)), ...
                'median_chol',median(base),'median_variant',median(v));
        end
    end
    writetable(struct2table([rows{:}]),fullfile(results_root,'speedup_summary.csv'));
end
