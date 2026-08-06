function write_schur_speedup(results_root, all_stats)
%WRITE_SCHUR_SPEEDUP  Max iteration gain of every arm against each baseline.
%   WRITE_SCHUR_SPEEDUP(RESULTS_ROOT, ALL_STATS)
%
%   For each (geometry, case) and each arm, reports max(base - arm) and
%   max(base ./ arm) against the single baseline this study keeps: `chol`, the
%   exact dense Cholesky of S_1 built once and recycled.
%
%   That is the honest comparison here.  S is dense, so there is no incomplete
%   factorization to compare against -- the realistic alternative to deflation
%   is simply reusing one exact factorization for as long as it holds up.
%
%   See also: write_schur_all_results.

    bases = {'chol'};
    rows  = {};

    for k = 1:numel(all_stats)
        A    = all_stats{k};
        keys = A.solver_keys;
        for i = 1:numel(keys)
            key = keys{i};
            if ismember(key, bases), continue; end
            r = struct();
            r.geometry  = string(A.geometry);
            r.case_name = string(A.case_name);
            r.variant   = string(key);
            v = A.solver_its.(key);
            for b = 1:numel(bases)
                if ~isfield(A.solver_its, bases{b}), continue; end
                base = A.solver_its.(bases{b});
                r.(['max_diff_vs_'   bases{b}]) = max(base - v);
                r.(['max_factor_vs_' bases{b}]) = max(base ./ max(v, 1));
                r.(['median_'        bases{b}]) = median(base);
            end
            r.median_variant = median(v);
            rows{end+1} = r; %#ok<AGROW>
        end
    end

    if isempty(rows), return; end
    T = struct2table([rows{:}]);
    writetable(T, fullfile(results_root, 'speedup_summary.csv'));
end
