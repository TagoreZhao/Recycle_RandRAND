function make_paper_summary_table(results_root)
%MAKE_PAPER_SUMMARY_TABLE  Per-(geometry,motion-case) paper summary from
% <results_root>/all_results.csv.  Solver columns are discovered from the CSV
% (any "<key>_its" column), so the table extends automatically as preconditioners
% are added to define_solver_list.  Reports mean/std iterations per solver and,
% relative to the unpreconditioned baseline, the max per-step speedup factor.
%
%   results_root  (optional) output directory holding all_results.csv; defaults
%                 to <thisDir>/benchmark_final.  run_benchmark passes its own
%                 results_root so the summary tracks redirected output dirs.
    here    = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(results_root)
        results_root = fullfile(here, 'benchmark_final');
    end
    in_csv  = fullfile(results_root, 'all_results.csv');
    out_csv = fullfile(results_root, 'paper_summary_table.csv');

    T  = readtable(in_csv);
    vn = T.Properties.VariableNames;
    its_cols = vn(endsWith(vn, '_its'));
    keys     = erase(its_cols, '_its');

    [G, geom, cas] = findgroups(T.geometry, T.case_name);
    out = table(geom, cas, 'VariableNames', {'Geometry', 'MotionCase'});

    agg = @(v, fn) splitapply(fn, v, G);
    for i = 1:numel(its_cols)
        out.(['mean_' keys{i}]) = agg(T.(its_cols{i}), @mean);
        out.(['std_'  keys{i}]) = agg(T.(its_cols{i}), @(v) std(v, 1));
    end

    base = 'minres_unprec';
    if ismember([base '_its'], vn)
        for i = 1:numel(keys)
            if strcmp(keys{i}, base), continue; end
            out.(['max_factor_' base '_over_' keys{i}]) = ...
                splitapply(@(b, o) max(b ./ max(o, 1)), ...
                           T.([base '_its']), T.([keys{i} '_its']), G);
        end
    end

    writetable(out, out_csv);
    fprintf('Wrote %s (%d rows)\n', out_csv, height(out));
end
