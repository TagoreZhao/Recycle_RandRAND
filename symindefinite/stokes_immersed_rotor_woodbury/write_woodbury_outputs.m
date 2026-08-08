function files = write_woodbury_outputs(Astats, params, outDir)
%WRITE_WOODBURY_OUTPUTS  CSVs for the Woodbury benchmark: per case, plus a summary.
%   FILES = WRITE_WOODBURY_OUTPUTS(ASTATS, PARAMS, OUTDIR)
%
%   ASTATS is a cell array of solve_woodbury_sequence outputs, one per case.
%
%   THE CSVs ARE THE CONTRACT between the expensive run and cheap figure
%   regeneration: write_woodbury_figures reads ONLY these files, never an Astat,
%   so replot_woodbury can redraw everything without solving anything.  Adding a
%   figure that needs a quantity means adding the column here first.
%
%   Writes, into OUTDIR:
%     <case>_results.csv    one row per timestep
%     woodbury_summary.csv  one row per case
%
%   See also: write_woodbury_figures, replot_woodbury, solve_woodbury_sequence.

    if ~exist(outDir, 'dir'), mkdir(outDir); end

    files    = {};
    summrows = cell(numel(Astats), 1);

    for i = 1:numel(Astats)
        A = Astats{i};
        T = local_case_table(A);
        f = fullfile(outDir, sprintf('%s_results.csv', A.case_name));
        writetable(T, f);
        fprintf('Wrote %s\n', f);
        files{end+1} = f;                                       %#ok<AGROW>
        summrows{i}  = local_summary_row(A);
    end

    Tsum = vertcat(summrows{:});
    fsum = fullfile(outDir, 'woodbury_summary.csv');
    writetable(Tsum, fsum);
    fprintf('Wrote %s\n', fsum);
    files{end+1} = fsum;

    % Config, minus anything unserializable.  params holds no function handles in
    % this study (the sequence kernel owns the motion closures), but the guard is
    % kept so a later knob cannot silently break the .json write.
    cfgFile = fullfile(outDir, 'run_config.mat');
    save(cfgFile, 'params');
    fprintf('Wrote %s\n', cfgFile);
    files{end+1} = cfgFile;

    jsonFile = fullfile(outDir, 'run_config.json');
    fid = fopen(jsonFile, 'w');
    if fid > 0
        fprintf(fid, '%s', jsonencode(local_jsonable(params), 'PrettyPrint', true));
        fclose(fid);
        fprintf('Wrote %s\n', jsonFile);
        files{end+1} = jsonFile;
    else
        warning('write_woodbury_outputs:noJson', ...
                'Could not open %s for writing; skipping.', jsonFile);
    end
end

%==========================================================================
function T = local_case_table(A)
%LOCAL_CASE_TABLE  One row per timestep.
    n = A.nsteps;
    T = table((1:n)', 'VariableNames', {'timestep'});

    T.nC   = A.nC(:);
    T.nsys = A.nsys(:);

    % --- operator drift ---------------------------------------------------
    T.dC_normF        = A.dC_normF(:);
    T.dC_rel          = A.dC_rel(:);
    T.coupling_change = A.coupling_change(:);
    T.correction_rel  = A.correction_rel(:);

    % --- capacitance conditioning (the metric the forward error tracks) ---
    T.cap_cond   = A.cap_cond(:);
    T.cap_smin   = A.cap_smin(:);
    T.cap_smax   = A.cap_smax(:);
    T.cap_rcond  = A.cap_rcond(:);
    T.cap_symres = A.cap_symres(:);

    % --- per-arm quality --------------------------------------------------
    for k = A.solver_keys(:)'
        key = k{1};
        T.(sprintf('%s_err', key))    = A.solver_err.(key)(:);
        T.(sprintf('%s_relres', key)) = A.solver_relres.(key)(:);
    end
    T.backslash_relres = A.backslash_relres(:);

    % --- per-arm cost -----------------------------------------------------
    T.t_woodbury_net  = A.t_woodbury_net(:);
    T.t_woodbury      = A.t_woodbury(:);
    T.t_woodbury_diag = A.t_woodbury_diag(:);
    T.t_frozen        = A.t_frozen(:);
    T.t_fresh         = A.t_fresh(:);
    T.cum_woodbury    = A.cum_woodbury(:);
    T.cum_fresh       = A.cum_fresh(:);

    % --- optional spectrum of K_n ----------------------------------------
    if any(isfinite(A.kappa))
        T.lambda_absmin = A.lambda_absmin(:);
        T.lambda_absmax = A.lambda_absmax(:);
        T.kappa         = A.kappa(:);
    end
end

%==========================================================================
function R = local_summary_row(A)
%LOCAL_SUMMARY_ROW  The headline numbers for one case.
    we = A.solver_err.woodbury;
    fe = A.solver_err.frozen;
    re = A.solver_err.fresh;

    tw = mean(A.t_woodbury_net, 'omitnan');
    tf = mean(A.t_fresh, 'omitnan');

    R = table();
    R.case_name = string(A.case_name);
    R.h0        = A.h0;
    R.dt        = A.dt;
    R.nsteps    = A.nsteps;
    R.ntot      = A.ntot;
    R.nC        = A.nC_const;
    R.ref       = A.ref;

    % --- the one factorization --------------------------------------------
    R.nnzK1      = A.nnzK1;
    R.nnzL       = A.nnzL;
    R.fill_ratio = A.fill_ratio;
    R.t_factor   = A.t_factor;
    R.t_setup    = A.t_setup;

    % --- drift -------------------------------------------------------------
    R.dC_rel_max = max(A.dC_rel, [], 'omitnan');

    % --- accuracy ----------------------------------------------------------
    R.woodbury_err_mean   = mean(we, 'omitnan');
    R.woodbury_err_max    = max(we, [], 'omitnan');
    R.woodbury_err_final  = we(end);
    R.woodbury_relres_max = max(A.solver_relres.woodbury, [], 'omitnan');
    R.frozen_err_mean     = mean(fe, 'omitnan');
    R.frozen_err_max      = max(fe, [], 'omitnan');
    % The ground-truth self-check: `fresh` solves the same system a second way,
    % so anything above ~1e-13 here invalidates every other number in the row.
    R.fresh_err_max       = max(re, [], 'omitnan');

    % --- conditioning ------------------------------------------------------
    R.cap_cond_max  = max(A.cap_cond, [], 'omitnan');
    R.cap_rcond_min = min(A.cap_rcond, [], 'omitnan');

    % --- cost --------------------------------------------------------------
    R.t_woodbury_net_mean = tw;
    R.t_frozen_mean       = mean(A.t_frozen, 'omitnan');
    R.t_fresh_mean        = tf;
    R.speedup_per_step    = tf / max(tw, realmin);
    R.break_even_step     = A.break_even_step;
    R.total_woodbury_s    = A.cum_woodbury(end);
    R.total_fresh_s       = A.cum_fresh(end);
    R.total_speedup       = A.cum_fresh(end) / max(A.cum_woodbury(end), realmin);
end

%==========================================================================
function s = local_jsonable(params)
%LOCAL_JSONABLE  Drop any field jsonencode would choke on (handles, objects).
    s = struct();
    f = fieldnames(params);
    for i = 1:numel(f)
        v = params.(f{i});
        if isa(v, 'function_handle') || isobject(v)
            continue;
        end
        s.(f{i}) = v;
    end
end
