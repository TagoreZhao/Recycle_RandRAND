%RUN_WOODBURY_BENCHMARK  Driver: can ONE factorization of A_1 serve the sequence?
%
%   Run:  run_woodbury_benchmark
%         SMOKE_TEST = true; run_woodbury_benchmark
%
%   Solves the immersed-rotor KKT sequence three ways at every timestep -- the
%   Woodbury update on a frozen factorization of A_1 = K_1, the same frozen
%   inverse with NO correction, and a fresh factorization each step -- then writes
%   CSVs and figures.
%
%   SMOKE_TEST trims h0 to 0.1 and the run to 5 steps.  It trims max_steps, NOT
%   Tstep: Tstep sets Tmax and hence the rotor's angular velocity, so shrinking it
%   would change the geometry under test instead of just doing fewer solves.  It
%   is read from the base workspace BEFORE the clear below, which is why the read
%   comes first.
%
%   See also: solve_woodbury_sequence, make_woodbury_params, replot_woodbury.

% --- Read the flag before clearing the workspace it lives in ---------------
if evalin('base', 'exist(''SMOKE_TEST'', ''var'')')
    SMOKE = evalin('base', 'SMOKE_TEST');
else
    SMOKE = false;
end

clearvars -except SMOKE; clc;

paths = add_woodbury_paths();
assert_woodbury_helpers();

params = make_woodbury_params();

if SMOKE
    params.h0        = 0.1;
    params.max_steps = 5;
    outDir = paths.smokeDir;
    fprintf('*** SMOKE TEST: h0 = %.2f, %d steps ***\n', ...
            params.h0, params.max_steps);
else
    outDir = paths.outDir;
end

nsteps_planned = params.Tstep - 1;
if ~isempty(params.max_steps)
    nsteps_planned = min(nsteps_planned, params.max_steps);
end

fprintf('================================================================\n');
fprintf('Woodbury-update benchmark for the immersed-rotor KKT sequence\n');
fprintf('  h0 = %.3f   dt = %.3f   Tstep = %d   steps = %d\n', ...
        params.h0, params.dt, params.Tstep, nsteps_planned);
fprintf('  reference: A_1 = K_1, frozen (no refresh cadence, by design)\n');
fprintf('  output   : %s\n', outDir);
fprintf('================================================================\n');

Astats = cell(numel(params.cases), 1);
t_all  = tic;

for ci = 1:numel(params.cases)
    cname = params.cases{ci};
    fprintf('\n--- case %d/%d: %s ---\n', ci, numel(params.cases), cname);
    Astats{ci} = solve_woodbury_sequence(cname, params);
end

fprintf('\nAll cases solved in %.1f s.\n', toc(t_all));

% --- Outputs ---------------------------------------------------------------
write_woodbury_outputs(Astats, params, outDir);
write_woodbury_figures(outDir);

% --- Console summary -------------------------------------------------------
fprintf('\n================== SUMMARY ==================\n');
fprintf('%-18s %10s %10s %10s %8s %7s\n', ...
        'case', 'wood err', 'froz err', 'cond(Cap)', 'speedup', 'b/even');
for ci = 1:numel(Astats)
    A = Astats{ci};
    fprintf('%-18s %10.2e %10.2e %10.2e %7.2fx %7s\n', ...
            A.case_name, ...
            max(A.solver_err.woodbury, [], 'omitnan'), ...
            max(A.solver_err.frozen,   [], 'omitnan'), ...
            max(A.cap_cond, [], 'omitnan'), ...
            mean(A.t_fresh, 'omitnan') / ...
                max(mean(A.t_woodbury_net, 'omitnan'), realmin), ...
            local_bestr(A.break_even_step));
end
fprintf('=============================================\n');
fprintf('Columns are MAXIMA over the sequence for the error/conditioning\n');
fprintf('columns, and per-step means for speedup.  See woodbury_summary.csv.\n');

%==========================================================================
function s = local_bestr(be)
    if isnan(be)
        s = 'never';        % setup never amortized over this many steps
    else
        s = sprintf('%d', be);
    end
end
