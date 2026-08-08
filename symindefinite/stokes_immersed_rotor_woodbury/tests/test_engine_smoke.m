%TEST_ENGINE_SMOKE  The engine runs, its Astat honours the contract, CSVs round-trip.
%
%   Run:  cd tests; test_engine_smoke
%
%   Everything numerical is already pinned by the other tests; this one guards the
%   plumbing that carries those numbers to the figures:
%
%     * solve_woodbury_sequence returns the Astat shape the family contract
%       specifies (solver_keys/labels plus per-key err/relres/its/flag), so the
%       sibling-style CSV and figure layer keeps working;
%     * write_woodbury_outputs writes every column write_woodbury_figures reads,
%       which is what makes replot_woodbury able to redraw without re-solving.
%       A figure needing a quantity that never reached the CSV is the failure this
%       catches;
%     * the ground truth validates itself: the `fresh` arm solves the same system
%       a second way and must agree to ~1e-13.
%
%   WRITES NOTHING OUTSIDE tempdir.
%
%   See also: solve_woodbury_sequence, write_woodbury_outputs, write_woodbury_figures.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
studyDir    = fileparts(thisFileDir);
addpath(studyDir);
add_woodbury_paths();
assert_woodbury_helpers();

rng(0);
fprintf('=== test_engine_smoke ===\n');

params           = make_woodbury_params();
params.h0        = 0.1;
params.max_steps = 3;
params.cases     = {'bar_rotating', 'disk_static'};
params.TIME_REPEATS = 1;            % this test is about plumbing, not timing

np = 0;  nf = 0;

Astats = cell(numel(params.cases), 1);
for ci = 1:numel(params.cases)
    Astats{ci} = solve_woodbury_sequence(params.cases{ci}, params);
end
A = Astats{1};

% --- T1  Astat contract --------------------------------------------------
need = {'solver_keys', 'solver_labels', 'solver_its', 'solver_flag', ...
        'solver_relres', 'solver_err', 'coupling_change', 'nC', 'nsteps', ...
        'cap_cond', 'cap_smin', 'cap_smax', 'cap_rcond', 'dC_rel', ...
        'correction_rel', 't_woodbury_net', 't_frozen', 't_fresh', ...
        'cum_woodbury', 'cum_fresh', 'break_even_step', 't_setup'};
missing = need(~isfield(A, need));
[np, nf] = chk(np, nf, ...
    sprintf('T1a Astat has every contract field (%d missing)', numel(missing)), ...
    isempty(missing));
if ~isempty(missing)
    fprintf(2, '        missing: %s\n', strjoin(missing, ', '));
end

ok_shape = true;
for k = A.solver_keys(:)'
    ok_shape = ok_shape && numel(A.solver_err.(k{1}))    == A.nsteps ...
                        && numel(A.solver_relres.(k{1})) == A.nsteps;
end
[np, nf] = chk(np, nf, ...
    sprintf('T1b every arm has nsteps = %d entries', A.nsteps), ok_shape);
[np, nf] = chk(np, nf, ...
    'T1c three arms registered: woodbury, frozen, fresh', ...
    isequal(A.solver_keys(:)', {'woodbury', 'frozen', 'fresh'}));

% --- T2  the ground truth validates itself -------------------------------
[np, nf] = chk(np, nf, ...
    sprintf('T2a fresh arm reproduces the ground truth (%.2e < 1e-13)', ...
            max(A.solver_err.fresh)), ...
    max(A.solver_err.fresh) < 1e-13);
[np, nf] = chk(np, nf, ...
    sprintf('T2b backslash residual is tiny (%.2e < 1e-10)', ...
            max(A.backslash_relres)), ...
    max(A.backslash_relres) < 1e-10);

% --- T3  the method works through the engine, not just standalone --------
[np, nf] = chk(np, nf, ...
    sprintf('T3a woodbury accurate via the engine (%.2e < 1e-9)', ...
            max(A.solver_err.woodbury)), ...
    max(A.solver_err.woodbury) < 1e-9);
[np, nf] = chk(np, nf, ...
    sprintf('T3b frozen control is O(1) wrong (%.2e > 1e-3)', ...
            max(A.solver_err.frozen)), ...
    max(A.solver_err.frozen) > 1e-3);

% --- T4  the static case still collapses through the engine -------------
Astatic = Astats{2};
[np, nf] = chk(np, nf, ...
    'T4  disk_static: correction is zero, frozen == woodbury', ...
    all(Astatic.correction_rel == 0) && ...
    max(Astatic.solver_err.frozen) < 1e-10);

% --- T5  CSV round-trip, into tempdir only ------------------------------
outDir = fullfile(tempdir, sprintf('woodbury_smoke_%s', ...
                                   char(matlab.lang.internal.uuid())));
cleanup = onCleanup(@() local_rmdir(outDir));
write_woodbury_outputs(Astats, params, outDir);

f1 = fullfile(outDir, 'bar_rotating_results.csv');
f2 = fullfile(outDir, 'woodbury_summary.csv');
[np, nf] = chk(np, nf, 'T5a both CSVs written', ...
    exist(f1, 'file') == 2 && exist(f2, 'file') == 2);

T = readtable(f1);
[np, nf] = chk(np, nf, ...
    sprintf('T5b results CSV has one row per step (%d)', height(T)), ...
    height(T) == A.nsteps);

% Every column the figure layer reads must be present -- this is the check that
% keeps replot_woodbury working after a figure is added.
figneed = {'timestep', 'woodbury_err', 'frozen_err', 'fresh_err', ...
           'woodbury_relres', 'frozen_relres', 'fresh_relres', ...
           'cap_cond', 'cap_smin', 'cap_smax', 'dC_rel', 'coupling_change', ...
           'correction_rel', 't_woodbury_net', 't_frozen', 't_fresh', ...
           'cum_woodbury', 'cum_fresh'};
have    = string(T.Properties.VariableNames);
figmiss = figneed(~ismember(figneed, have));
[np, nf] = chk(np, nf, ...
    sprintf('T5c CSV carries every column the figures read (%d missing)', ...
            numel(figmiss)), ...
    isempty(figmiss));
if ~isempty(figmiss)
    fprintf(2, '        missing: %s\n', strjoin(figmiss, ', '));
end

Tsum = readtable(f2, 'TextType', 'string');
[np, nf] = chk(np, nf, ...
    sprintf('T5d summary CSV has one row per case (%d)', height(Tsum)), ...
    height(Tsum) == numel(params.cases));

% --- T6  figures generate from the CSVs alone ---------------------------
figs = write_woodbury_figures(outDir);
[np, nf] = chk(np, nf, ...
    sprintf('T6  figures generated from CSVs only (%d files)', numel(figs)), ...
    numel(figs) >= 2*5 + 1 && all(cellfun(@(f) exist(f, 'file') == 2, figs)));

fprintf('\n  %d passed, %d failed\n', np, nf);
if nf > 0
    error('test_engine_smoke:fail', '%d assertion(s) failed.', nf);
end

%==========================================================================
function [np, nf] = chk(np, nf, name, cond)
    if cond
        np = np + 1;
        fprintf('  PASS  %s\n', name);
    else
        nf = nf + 1;
        fprintf(2, '  FAIL  %s\n', name);
    end
end

%==========================================================================
function local_rmdir(d)
    if exist(d, 'dir'), rmdir(d, 's'); end
end
