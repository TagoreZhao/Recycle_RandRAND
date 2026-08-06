%TEST_PLOT_HELPERS  Unit tests for the stokes_immersed_rotor plotting helpers.
%
%   Run:  cd tests; test_plot_helpers
%
%   These cover the pieces that decide whether a figure is readable: the legend
%   text, the per-curve styles, the coincident-curve detection, and the CSV ->
%   stats reconstruction that lets replot_benchmark redraw without re-solving.
%   T20 is the direct regression test for the defect that motivated all of this
%   -- the all-cases figure used to carry one full legend PER PANEL, so the
%   boxes overlapped each other and the data.
%
%   No solves are run and nothing outside tempdir is written.

clear; clc;
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(fileparts(fileparts(root)));   % repo root, for src.*
addpath(root);

np = 0; nf = 0;
fprintf('=== test_plot_helpers ===\n');

KEYS = {'minres_unprec'; 'block_jacobi'; 'ildl_nofill'; 'exact_ldl_frozen'; ...
        'two_level_sjlt'; 'two_level_gaussian'; 'two_level_polynomial'; ...
        'two_level_exact'; 'two_level_krylov'};
nK = numel(KEYS);

%% ------------------------------------------------- solver_short_label ----
short = solver_short_label(KEYS);

% T1  every registry key maps to the documented compact name.
[np, nf] = chk(np, nf, 'T1  short labels match the documented table', ...
    isequal(short, {'unpreconditioned'; 'block Jacobi'; 'ILDL (no-fill)'; ...
                    'exact LDL (frozen)'; '2-level: sjlt V'; ...
                    '2-level: gaussian V'; '2-level: polynomial V'; ...
                    '2-level: exact V'; '2-level: gauss V + recycling'}));

% T2  labels must be distinguishable, or the legend is useless.
[np, nf] = chk(np, nf, 'T2  short labels are pairwise unique', ...
    numel(unique(short)) == numel(short));

% T3  short enough that 3 columns fit the figure width (the whole point).
[np, nf] = chk(np, nf, 'T3  every short label <= 30 chars', ...
    all(cellfun(@numel, short) <= 30));

% T4  a solver added to the registry later must still plot.
[np, nf] = chk(np, nf, 'T4  unknown key falls back readably', ...
    strcmp(solver_short_label('two_level_wibble'), '2-level: wibble') && ...
    strcmp(solver_short_label('some_new_thing'), 'some new thing'));

% T5  shape contract: char in -> char out, cellstr in -> same-size cellstr.
[np, nf] = chk(np, nf, 'T5  char in -> char out, cell in -> cell out', ...
    ischar(solver_short_label('block_jacobi')) && ...
    iscell(short) && isequal(size(short), size(KEYS)));

%% ------------------------------------------------- solver_style_table ----
sty = solver_style_table(nK);

% T6  one style per curve.
[np, nf] = chk(np, nf, 'T6  solver_style_table(nK) returns one row per key', ...
    numel(sty) == nK);

% T7  no two curves share both colour and marker.
dup = false;
for i = 1:nK
    for j = i+1:nK
        if isequal(sty(i).color, sty(j).color) && strcmp(sty(i).marker, sty(j).marker)
            dup = true;
        end
    end
end
[np, nf] = chk(np, nf, 'T7  no two styles share colour AND marker', ~dup);

% T8  the trio that genuinely coincides (sjlt=5, gaussian=6, krylov=9) must
%     differ in all three attributes, not just one.
trio = [5 6 9];
ok = true;
for a = 1:3
    for b = a+1:3
        i = trio(a); j = trio(b);
        ok = ok && ~isequal(sty(i).color, sty(j).color) ...
                && ~strcmp(sty(i).marker, sty(j).marker) ...
                && ~strcmp(sty(i).linestyle, sty(j).linestyle);
    end
end
[np, nf] = chk(np, nf, 'T8  sjlt/gaussian/krylov differ in colour, marker AND line', ok);

% T9  decreasing widths let a later coincident curve leave the earlier visible.
%     Pinned to the documented EIGHT base rows: past the palette the table cycles
%     (row 9 is row 1's width again), which is by design, not a regression.
sty8 = solver_style_table(8);
[np, nf] = chk(np, nf, 'T9  linewidths strictly decreasing over the 8 base rows', ...
    all(diff([sty8.linewidth]) < 0));

% T10 registry growth past the palette must not error.
[np, nf] = chk(np, nf, 'T10 solver_style_table(12) cycles without error', ...
    numel(solver_style_table(12)) == 12 && numel(solver_style_table(0)) == 0);

%% --------------------------------------------- mark_coincident_curves ----
ns = 20;
mk = @(v) struct('solver_keys', {KEYS}, 'solver_its', v);
its = struct();
rng(0);
base = 100 + randi(20, ns, 1);
for s = 1:numel(KEYS), its.(KEYS{s}) = base + 10*s; end
its.two_level_krylov = its.two_level_gaussian;             % exact duplicate
stD = mk(its);
tags = mark_coincident_curves(stD);

% T11 an exact duplicate is named after the curve it hides under.
[np, nf] = chk(np, nf, 'T11 exact duplicate tagged with the earlier solver', ...
    strcmp(tags{nK}, ' (= 2-level: gaussian V)'));

% T12 distinct curves are not tagged.
[np, nf] = chk(np, nf, 'T12 distinct curves carry no tag', ...
    all(cellfun(@isempty, tags(1:nK-1))));

% T13 one iteration of difference is a real difference.
its.two_level_krylov(3) = its.two_level_krylov(3) + 1;
[np, nf] = chk(np, nf, 'T13 off-by-one curve is NOT reported as coincident', ...
    isempty(subsref(mark_coincident_curves(mk(its)), substruct('{}', {nK}))));

%% ------------------------------------------------------ CSV round-trip ----
tmp = fullfile(tempdir, sprintf('rrp_test_%d', feature('getpid')));
if exist(tmp, 'dir'), rmdir(tmp, 's'); end
mkdir(tmp);
cleanup = onCleanup(@() rmdir(tmp, 's'));

k3 = {'minres_unprec'; 'block_jacobi'; 'two_level_exact'};
syn = cell(2, 1);
for c = 1:2
    s = struct('case_name', sprintf('case_%d', c), 'geometry', 'synthetic', ...
               'dt', 0.02, 'solver_keys', {k3}, 'solver_labels', {k3});
    for j = 1:3
        s.solver_its.(k3{j})  = (1:5)' * 10 * j + c;
        s.solver_flag.(k3{j}) = zeros(5, 1);
    end
    s.solver_relres.(k3{end}) = 1e-9 * (1:5)';
    s.solver_err.(k3{end})    = 1e-10 * (1:5)';
    s.coupling_change   = 0.1 * (1:5)';
    s.backslash_relres  = 1e-15 * (1:5)';
    s.constraint_res    = 1e-16 * (1:5)';
    s.nC                = 20 * ones(5, 1);
    syn{c} = s;
end
write_all_results_csv(tmp, syn, 'synthetic', k3);
[back, cfgb] = load_benchmark_stats(tmp);

% T14 iteration counts and flags are integers and must round-trip EXACTLY --
%     they are the plotted quantity and the basis of every reported speedup.
ok = numel(back) == 2;
for c = 1:2
    for j = 1:3
        ok = ok && isequal(back{c}.solver_its.(k3{j}),  syn{c}.solver_its.(k3{j})) ...
                && isequal(back{c}.solver_flag.(k3{j}), syn{c}.solver_flag.(k3{j}));
    end
    ok = ok && isequal(back{c}.nC, syn{c}.nC);
end
[np, nf] = chk(np, nf, 'T14 CSV round-trip preserves iterations/flags exactly', ok);

% T14b the float columns go through writetable's ~15-significant-digit text
%      form, so they come back to double rounding, not bit-identically. Assert
%      the tolerance that actually holds rather than pretending it is exact.
relerr = @(a, b) max(abs(a - b) ./ max(abs(b), realmin));
ok = true;
for c = 1:2
    ok = ok && relerr(back{c}.coupling_change,  syn{c}.coupling_change)  < 1e-14 ...
            && relerr(back{c}.constraint_res,   syn{c}.constraint_res)   < 1e-14 ...
            && relerr(back{c}.backslash_relres, syn{c}.backslash_relres) < 1e-14;
end
[np, nf] = chk(np, nf, 'T14b float columns round-trip to 1e-14 relative', ok);

% T15 order matters: the legend and the styles are assigned by position.
[np, nf] = chk(np, nf, 'T15 round-trip preserves key and case order', ...
    isequal(cfgb.solver_keys(:), k3(:)) && ...
    isequal(cfgb.case_names(:), {'case_1'; 'case_2'}));

% T16 the new column carries the accuracy curve back.
[np, nf] = chk(np, nf, 'T16 solver_err_last round-trips into stats.solver_err', ...
    isfield(back{1}, 'solver_err') && ...
    isequal(back{1}.solver_err.(k3{end}), syn{1}.solver_err.(k3{end})));

%% ---------------------------------------------- real results directory ----
realRoot = fullfile(root, 'benchmark_no_krylov_recycle');
if exist(fullfile(realRoot, 'all_results.csv'), 'file')
    [rs, rcfg] = load_benchmark_stats(realRoot);

    % T17 the committed run loads with its actual shape: 60 steps, not the
    %     params.Tstep = 61 the config advertises (the engine records 60).
    %     The key COUNT is a lower bound, not a literal: run_benchmark writes into
    %     this very directory, so a re-run after a registry change legitimately
    %     lands here with more columns than the currently committed CSV has.
    recorded = rcfg.solver_keys(:);
    nsteps = cellfun(@(s) numel(s.solver_its.minres_unprec), rs);
    [np, nf] = chk(np, nf, 'T17 real run: 3 cases, >= 8 keys, 60 steps each, dt = 0.02', ...
        numel(rs) == 3 && numel(recorded) >= 8 && ...
        all(nsteps == 60) && abs(rs{1}.dt - 0.02) < 1e-12);

    % T18 keys in registry order -> styles/labels line up with the live run.
    %     Stated relative to the current registry rather than pinned to the whole
    %     of KEYS, so it holds both for the committed CSV (written before
    %     exact_ldl_frozen existed) and for one regenerated with it.
    [np, nf] = chk(np, nf, 'T18 real run: keys in registry order', ...
        all(ismember(recorded, KEYS)) && ...
        isequal(recorded, KEYS(ismember(KEYS, recorded))));

    % T19 the DEFLAT_RECYCLE_K = 0 collapse is detected and will be stated.
    rtags = mark_coincident_curves(rs{1});
    ik = find(strcmp(recorded, 'two_level_krylov'), 1);
    [np, nf] = chk(np, nf, 'T19 real run: krylov flagged as identical to gaussian', ...
        ~isempty(ik) && strcmp(rtags{ik}, ' (= 2-level: gaussian V)'));
else
    fprintf('  skip T17-T19 (no committed benchmark_no_krylov_recycle)\n');
    rs = back;
end

%% ------------------------------------------------------- figure smoke ----
figdir = fullfile(tmp, 'figs');
opts   = benchmark_fig_defaults(struct('close', false));

% T20 ONE legend on the all-cases figure, not one per panel. This is the
%     regression test for the original overlap defect.
write_all_cases_comparison(figdir, rs, opts);
fh  = gcf;
nlg = numel(findall(fh, 'Type', 'legend'));
nax = numel(findall(fh, 'Type', 'axes'));
[np, nf] = chk(np, nf, 'T20 all-cases figure has exactly ONE legend', nlg == 1);

% T21 ... spread over the panels it describes.
[np, nf] = chk(np, nf, 'T21 all-cases figure has one axes per case', nax == numel(rs));

% T22 the shared legend sits in its own tile, not inside an axes.
lg = findall(fh, 'Type', 'legend');
[np, nf] = chk(np, nf, 'T22 legend is placed in the south tile', ...
    ~isempty(lg) && strcmp(string(lg(1).Layout.Tile), "south"));
close(fh);

% T23 every figure kind writes a real file.
opts = benchmark_fig_defaults();
write_case_figures(figdir, rs{1}, opts);
write_iteration_vs_timestep(figdir, rs{1}, opts);
want = {'all_solvers_comparison.png', 'relative_step_to_step_change.png', ...
        'accuracy.png', [rs{1}.case_name '.png'], ...
        [rs{1}.solver_keys{1} '_solver_iterations.png']};
ok = true;
for i = 1:numel(want)
    d = dir(fullfile(figdir, want{i}));
    ok = ok && ~isempty(d) && d.bytes > 10e3;
end
[np, nf] = chk(np, nf, 'T23 every figure kind writes a non-trivial PNG', ok);

% T24 replot must not touch the data files it was handed.
[np, nf] = chk(np, nf, 'T24 figures-only run writes no per-solver CSV', ...
    isempty(dir(fullfile(figdir, '*_solver_iterations.csv'))));

% T25 the two drivers must draw the SAME accuracy figure. solve_stokes_immersed
%     hands over solver_relres (a per-solver struct); all_results.csv flattens
%     the last solver's column to relres. Reading only one of those names cost
%     the live figure a curve, so both shapes are exercised here.
liveShape = syn{1};                                   % has solver_relres, no relres
csvShape  = back{1};                                  % has relres, no solver_relres
optsK     = benchmark_fig_defaults(struct('close', false));
nline = @(st) accuracy_line_count(fullfile(tmp, 'acc'), st, optsK);
[np, nf] = chk(np, nf, 'T25 live and CSV stats give the same accuracy curves', ...
    ~isfield(liveShape, 'relres') && ~isfield(csvShape, 'solver_relres') && ...
    nline(liveShape) == nline(csvShape) && nline(liveShape) == 4);

%% ------------------------------------------------------------ summary ----
fprintf('\n%d passed, %d FAILED\n', np, nf);
if nf > 0, error('test_plot_helpers:failures', '%d checks failed', nf); end

%==========================================================================
function n = accuracy_line_count(dir_, st, opts)
%ACCURACY_LINE_COUNT  Draw the accuracy figure and count the curves on it.
    write_case_figures(dir_, st, opts);
    fh = gcf;
    n  = numel(findall(fh, 'Type', 'line', '-not', 'Tag', 'legend'));
    ax = findall(fh, 'Type', 'axes');
    n  = numel(findobj(ax(1), 'Type', 'line'));
    close(fh);
end

%==========================================================================
function [np, nf] = chk(np, nf, name, cond)
    if cond
        np = np + 1;  fprintf('  ok   %s\n', name);
    else
        nf = nf + 1;  fprintf('  FAIL %s\n', name);
    end
end
