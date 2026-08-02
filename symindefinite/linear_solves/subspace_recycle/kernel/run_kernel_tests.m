% RUN_KERNEL_TESTS  Run every kernel unit test and summarize.
%
% Nothing in diagnosis/ is trustworthy until these pass — in particular
% test_build_stokes_sequence T1 (the rank-2nC identity the whole study rests on)
% and T7 (agreement with the stored benchmark).
%
% Each test is a script that opens with `clear; clc`.  `clear` wipes the caller's
% workspace, so the accumulator lives in root appdata where it cannot reach; and
% `clc` would erase the console, so each test's output is captured with evalc and
% replayed afterwards.  Per-test timings are reported so an implausibly fast
% "all passed" cannot pass unnoticed.
%
% Run:  run_kernel_tests

clc;
setappdata(0, 'srk_tests', { ...
    'test_build_stokes_sequence', ...
    'test_ildl_coordinate_map', ...
    'test_transport_V', ...
    'test_lowrank_update_basis', ...
    'test_two_level_it', ...
    'test_matvec_budget'});
setappdata(0, 'srk_i', 0);
setappdata(0, 'srk_fail', {});
setappdata(0, 'srk_log', {});
setappdata(0, 'srk_sec', []);
setappdata(0, 'srk_dir', fileparts(mfilename('fullpath')));

while getappdata(0, 'srk_i') < numel(getappdata(0, 'srk_tests'))
    setappdata(0, 'srk_i', getappdata(0, 'srk_i') + 1);
    names = getappdata(0, 'srk_tests');
    nm    = names{getappdata(0, 'srk_i')};
    setappdata(0, 'srk_tic', tic);
    try
        out = evalc(sprintf('run(''%s'')', ...
                    fullfile(getappdata(0, 'srk_dir'), [nm '.m'])));
    catch ME
        out = sprintf('  FAILED: %s', ME.message);
        f = getappdata(0, 'srk_fail');
        f{end+1} = sprintf('%s: %s', nm, ME.message); %#ok<SAGROW>
        setappdata(0, 'srk_fail', f);
    end
    L = getappdata(0, 'srk_log');   L{end+1} = out;              %#ok<SAGROW>
    setappdata(0, 'srk_log', L);
    setappdata(0, 'srk_sec', [getappdata(0, 'srk_sec'), toc(getappdata(0, 'srk_tic'))]);
end

names = getappdata(0, 'srk_tests');
logs  = getappdata(0, 'srk_log');
secs  = getappdata(0, 'srk_sec');
fail  = getappdata(0, 'srk_fail');

for i = 1:numel(logs)
    fprintf('%s', logs{i});
    fprintf('  [%s: %.1f s]\n\n', names{i}, secs(i));
end

fprintf('==================================================================\n');
if isempty(fail)
    fprintf('  ALL %d KERNEL TEST FILES PASSED  (%.1f s total)\n', ...
            numel(names), sum(secs));
else
    fprintf(2, '  %d TEST FILE(S) FAILED:\n', numel(fail));
    for i = 1:numel(fail), fprintf(2, '    %s\n', fail{i}); end
end
fprintf('==================================================================\n');

for f = {'srk_tests','srk_i','srk_fail','srk_dir','srk_log','srk_sec','srk_tic'}
    if isappdata(0, f{1}), rmappdata(0, f{1}); end
end
