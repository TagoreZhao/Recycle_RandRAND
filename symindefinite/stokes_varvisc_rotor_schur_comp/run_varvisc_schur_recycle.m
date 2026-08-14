% RUN_VARVISC_SCHUR_RECYCLE  Full variable-viscosity Schur benchmark.
%
% Usage:
%   run_varvisc_schur_recycle
%   SMOKE_TEST = true; run_varvisc_schur_recycle

is_smoke = evalin('base', ...
    'exist(''SMOKE_TEST'',''var'') && logical(SMOKE_TEST)');
clearvars -except is_smoke
clc;

paths = add_varvisc_schur_paths();
assert_varvisc_schur_helpers();
rng(1);
params = make_varvisc_schur_params();
case_names = {'bar_rotating_nu_orbiting', ...
              'disk_translating_nu_wake', ...
              'disk_static_nu_const'};

if is_smoke
    params.max_steps = 3;
    params.h0 = 0.1;
    params.sm_eig = 20;
    case_names = case_names(1);
    results_root = fullfile(paths.thisDir,'varvisc_schur_recycle_smoke');
else
    results_root = paths.outDir;
end
if ~exist(results_root,'dir'), mkdir(results_root); end

fprintf('[varvisc_schur] building mesh h0=%.3f ...\n',params.h0);
[~,msh] = varvisc_schur_make_cfg(case_names{1},params,[]);
fprintf('  nodes=%d, velocity DOFs=%d, pressure DOFs=%d\n',msh.N,2*msh.N,msh.N);

opts = varvisc_schur_fig_defaults();
all_stats = cell(numel(case_names),1);
for ci = 1:numel(case_names)
    cname = case_names{ci}; rng(1);
    fprintf('\n========== %s ==========\n',cname);
    cfg = varvisc_schur_make_cfg(cname,params,msh);
    run_dir = fullfile(results_root,cname);
    A = solve_varvisc_schur_sequence(cfg,params,run_dir);
    all_stats{ci} = A;
    write_varvisc_schur_case_outputs(run_dir,A,opts);
end

write_varvisc_schur_all_results(results_root,all_stats);
write_varvisc_schur_speedup(results_root,all_stats);
write_varvisc_schur_summary(results_root,all_stats,opts);

cfg_dump = params;
if isfield(cfg_dump,'standalone_variants')
    cfg_dump.variant_names = {cfg_dump.standalone_variants.name};
    cfg_dump = rmfield(cfg_dump,'standalone_variants');
end
cfg_dump.case_names = case_names;
cfg_dump.solver_keys = all_stats{1}.solver_keys;
cfg_dump.solver_labels = all_stats{1}.solver_labels;
cfg_dump.tau_used = all_stats{1}.tau;
cfg_dump.deflat_dim = all_stats{1}.deflat_dim;
cfg_dump.mesh_N = msh.N;
cfg_dump.matlab = version;
cfg_dump.finished = datestr(now,'yyyy-mm-ddTHH:MM:SS'); %#ok<TNOW1,DATST>
save(fullfile(results_root,'run_config.mat'),'cfg_dump');
fid = fopen(fullfile(results_root,'run_config.json'),'w');
if fid > 0
    fprintf(fid,'%s',jsonencode(cfg_dump,'PrettyPrint',true)); fclose(fid);
end
fprintf('\n[varvisc_schur] done -> %s\n',results_root);
