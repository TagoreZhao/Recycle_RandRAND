% RUN_VARVISC_SCHUR_RANK  Diagnose full-rank variable-viscosity Schur drift.
clearvars; clc;
paths = add_varvisc_schur_paths(); params = make_varvisc_schur_params();
params.h0 = 0.1; nsteps = 6;
cfg = varvisc_schur_make_cfg('bar_rotating_nu_orbiting',params,[]);
ctx = varvisc_schur_context_init(cfg,params); u = zeros(ctx.nU,1);
Sprev = []; rows = {};
out_dir = fullfile(paths.outDir,'rank'); if ~exist(out_dir,'dir'), mkdir(out_dir); end
for n = 1:nsteps
    st = varvisc_schur_step_operator(ctx,n*params.dt,u);
    if ~isempty(Sprev)
        dS = st.S-Sprev; dSpp = dS(1:ctx.nP-1,1:ctx.nP-1);
        tol = 1e-10*norm(st.S,'fro');
        rows{end+1} = struct('step',n,'nS',st.nS,'nC',st.nC, ... %#ok<AGROW>
            'rank_dS',rank(dS,tol),'old_border_bound',2*st.nC, ...
            'rank_dSpp',rank(dSpp,tol), ...
            'relative_change',norm(dS,'fro')/norm(st.S,'fro'), ...
            'pressure_block_energy',norm(dSpp,'fro')/norm(dS,'fro'));
    end
    Sprev = st.S; xr = st.K\st.b; u = xr(1:ctx.nU);
end
T = struct2table([rows{:}]);
assert(any(T.rank_dS>T.old_border_bound), ...
       'No update exceeded the constant-viscosity rank-2nC bound.');
assert(all(T.rank_dSpp>0),'The pressure-pressure Schur block did not move.');
writetable(T,fullfile(out_dir,'rank_summary.csv'));
fprintf('[rank] median rank(dS)=%g of nS=%d; old bound=%g\n', ...
        median(T.rank_dS),T.nS(1),median(T.old_border_bound));
