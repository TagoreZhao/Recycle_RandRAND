% RUN_VARVISC_SCHUR_SPECTRUM  Plot exact Schur-complement spectra.
%   Writes the actual ordered eigenvalue curves of S_n and of the same
%   operator seen through the frozen chol(S_1), plus the conditioning and
%   deflation-width summaries. All eigenvalues come from dense eig(), not an
%   estimate or a condition-number surrogate.
clearvars; clc;
paths = add_varvisc_schur_paths(); rng(1);
params = make_varvisc_schur_params();
h0_list = [0.10 0.07 0.05 0.03]; snapshot_steps = [1 3 6];
k_list = [5 10 20 30 50 100 200]; rows = {}; krows = {};
spec_store = struct('h0',{},'step',{},'ev',{},'evR',{});
out_dir = fullfile(paths.outDir,'spectrum');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

for hi = 1:numel(h0_list)
    p = params; p.h0 = h0_list(hi);
    cfg = varvisc_schur_make_cfg('bar_rotating_nu_orbiting',p,[]);
    ctx = varvisc_schur_context_init(cfg,p); u = zeros(ctx.nU,1); R1 = [];
    for n = 1:max(snapshot_steps)
        st = varvisc_schur_step_operator(ctx,n*p.dt,u); S = st.to_dense();
        if isempty(R1), R1 = chol(S,'lower'); end
        if ismember(n,snapshot_steps)
            ev = sort(real(eig(S)),'ascend');
            Tprec = (R1\S)/R1'; Tprec = (Tprec+Tprec')/2;
            evR = sort(real(eig(Tprec)),'ascend');
            rows{end+1} = struct('h0',p.h0,'step',n,'nS',st.nS,'nC',st.nC, ... %#ok<AGROW>
                'lambda_min',ev(1),'lambda_max',ev(end),'kappa',ev(end)/ev(1), ...
                'kappa_frozen_chol',max(evR)/min(evR));
            spec_store(end+1) = struct('h0',p.h0,'step',n, ...
                'ev',ev,'evR',evR); %#ok<SAGROW>
            for k = k_list
                if k < numel(ev)
                    krows{end+1} = struct('h0',p.h0,'step',n,'k',k, ... %#ok<AGROW>
                        'kappa_deflated',ev(end)/ev(k+1));
                end
            end
        end
        xr = st.recover(S\st.rhs_S); u = xr(1:ctx.nU);
    end
end
Tsummary = struct2table([rows{:}]);
writetable(Tsummary,fullfile(out_dir,'spectrum_summary.csv'));
writetable(struct2table([krows{:}]),fullfile(out_dir,'deflated_kappa_vs_k.csv'));

%% Actual eigenvalue plots, matching the constant-viscosity Schur study.
Tk = struct2table([krows{:}]); opts = varvisc_schur_fig_defaults();

% Raw and frozen-Cholesky-preconditioned spectra at the finest mesh and last
% snapshot. The last snapshot is intentional: at step 1 the right panel is
% identically one because the factor is exact by construction.
sel = find([spec_store.h0]==h0_list(end) & ...
           [spec_store.step]==snapshot_steps(end),1);
fh = figure('Visible','off','Units','inches', ...
            'Position',[1 1 opts.multi_width 4],'Color','w');
tl = tiledlayout(fh,1,2,'Padding','compact','TileSpacing','compact');
ax = nexttile(tl);
semilogy(ax,spec_store(sel).ev,'LineWidth',1.6,'Color',[0 .45 .70]);
xlabel(ax,'ordered eigenvalue index'); ylabel(ax,'\lambda_i(S_n)');
title(ax,sprintf('raw S, h_0=%.2f, step %d', ...
      h0_list(end),snapshot_steps(end)));
ax = nexttile(tl);
semilogy(ax,spec_store(sel).evR,'LineWidth',1.6,'Color',[.84 .37 0]);
xlabel(ax,'ordered eigenvalue index');
ylabel(ax,'\lambda_i(L_1^{-1}S_nL_1^{-T})');
title(ax,'through frozen chol(S_1)');
title(tl,'Actual Schur-complement spectra','FontSize',opts.titlefontsize);
save_varvisc_schur_figure(fh,fullfile(out_dir,'spectrum_raw_vs_prec.png'),opts);

% Raw spectra at all requested time snapshots on the finest mesh, so movement
% of the full eigenvalue distribution is visible rather than reduced to kappa.
fh = figure('Visible','off','Units','inches', ...
            'Position',[1 1 opts.multi_width 4],'Color','w'); ax = axes(fh);
colors = lines(numel(snapshot_steps));
for si = 1:numel(snapshot_steps)
    jj = find([spec_store.h0]==h0_list(end) & ...
              [spec_store.step]==snapshot_steps(si),1);
    semilogy(ax,spec_store(jj).ev,'LineWidth',1.5,'Color',colors(si,:), ...
        'DisplayName',sprintf('step %d',snapshot_steps(si))); hold(ax,'on');
end
xlabel(ax,'ordered eigenvalue index'); ylabel(ax,'\lambda_i(S_n)');
title(ax,sprintf('Raw Schur spectrum over time, h_0=%.2f',h0_list(end)));
legend(ax,'Location','best');
save_varvisc_schur_figure(fh,fullfile(out_dir,'spectrum_raw_snapshots.png'),opts);

%% Conditioning summaries.
fh = figure('Visible','off','Units','inches', ...
            'Position',[1 1 opts.multi_width 3.8],'Color','w');
tl = tiledlayout(fh,1,3,'Padding','compact','TileSpacing','compact');

ax = nexttile(tl);
k0 = arrayfun(@(h) Tsummary.kappa(find(Tsummary.h0==h & ...
    Tsummary.step==snapshot_steps(1),1)),h0_list);
loglog(ax,h0_list,k0,'-o','LineWidth',1.6);
set(ax,'XDir','reverse'); xlabel(ax,'h_0'); ylabel(ax,'\kappa(S)');
title(ax,'mesh dependence');

ax = nexttile(tl);
for hi = 1:numel(h0_list)
    m = Tk.h0==h0_list(hi) & Tk.step==snapshot_steps(1);
    loglog(ax,Tk.k(m),Tk.kappa_deflated(m),'-o','LineWidth',1.6, ...
        'DisplayName',sprintf('h0=%.2f',h0_list(hi))); hold(ax,'on');
end
xlabel(ax,'coarse-space width'); ylabel(ax,'\lambda_{max}/\lambda_{k+1}');
title(ax,'deflated conditioning'); legend(ax,'Location','best');

ax = nexttile(tl);
for hi = 1:numel(h0_list)
    m = Tsummary.h0==h0_list(hi);
    semilogy(ax,Tsummary.step(m),Tsummary.kappa_frozen_chol(m),'-o', ...
        'LineWidth',1.6,'DisplayName',sprintf('h0=%.2f',h0_list(hi))); hold(ax,'on');
end
xlabel(ax,'time step'); ylabel(ax,'\kappa(L_1^{-1}S_nL_1^{-T})');
title(ax,'frozen-factor staleness'); legend(ax,'Location','best');
title(tl,'Schur conditioning diagnostics','FontSize',opts.titlefontsize);
save_varvisc_schur_figure(fh,fullfile(out_dir,'conditioning_summary.png'),opts);

fprintf('[varvisc_schur_spectrum] wrote actual spectra to %s\n',out_dir);
