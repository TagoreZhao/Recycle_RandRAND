% RUN_SCHUR_LOWRANK  How low-rank is the step-to-step motion of S, really?
%
% Structurally, S_n - S_m must be ZERO throughout the (p,p) block and have rank
% at most 2*nC, because only C(t) moves and the pressure block was hoisted out
% of the time loop.  That makes this the friendliest imaginable operator for
% recycling -- IF the update is also small in magnitude.
%
% The benchmark already hints it is not: ReldiffF ~ 0.1-0.2 per step here versus
% 0.006-0.017 for the sphere reference experiment.  So this script separates the
% two questions:
%
%   (1) is the update confined and low-rank?          -> hard assertions
%   (2) is it SMALL?                                  -> singular-value decay
%
% A rank-40 update inside a ~540-dimensional operator is only 7% of the
% directions, but if those directions carry a large share of the norm, a cached
% basis still goes stale.  That is the tension this measures.
%
% Outputs -> schur_recycle/lowrank/

clearvars; clc;
paths = add_schur_paths();
assert_local_helpers();
rng(1);

params = make_schur_params();
params.h0 = 0.1;               % keep every S in memory; structure is size-free
nsteps    = 20;

out_dir = fullfile(paths.outDir, 'lowrank');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

cfg = schur_make_cfg('bar_rotating', params, []);
ctx = schur_context_init(cfg, params);
nP  = ctx.nP;

S_all  = cell(nsteps, 1);
nC_all = zeros(nsteps, 1);
u_prev = zeros(ctx.nU, 1);

for n = 1:nsteps
    st = schur_step_operator(ctx, n * params.dt, u_prev);
    S_all{n}  = st.S;
    nC_all(n) = st.nC;
    u_prev = st.K \ st.b;
    u_prev = u_prev(1:ctx.nU);
end

nS = size(S_all{1}, 1);
fprintf('Schur low-rank structure: nS = %d, nC = %d, nP-1 = %d\n', ...
        nS, nC_all(1), nP - 1);

%% --- (1) confinement + rank, against the structural bound ----------------
fprintf('\n lag |  rank  bound |  ||dS||_F/||S||_F | energy in the border block\n');
lags = [1 2 5 10 19];
rows = {};
for L = lags
    r_list = []; nrm_list = []; frac_list = [];
    for n = (L+1):nsteps
        if size(S_all{n},1) ~= size(S_all{n-L},1), continue; end
        dS = S_all{n} - S_all{n-L};

        % hard assertion: the pressure block is EXACTLY untouched
        blk_pp = dS(1:nP-1, 1:nP-1);
        assert(max(abs(blk_pp(:))) == 0, ...
            'VIOLATION: (p,p) block moved by %.3e at lag %d, step %d', ...
            max(abs(blk_pp(:))), L, n);

        r = rank(dS, 1e-10 * norm(S_all{n}, 'fro'));
        assert(r <= 2 * nC_all(n), ...
            'VIOLATION: rank %d exceeds 2*nC = %d at lag %d, step %d', ...
            r, 2 * nC_all(n), L, n);

        r_list(end+1)   = r;                              %#ok<SAGROW>
        nrm_list(end+1) = norm(dS, 'fro') / norm(S_all{n}, 'fro'); %#ok<SAGROW>
        % share of the total Frobenius energy that sits in the moving border
        border = dS;  border(1:nP-1, 1:nP-1) = 0;
        frac_list(end+1) = norm(border,'fro') / max(norm(dS,'fro'), eps); %#ok<SAGROW>
    end
    fprintf(' %3d | %5.1f  %5d |          %8.4f | %8.4f\n', ...
            L, median(r_list), 2*median(nC_all), median(nrm_list), median(frac_list));
    rows{end+1} = struct('lag', L, 'median_rank', median(r_list), ...
        'rank_bound', 2*median(nC_all), 'median_relnorm', median(nrm_list)); %#ok<SAGROW>
end
fprintf('\nASSERTIONS HELD: every update is confined to the multiplier border\n');
fprintf('and has rank <= 2*nC at every lag tested.\n');

writetable(struct2table([rows{:}]), fullfile(out_dir, 'lowrank_by_lag.csv'));

%% --- (2) is the low-rank update actually SMALL? --------------------------
dS1 = S_all{end} - S_all{1};
sv  = svd(dS1);
sv  = sv / sv(1);
keep = find(sv > 1e-12);

nrm_S = norm(S_all{end}, 'fro');
fprintf('\n||S_20 - S_1||_F / ||S_20||_F = %.4f  (numerical rank %d of %d)\n', ...
        norm(dS1,'fro')/nrm_S, numel(keep), nS);
fprintf('That is %.1f%% of the directions carrying %.1f%% relative change.\n', ...
        100*numel(keep)/nS, 100*norm(dS1,'fro')/nrm_S);

opts = benchmark_fig_defaults();
fh = figure('Visible','off','Units','inches', ...
            'Position',[1 1 opts.multi_width 4.0],'Color','w');
tl = tiledlayout(fh,1,2,'Padding','compact','TileSpacing','compact');

ax = nexttile(tl);
semilogy(ax, sv(keep), 'LineWidth', 1.8, 'Color', [0 0.45 0.70]);
hold(ax,'on');
xline(ax, 2*nC_all(end), '--', '2 n_C', 'Color', [0.84 0.37 0], ...
      'FontSize', 9, 'LabelVerticalAlignment','bottom');
grid(ax,'on');
xlabel(ax,'index'); ylabel(ax,'\sigma_i / \sigma_1');
title(ax,'singular values of S_{20} - S_1');

ax = nexttile(tl);
relnorm = nan(nsteps,1);
for n = 2:nsteps
    relnorm(n) = norm(S_all{n} - S_all{1}, 'fro') / norm(S_all{n}, 'fro');
end
plot(ax, 1:nsteps, relnorm, 'LineWidth', 1.8, 'Color', [0.35 0.35 0.35], ...
     'Marker','o','MarkerSize',4);
grid(ax,'on');
xlabel(ax,'time step'); ylabel(ax,'||S_n - S_1||_F / ||S_n||_F');
title(ax,'magnitude of the rank-\leq2n_C drift');

title(tl, 'Low-rank structure of the Schur-complement sequence', ...
      'FontSize', opts.titlefontsize);
save_benchmark_figure(fh, fullfile(out_dir,'lowrank_structure.png'), opts);

fprintf('\n[run_schur_lowrank] wrote %s\n', out_dir);
