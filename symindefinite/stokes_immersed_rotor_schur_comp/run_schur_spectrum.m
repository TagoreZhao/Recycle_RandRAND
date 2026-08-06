% RUN_SCHUR_SPECTRUM  Exact spectra of S(t_n), and how fast the frozen chol goes stale.
%
% S is dense and modest, so the spectrum comes from a full `eig` -- no eigs
% tolerance games, no shift-invert, no lower bounds.  That is the main practical
% payoff of forming S explicitly.
%
% RUN THIS BEFORE TRUSTING sm_eig.  Two numbers decide the benchmark:
%
%   (1) the DEFLATED condition number as a function of the coarse-space width,
%       kappa_defl(k) = lam_max / lam_{k+1}.  Deflating the k smallest modes
%       moves them to tau = lam_max, so this is exactly the conditioning a
%       width-k coarse space buys, and it is what params.sm_eig should be set
%       from.  The value in make_schur_params was inherited from an experiment
%       on a DIFFERENT operator (an ichol-preconditioned one) and means nothing
%       here.
%
%       A "count the eigenvalues below 1% of lam_max" threshold -- the metric
%       the ichol-era version of this script used -- is USELESS on the raw S:
%       measured, 83-99% of the spectrum sits below it, because S is not
%       preconditioned and its spectrum spans five orders of magnitude.  It
%       would report "deflate almost everything" at every mesh size.
%
%   (2) whether kappa(R^-1 S_n R^-T) grows away from 1, where R = chol(S_1) is
%       frozen.  That is the staleness of the `chol` baseline.  If it stayed at
%       1 the baseline would never degrade and there would be nothing for a
%       recycled basis to recover -- which is itself the result.  Note this is
%       NOT monotone in n: the bar sweeps two full revolutions, so S(t) moves
%       away from and back toward S_1.
%
% Outputs -> schur_recycle/spectrum/

clearvars; clc;
paths = add_schur_paths();
assert_local_helpers();
rng(1);

params = make_schur_params();
snapshot_steps = [1 15 30];
h0_list        = [0.10 0.07 0.05];

out_dir = fullfile(paths.outDir, 'spectrum');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

rows = {};
krows = {};                             % long table: deflated kappa vs width k
spec_store = struct('h0', {}, 'step', {}, 'ev', {}, 'evR', {});

k_list       = [5 10 20 30 50 100 200];     % candidate coarse-space widths
kappa_targets = [1e4 1e3 1e2];              % "what k do I need to reach this?"

for hi = 1:numel(h0_list)
    p = params;
    p.h0 = h0_list(hi);
    cfg = schur_make_cfg('bar_rotating', p, []);
    ctx = schur_context_init(cfg, p);

    Rfrozen = [];                       % chol(S_1), built once -- the baseline
    u_prev  = zeros(ctx.nU, 1);
    for n = 1:max(snapshot_steps)
        st = schur_step_operator(ctx, n * p.dt, u_prev);
        S  = st.S;

        if isempty(Rfrozen)
            [Rfrozen, cflag] = chol(S, 'lower');
            assert(cflag == 0, 'chol(S_1) failed -- S is not SPD at h0=%.3f.', p.h0);
        end

        if ismember(n, snapshot_steps)
            ev = sort(eig(S), 'ascend');

            % The BASELINE operator: S_n seen through the step-1 factor.  At
            % n = 1 this is the identity by construction (kappa = 1); every
            % later step measures how far S has moved away from it.
            R   = (Rfrozen \ S) / Rfrozen';
            R   = (R + R') / 2;
            evR = sort(eig(R), 'ascend');

            nS = size(S, 1);

            % kappa_defl(k) = lam_max / lam_{k+1}: the conditioning a width-k
            % coarse space buys, since deflation moves the k smallest modes to
            % tau = lam_max and leaves the rest alone.
            kappa_defl = @(k) ev(end) / ev(min(k, nS - 1) + 1);

            r = struct('h0', p.h0, 'nS', nS, 'nC', st.nC, 'step', n, ...
                'lam_min', ev(1), 'lam_max', ev(end), 'kappa', ev(end)/ev(1), ...
                'lam_min_R', evR(1), 'lam_max_R', evR(end), ...
                'kappa_R', evR(end)/evR(1));
            % smallest width reaching each kappa target (NaN = unreachable)
            for tg = kappa_targets
                kneed = find(ev(end) ./ ev(2:end) <= tg, 1);
                fname = sprintf('k_for_kappa_%g', tg);
                if isempty(kneed), r.(fname) = NaN; else, r.(fname) = kneed; end
            end
            rows{end+1} = r; %#ok<SAGROW>

            for k = k_list
                if k + 1 > nS, continue; end
                krows{end+1} = struct('h0', p.h0, 'nS', nS, 'step', n, ...
                    'k', k, 'kappa_defl', kappa_defl(k)); %#ok<SAGROW>
            end

            spec_store(end+1) = struct('h0', p.h0, 'step', n, ...
                                       'ev', ev, 'evR', evR); %#ok<SAGROW>

            fprintf(['h0=%.3f nS=%4d step %2d | kappa(S)=%9.3e' ...
                     '  kappa(frozen chol)=%9.3e | kappa_defl:'], ...
                p.h0, nS, n, ev(end)/ev(1), evR(end)/evR(1));
            for k = k_list
                if k + 1 > nS, continue; end
                fprintf('  k=%d:%8.2e', k, kappa_defl(k));
            end
            fprintf('\n');
        end

        u_prev = st.K \ st.b;
        u_prev = u_prev(1:ctx.nU);
    end
end

T = struct2table([rows{:}]);
writetable(T, fullfile(out_dir, 'spectrum_summary.csv'));

Tk = struct2table([krows{:}]);
writetable(Tk, fullfile(out_dir, 'deflated_kappa_vs_k.csv'));

%% ---------------- figures --------------------------------------------------
opts = benchmark_fig_defaults();

% (a) raw vs frozen-chol-preconditioned spectrum at the finest mesh, LAST snapshot
% (the last snapshot, not the first: at step 1 the frozen factor is exact and
% the right panel would be a flat line at 1 by construction)
sel = find([spec_store.h0] == h0_list(end) & ...
           [spec_store.step] == snapshot_steps(end), 1);
fh = figure('Visible','off','Units','inches', ...
            'Position',[1 1 opts.multi_width 4.0],'Color','w');
tl = tiledlayout(fh,1,2,'Padding','compact','TileSpacing','compact');

ax = nexttile(tl);
semilogy(ax, spec_store(sel).ev, 'LineWidth', 1.6, 'Color', [0 0.45 0.70]);
grid(ax,'on'); xlabel(ax,'index'); ylabel(ax,'\lambda_i(S)');
title(ax, sprintf('S, h0 = %.3f, step %d', h0_list(end), snapshot_steps(end)));

ax = nexttile(tl);
semilogy(ax, spec_store(sel).evR, 'LineWidth', 1.6, 'Color', [0.84 0.37 0]);
grid(ax,'on'); xlabel(ax,'index'); ylabel(ax,'\lambda_i(R^{-1} S R^{-T})');
title(ax, 'through the frozen chol(S_1)');
title(tl, 'Schur-complement spectrum, raw vs recycled exact factor', ...
      'FontSize', opts.titlefontsize);
save_benchmark_figure(fh, fullfile(out_dir,'spectrum_raw_vs_prec.png'), opts);

% (b) mesh refinement of kappa(S), what deflation buys, and frozen-factor staleness
fh = figure('Visible','off','Units','inches', ...
            'Position',[1 1 opts.multi_width 3.6],'Color','w');
tl = tiledlayout(fh,1,3,'Padding','compact','TileSpacing','compact');

ax = nexttile(tl);
k1 = arrayfun(@(h) T.kappa(find(T.h0==h & T.step==snapshot_steps(1),1)), h0_list);
loglog(ax, h0_list, k1, '-o', 'LineWidth', 1.8, 'DisplayName', '\kappa(S)');
grid(ax,'on'); set(ax,'XDir','reverse');
xlabel(ax,'h_0'); ylabel(ax,'condition number');
legend(ax,'Location','best','FontSize',opts.legendfontsize);
title(ax,'mesh dependence');

% What a width-k coarse space actually buys -- this is what sizes sm_eig.
ax = nexttile(tl);
for hi = 1:numel(h0_list)
    m = Tk.h0 == h0_list(hi) & Tk.step == snapshot_steps(1);
    loglog(ax, Tk.k(m), Tk.kappa_defl(m), '-o', 'LineWidth', 1.8, ...
           'DisplayName', sprintf('h_0 = %.2f', h0_list(hi)));
    hold(ax,'on');
end
grid(ax,'on'); xlabel(ax,'coarse-space width k'); ylabel(ax,'\lambda_{max} / \lambda_{k+1}');
legend(ax,'Location','best','FontSize',opts.legendfontsize);
title(ax,'deflated conditioning');

ax = nexttile(tl);
for hi = 1:numel(h0_list)
    m = T.h0 == h0_list(hi);
    semilogy(ax, T.step(m), T.kappa_R(m), '-o', 'LineWidth', 1.8, ...
             'DisplayName', sprintf('h_0 = %.2f', h0_list(hi)));
    hold(ax,'on');
end
grid(ax,'on'); xlabel(ax,'time step'); ylabel(ax,'\kappa(R^{-1} S_n R^{-T})');
legend(ax,'Location','best','FontSize',opts.legendfontsize);
title(ax,'staleness of the frozen chol(S_1)');
title(tl, 'Conditioning: mesh refinement, deflation width, operator drift', ...
      'FontSize', opts.titlefontsize);
save_benchmark_figure(fh, fullfile(out_dir,'kappa_mesh_refinement.png'), opts);

fprintf('\n[run_schur_spectrum] wrote %s\n', out_dir);
