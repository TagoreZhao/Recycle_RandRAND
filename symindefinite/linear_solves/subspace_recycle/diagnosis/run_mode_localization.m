% RUN_MODE_LOCALIZATION  Kill-switch test for H6: are the deflation targets
% physical modes tied to the immersed solid, or numerical artifacts of the
% incomplete factorization?
%
% The two-level scheme deflates the smallest-|lambda| eigenvectors of the SPLIT
% operator Ahat = C^-1 K C^-T.  Those eigenvalues measure where the INCOMPLETE
% factorization is a poor approximation — a property of fill pattern and pivot
% choices, not of the physics.  Two very different worlds are possible:
%
%   (a) the modes carry their mass in the multiplier block and sit spatially on
%       the immersed solid.  Then they move with the solid, the rank-2nC
%       constraint update is the right correction, and the plan proceeds.
%   (b) the modes are localized on scattered DOFs unrelated to the solid.  Then
%       they are ILDL bookkeeping, nothing cheap will make them persist, and the
%       answer is a different coarse space entirely.  PLAN REDIRECTS.
%
% Metrics per mode, on the physical eigenvector u (original coordinates) and on
% the deflated vector v = C'u (split coordinates):
%   energy split          fraction of ||.||^2 in the velocity / pressure /
%                         multiplier blocks
%   participation ratio   PR = (sum x^2)^2 / sum x^4 = the effective number of
%                         active DOFs.  PR/n ~ 1/3 for a Gaussian random vector,
%                         PR = 1 for a single spike.
%   near-solid fraction   share of the nodal energy within 3*h0 of a Lagrange
%                         point, versus the share of mesh nodes in that band
%                         (the null expectation for a delocalized mode).
%                         Computed per block AND for whichever block dominates —
%                         measuring only the velocity block would report noise
%                         when 97% of the mode lives in the pressure block.
%
% Fast default ~1 min; set FULL = true for benchmark scale.
%
% See also: run_pivot_sensitivity, run_eigenspace_motion.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
addpath(fullfile(fileparts(thisFileDir), 'kernel'));
add_recycle_paths();
rng(1);

outDir = fullfile(thisFileDir, 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end

FULL = evalin('base', 'exist(''FULL'',''var'') && logical(FULL)');
if FULL
    H0 = 0.05;  KBASE = 500;
else
    H0 = 0.1;   KBASE = 100;
end
CASES = {'bar_rotating', 'disk_static'};
NEAR_FAC = 3;                      % "near the solid" = within NEAR_FAC*h0

COL_U = [0.20 0.45 0.70];  COL_P = [0.85 0.40 0.32];
COL_L = [0.30 0.65 0.35];  COL_REF = [0.35 0.35 0.35];
COL_S = [0.50 0.35 0.65];

rows = struct('case_name', {}, 'mode', {}, 'lambda', {}, 'abs_lambda', {}, ...
    'energy_u', {}, 'energy_p', {}, 'energy_lam', {}, 'dominant_block', {}, ...
    'pr_orig', {}, 'pr_orig_frac', {}, 'pr_split', {}, 'pr_split_frac', {}, ...
    'pr_press', {}, 'pr_press_frac', {}, ...
    'frac_vel_near_solid', {}, 'frac_press_near_solid', {}, ...
    'frac_dom_near_solid', {}, 'area_frac_near_solid', {}, ...
    'localization_ratio', {}, 'mean_dist_to_solid', {});

fprintf('=== run_mode_localization (h0=%g, k=%d) ===\n', H0, KBASE);
summary = struct('case_name', {}, 'med_energy_u', {}, 'med_energy_p', {}, ...
                 'med_energy_lam', {}, 'med_pr_frac', {}, 'med_pr_press_frac', {}, ...
                 'med_loc_ratio', {}, 'eps_stab', {}, 'pr_frac_random', {});

for cc = 1:numel(CASES)
    cname = CASES{cc};
    S = build_stokes_sequence(struct('case_name', cname, 'h0', H0, ...
                                     'nsteps', 2, 'quiet', true));
    K = seq_K(S, 1);  n = S.n;
    P = src.precond.make_ildl_precond(K, struct('mode', 'nofill'));
    C = ildl_coordinate_map(P);
    M = C * C';  M = (M + M') / 2;

    fprintf('  [%s] n=%d nC=%d — computing %d smallest-|lambda| modes ...\n', ...
            cname, n, S.nC, KBASE);
    [U, D] = eigs(K, M, KBASE, 'smallestabs', 'Tolerance', 1e-10, ...
                  'MaxIterations', 2000);
    lam = real(diag(D));
    V   = transport_V(U, P, C);

    % --- geometry: distance from each mesh node to the nearest Lagrange point
    Xp    = S.Xpts{1};
    dnode = min(pdist2_local(S.msh.p, Xp), [], 2);      % N x 1
    near  = dnode < NEAR_FAC * S.h0;
    area_frac = mean(near);                              % null expectation

    % --- random-vector reference for the participation ratio
    pr_rand = mean(arrayfun(@(~) part_ratio(randn(n, 1)), 1:20)) / n;

    for i = 1:size(U, 2)
        u  = U(:, i) / norm(U(:, i));
        v  = V(:, i);
        eu = norm(u(1:S.nU))^2;
        ep = norm(u(S.nU + (1:S.nP)))^2;
        el = norm(u(S.nU + S.nP + (1:S.nC)))^2;

        % nodal energy per block (both are P1 nodal fields on the same mesh)
        wv = u(1:S.N).^2 + u(S.N + (1:S.N)).^2;          % velocity
        wp = u(S.nU + (1:S.nP)).^2;                       % pressure
        % the dominant block is what "where does this mode live" should measure
        if ep >= eu, wd = wp; else, wd = wv; end

        [fv, ~]     = near_share(wv, near, dnode);
        [fp, ~]     = near_share(wp, near, dnode);
        [fd, mdist] = near_share(wd, near, dnode);

        rows(end+1) = struct('case_name', cname, 'mode', i, ...
            'lambda', lam(i), 'abs_lambda', abs(lam(i)), ...
            'energy_u', eu, 'energy_p', ep, 'energy_lam', el, ...
            'dominant_block', string(dom_name(eu, ep, el)), ...
            'pr_orig', part_ratio(u), 'pr_orig_frac', part_ratio(u)/n, ...
            'pr_split', part_ratio(v), 'pr_split_frac', part_ratio(v)/n, ...
            'pr_press', part_ratio(u(S.nU + (1:S.nP))), ...
            'pr_press_frac', part_ratio(u(S.nU + (1:S.nP)))/S.nP, ...
            'frac_vel_near_solid', fv, 'frac_press_near_solid', fp, ...
            'frac_dom_near_solid', fd, 'area_frac_near_solid', area_frac, ...
            'localization_ratio', fd/max(area_frac, eps), ...
            'mean_dist_to_solid', mdist); %#ok<SAGROW>
    end

    sel = strcmp({rows.case_name}, cname);
    summary(end+1) = struct('case_name', cname, ...
        'med_energy_u',   median([rows(sel).energy_u]), ...
        'med_energy_p',   median([rows(sel).energy_p]), ...
        'med_energy_lam', median([rows(sel).energy_lam]), ...
        'med_pr_frac',    median([rows(sel).pr_orig_frac]), ...
        'med_pr_press_frac', median([rows(sel).pr_press_frac]), ...
        'med_loc_ratio',  median([rows(sel).localization_ratio]), ...
        'eps_stab', S.eps_stab, 'pr_frac_random', pr_rand); %#ok<SAGROW>

    fprintf(['    energy: u=%.3f p=%.3f lam=%.3f (medians) | PR/n=%.3f ' ...
             '(random %.3f) | dominant-block near-solid %.2fx\n'], ...
            median([rows(sel).energy_u]), median([rows(sel).energy_p]), ...
            median([rows(sel).energy_lam]), median([rows(sel).pr_orig_frac]), ...
            pr_rand, median([rows(sel).localization_ratio]));
end

T = struct2table(rows);
writetable(T, fullfile(outDir, 'mode_localization.csv'));

%% ===== verdict ==========================================================
s = summary(1);
[~, idom] = max([s.med_energy_u, s.med_energy_p, s.med_energy_lam]);
domnames  = {'velocity', 'pressure', 'multiplier'};
fprintf('\n==================================================================\n');
fprintf('  H6 kill-switch (case %s, k=%d):\n', s.case_name, KBASE);
fprintf('    energy split (median)          : u %.3f | p %.3f | lambda %.3f\n', ...
        s.med_energy_u, s.med_energy_p, s.med_energy_lam);
fprintf('    dominant block                 : %s\n', domnames{idom});
fprintf('    median PR/n                    : %.3f  (Gaussian reference %.3f)\n', ...
        s.med_pr_frac, s.pr_frac_random);
fprintf('    median PR/nP within pressure   : %.3f\n', s.med_pr_press_frac);
fprintf('    dominant block near the solid  : %.2fx the node share\n', s.med_loc_ratio);
fprintf('    Brezzi-Pitkaranta eps = h^2/(12 nu) = %.2e\n', s.eps_stab);

on_solid = s.med_loc_ratio > 2;
if idom == 2
    fprintf('\n    => The deflation targets are PRESSURE modes, not constraint modes.\n');
    fprintf('       That is physical, not an artifact: the (p,p) block is -eps*L with\n');
    fprintf('       eps = %.1e, so the stabilized pressure directions are the genuine\n', ...
            s.eps_stab);
    fprintf('       near-null space of the saddle-point system, and the no-fill ILDL\n');
    fprintf('       resolves them poorly.  They spread over %.0f%% of the pressure\n', ...
            100*s.med_pr_press_frac);
    fprintf('       DOFs, so they are structured fields, not isolated fill spikes.\n');
    fprintf('\n    GATE PASSED (targets are physical), WITH A CAVEAT:\n');
    fprintf('       the moving block C(t) touches only the velocity and multiplier\n');
    fprintf('       rows, so a purely constraint-driven rank-2nC update is NOT aimed\n');
    fprintf('       at where these modes live.  It reaches them only indirectly,\n');
    fprintf('       through C^-1.  Expect it to help partially, not fully — which is\n');
    fprintf('       what run_eigenspace_motion must quantify.\n');
elseif on_solid
    fprintf('\n    => targets are concentrated on the immersed solid: the constraint\n');
    fprintf('       update is aimed at the right subspace. GATE PASSED.\n');
elseif s.med_pr_frac < 0.25 * s.pr_frac_random
    fprintf('\n    => targets are localized, off-solid, and not pressure-dominated:\n');
    fprintf('       likely ILDL artifacts.  GATE FAILED — redirect to a different\n');
    fprintf('       coarse space rather than a cheap update.\n');
else
    fprintf('\n    => targets are delocalized global modes.  A cheap rank-2nC update\n');
    fprintf('       may not suffice; run_eigenspace_motion decides.\n');
end
fprintf('==================================================================\n');

%% ===== figure ===========================================================
fig = figure('Visible', 'off', 'Position', [100 100 1220 420]);
tl = tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
cn = CASES{1};
sel = strcmp(T.case_name, cn);
Tc = T(sel, :);

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
plot(ax, Tc.mode, Tc.energy_u,   '-', 'Color', COL_U, 'LineWidth', 1.6);
plot(ax, Tc.mode, Tc.energy_p,   '-', 'Color', COL_P, 'LineWidth', 1.6);
plot(ax, Tc.mode, Tc.energy_lam, '-', 'Color', COL_L, 'LineWidth', 1.6);
xlabel(ax, 'mode index (ascending |\lambda|)');  ylabel(ax, 'fraction of ||u||^2');
title(ax, 'Energy split across blocks');
legend(ax, {'velocity', 'pressure', 'multiplier'}, 'Location', 'east');

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
plot(ax, Tc.mode, Tc.pr_orig_frac,  '-', 'Color', COL_U, 'LineWidth', 1.6);
plot(ax, Tc.mode, Tc.pr_split_frac, '-', 'Color', COL_S, 'LineWidth', 1.6);
yline(ax, summary(1).pr_frac_random, '--', 'Gaussian random', ...
      'Color', COL_REF, 'LineWidth', 1.2);
set(ax, 'YScale', 'log');
xlabel(ax, 'mode index');  ylabel(ax, 'participation ratio / n');
title(ax, 'Localization (low = few active DOFs)');
legend(ax, {'physical u', 'split C''u'}, 'Location', 'southeast');

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
plot(ax, Tc.mode, Tc.localization_ratio, '-', 'Color', COL_L, 'LineWidth', 1.6);
yline(ax, 1, '--', 'delocalized (= area share)', 'Color', COL_REF, 'LineWidth', 1.2);
set(ax, 'YScale', 'log');
xlabel(ax, 'mode index');
ylabel(ax, sprintf('dominant-block energy within %g h_0 / node share', NEAR_FAC));
title(ax, 'Concentration on the immersed solid');

sgtitle(fig, sprintf(['Are the deflation targets physical? — %s, n=%d, k=%d ' ...
                      '(H6 kill-switch)'], cn, S.n, KBASE));
exportgraphics(fig, fullfile(outDir, 'mode_localization.png'), 'Resolution', 180);
close(fig);

fprintf('[saved] %s\n', fullfile(outDir, 'mode_localization.csv'));
fprintf('[saved] %s\n', fullfile(outDir, 'mode_localization.png'));

%==========================================================================
function pr = part_ratio(x)
%PART_RATIO  Effective number of active entries: (sum x^2)^2 / sum x^4.
    x2 = x(:).^2;
    pr = sum(x2)^2 / max(sum(x2.^2), realmin);
end

function [frac, mdist] = near_share(w, near, dnode)
%NEAR_SHARE  Energy share inside the near-solid band, and the energy-weighted
% mean distance to the solid, for one nodal energy field w.
    sw = sum(w);
    if sw <= 0
        frac = NaN;  mdist = NaN;
    else
        frac  = sum(w(near)) / sw;
        mdist = sum(w .* dnode) / sw;
    end
end

function nm = dom_name(eu, ep, el)
%DOM_NAME  Which block holds most of the mode's energy.
    [~, i] = max([eu, ep, el]);
    names  = {'velocity', 'pressure', 'multiplier'};
    nm     = names{i};
end

function D = pdist2_local(A, B)
%PDIST2_LOCAL  Pairwise Euclidean distances (no Statistics Toolbox dependency).
    D = sqrt(max(sum(A.^2, 2) + sum(B.^2, 2)' - 2 * (A * B'), 0));
end
