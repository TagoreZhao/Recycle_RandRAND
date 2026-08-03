function [V, D] = exp6_cost_vs_angle(opts)
%EXP6_COST_VS_ANGLE  What a wrong subspace costs, as a function of one angle.
%Tests Thm 4.1 (the closed form), Cor 4.2 (the MINRES rate) and Obs 4.3
%(deflation can be strictly worse than no deflation).
%
%   [V, D] = EXP6_COST_VS_ANGLE(OPTS)
%
%   Part A -- the closed form.  With Ahat = diag(l1,l2), a one-dimensional
%   coarse space at angle theta from the target, c = cos, s = sin, and
%   G = (I - vv') + sqrt(tau) (v'Ahat^2 v)^{-1/2} vv' the production coarse
%   correction, the operator MINRES sees is B = G*Ahat, so
%
%       E     = l1^2 c^2 + l2^2 s^2,      beta  = sqrt(tau / E),
%       alpha = l1 c^2 + l2 s^2,
%       trace = l1 + l2 + (beta - 1) alpha,      det = beta * l1 * l2,
%
%   and the two eigenvalues are the roots of mu^2 - trace*mu + det.  At
%   theta = 0 they are {sqrt(tau)*sign(l1), l2}: an exactly captured mode is
%   moved to +-sqrt(tau), the textbook deflation target.  Checked against the
%   assembled operator to 1e-12.
%
%   The threshold that matters: E stops being l1^2-dominated once tan(theta)
%   exceeds |l1/l2|, after which beta ~ sqrt(tau)/(l2 s) and the small
%   eigenvalue decays like sqrt(tau)*l1/(s*l2).  The tolerance on the coarse
%   space is therefore O(|l1/l2|) -- proportional to the very ratio that made
%   deflation worth doing.  That is why the failure is silent: an angle far too
%   small to look wrong is already far too large to work.
%
%   Part B -- the same sweep at realistic size, with MINRES iteration counts and
%   the undeflated baseline drawn as a line, so the crossing point where the
%   coarse space starts costing more than it saves is visible.
%
%   Writes figures/cost_vs_angle.png.
%
%   See also: deflated_spectrum, src.precond.deflation_Psqrt_apply.

    if nargin < 1 || isempty(opts), opts = struct(); end
    p   = add_paths();
    tau = getdef(opts, 'tau', 0.5);
    rng(0);

    V = struct([]);  D = struct();

    %% ---- Part A: 2x2 closed form ----------------------------------------
    l1 = 0.01;  l2 = 1.0;
    th = [0, logspace(-5, log10(pi/2 - 1e-6), 60)];   % log grid: the threshold
                                                      % sits at ~1e-2, and a
                                                      % linear grid would miss it
    err = 0;  rows = zeros(numel(th), 5);
    for t = 1:numel(th)
        c = cos(th(t));  s = sin(th(t));
        E  = l1^2*c^2 + l2^2*s^2;   al = l1*c^2 + l2*s^2;   be = sqrt(tau/E);
        tr = l1 + l2 + (be - 1)*al;     dt = be*l1*l2;
        mu = roots([1 -tr dt]);
        o  = deflated_spectrum(diag([l1 l2]), [c; s], tau);
        err = max(err, norm(sort(real(mu)) - sort(o.lam), inf) / max(abs(o.lam)));
        rows(t, :) = [th(t), min(abs(o.lam)), max(abs(o.lam)), o.kappa, dt];
    end
    D.model = array2table(rows, 'VariableNames', ...
        {'theta', 'min_abs_lam', 'max_abs_lam', 'kappa', 'det'});

    % The discriminator is min|mu|: how close to zero the deflated spectrum
    % still comes.  E stops being l1^2-dominated once tan(theta) exceeds
    % |l1/l2|, and that is where min|mu| starts to fall.
    kap0 = abs(l2 / l1);                              % undeflated kappa
    thr  = atan(abs(l1 / l2));                        % the predicted onset
    half = rows(1, 2) / 2;
    i_h  = find(rows(:, 2) < half, 1);
    th_half = NaN;  if ~isempty(i_h), th_half = th(i_h); end
    i_lost  = find(rows(:, 2) < 2*abs(l1), 1);        % tiny eigenvalue fully back
    th_lost = NaN;  if ~isempty(i_lost), th_lost = th(i_lost); end

    V = [V, vrec('exp6A', 'Thm 4.1 closed form matches the assembled operator', ...
                 'max rel. eigenvalue error over theta', err, '< 1e-12', err < 1e-12)];
    pr0 = sort(abs([sqrt(tau)*sign(l1), l2]));
    V = [V, vrec('exp6A', 'Thm 4.1 theta=0 gives {sqrt(tau) sign(l1), l2}', ...
                 'deflated |eigenvalues| at theta=0', ...
                 sprintf('%.4g, %.4g (pred %.4g, %.4g)', ...
                         rows(1,2), rows(1,3), pr0(1), pr0(2)), ...
                 'equal', abs(rows(1,2) - pr0(1)) < 1e-10 && ...
                          abs(rows(1,3) - pr0(2)) < 1e-10)];
    V = [V, vrec('exp6A', 'Thm 4.1 degradation sets in at theta of order |l1/l2|', ...
                 'half-loss angle | atan|l1/l2|', ...
                 sprintf('%.4g | %.4g', th_half, thr), 'same order (within 30x)', ...
                 ~isnan(th_half) && th_half < 30*thr && th_half > thr/30)];
    V = [V, vrec('exp6A', 'Thm 4.1 total loss needs theta = O(1)', ...
                 'theta where min|mu| falls back to 2|l1|', th_lost, ...
                 'O(1), not O(|l1/l2|)', NaN)];
    fprintf(['[exp6A] closed form err %.1e; kappa: theta=0 -> %.4g, ' ...
             'undeflated %.4g; half-loss at theta=%.2e, total loss at %.2e ' ...
             '(atan|l1/l2|=%.2e)\n'], err, rows(1,4), kap0, th_half, th_lost, thr);

    %% ---- Part B: the real operator, real solver, iterations --------------
    % The immersed-rotor KKT with its own ILDL chart, and the EXACT coarse space
    % rotated away from the truth by a controlled angle theta.  Using the real
    % system rather than a synthetic diagonal keeps the iteration counts
    % meaningful: a contrived spectrum makes MINRES stagnate for reasons that
    % have nothing to do with the coarse space.
    cs  = make_case('ildl', 1, opts);
    n   = cs.n;  kk = getdef(opts, 'k', 20);
    U   = pencil_subspace(cs.A, cs.M, kk, opts);
    Vex = orth_trunc(cs.C' * U);
    Wc  = randn(n, kk);
    Wc  = orth_trunc(Wc - Vex * (Vex' * Wc));      % an orthogonal complement
    b   = ones(n, 1);

    thB = [0 logspace(-5, log10(pi/2 - 1e-3), 12)];
    rowsB = zeros(numel(thB), 4);
    [~, ~, ~, it_none] = src.precond.two_level_split_solve(cs.A, b, 1e-8, 3000, cs, [], tau);
    for t = 1:numel(thB)
        Vt = orth_trunc(Vex * cos(thB(t)) + Wc * sin(thB(t)));
        [~, ~, ~, it] = src.precond.two_level_split_solve(cs.A, b, 1e-8, 3000, cs, Vt, tau);
        ang = gap(Vt, Vex);
        rowsB(t, :) = [thB(t), ang, it, it_none];
    end
    D.sweep = array2table(rowsB, 'VariableNames', ...
        {'theta', 'gap_to_truth', 'iterations', 'iterations_none'});
    disp('[exp6B] deflation quality vs coarse-space angle (real KKT + ILDL)');
    disp(D.sweep);

    worse = any(rowsB(:, 3) > it_none);
    i_w   = find(rowsB(:, 3) > it_none, 1);
    i_h   = find(rowsB(:, 3) > 1.5 * rowsB(1, 3), 1);
    mono  = all(diff(rowsB(:, 3)) >= -2);          % non-decreasing up to noise
    V = [V, vrec('exp6B', 'Obs 4.3 an inexact coarse space can be worse than none', ...
                 'its(no deflation) | first theta that loses | its there', ...
                 sprintf('%d | %.2e | %d', it_none, ...
                         nanget(thB, i_w), nanget(rowsB(:,3), i_w)), ...
                 'a losing theta exists', worse)];
    V = [V, vrec('exp6B', 'cost is monotone in the angle to the truth', ...
                 'its non-decreasing in theta (tol 2)', mono, 'true', mono)];
    V = [V, vrec('exp6B', 'degradation with angle is gradual, not a cliff', ...
                 'exact its | first theta with 1.5x | no-deflation its', ...
                 sprintf('%d | %.2e | %d', rowsB(1,3), nanget(thB, i_h), it_none), ...
                 'the failure mode is a saturated angle, not a small one', NaN)];
    fprintf('[exp6B] undeflated %d its; exact coarse space %d its; worst %d its\n', ...
            it_none, rowsB(1, 3), max(rowsB(:, 3)));

    %% ---- figure ----------------------------------------------------------
    fh = figure('Visible', 'off');
    tl = tiledlayout(fh, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile(tl);
    loglog(D.model.theta(2:end), D.model.min_abs_lam(2:end), '-', 'LineWidth', 2, ...
           'DisplayName', 'min|\mu| after deflation');  hold on;
    yline(abs(l1), 'k--', 'LineWidth', 1.5, 'DisplayName', 'min|\lambda| = |\lambda_1|, no deflation');
    xline(thr,     'r:',  'LineWidth', 1.5, 'DisplayName', '\theta = atan|\lambda_1/\lambda_2|');
    xlabel('coarse-space angle \theta  (rad)');
    ylabel('smallest |eigenvalue| of G A_{hat}');
    title(sprintf('2\\times2 model: \\lambda = (%.2g, %.2g), \\tau = %.2g', l1, l2, tau));
    legend('Location', 'southwest');  hold off;

    nexttile(tl);
    semilogx(max(D.sweep.theta, 1e-6), D.sweep.iterations, '-o', 'LineWidth', 2, ...
             'DisplayName', 'MINRES, two-level');  hold on;
    yline(it_none, 'k--', 'LineWidth', 1.5, 'DisplayName', 'MINRES, no coarse space');
    xlabel('coarse-space angle \theta  (rad)');  ylabel('MINRES iterations');
    title(sprintf('immersed-rotor KKT + ILDL: n = %d, k = %d, \\tau = %.2g', n, kk, tau));
    legend('Location', 'northwest');  hold off;

    save_figure(fh, 'cost_vs_angle');
    writetable(D.model, fullfile(p.outDir, 'exp6_model.csv'));
    writetable(D.sweep, fullfile(p.outDir, 'exp6_sweep.csv'));
end

function v = nanget(x, i)
    if isempty(i), v = NaN; else, v = x(i); end
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
