function [V, D] = exp6_cost_vs_angle(opts)
%EXP6_COST_VS_ANGLE  What a wrong subspace costs, as a function of one angle.
%Tests Thm 4.1 (the closed form, both families), Cor 4.2 (the usable angle) and
%Obs 4.3 (deflation can be strictly worse than no deflation).
%
%   [V, D] = EXP6_COST_VS_ANGLE(OPTS)
%
%   Part A -- the closed forms.  With Ahat = diag(l1,l2), a one-dimensional
%   coarse space v at angle theta from the target, c = cos, s = sin, and
%   G = I + (beta-1)vv' the coarse correction, the operator the Krylov method
%   sees is B = G*Ahat, whose two eigenvalues are the roots of
%   mu^2 - trace*mu + det.  The two families differ in what the coarse matrix is
%   built from, and that is the whole of the difference:
%
%     indef (ildl)   E = l1^2 c^2 + l2^2 s^2      beta = sqrt(tau/E)
%                    trace = l1 + l2 + (beta-1)*alpha,  alpha = l1 c^2 + l2 s^2
%                    det   = beta*l1*l2
%                    theta = 0  ->  {sqrt(tau)*sign(l1), l2}
%
%     spd   (ichol)  E = l1 c^2 + l2 s^2          beta = tau/E
%                    alpha = E, so trace = l1 + l2 + tau - E
%                    det   = beta*l1*l2
%                    theta = 0  ->  {tau, l2}
%
%   Both checked against the assembled operator to 1e-12.
%
%   THE THRESHOLD, AND WHY IT DIFFERS.  E stops being l1-dominated when the s^2
%   term overtakes the c^2 term.  For the indefinite form that needs only
%   tan(theta) > |l1/l2|; for the SPD form it needs tan^2(theta) > l1/l2.  So the
%   usable angle is O(|l1/l2|) for indef and O(sqrt(l1/l2)) for spd -- the SQUARE
%   ROOT of the indefinite tolerance, hence far larger.  Part A2 measures the
%   log-log slope of the half-loss angle against l1/l2 over five decades and
%   expects 1.0 (indef) versus 0.5 (spd).
%
%   Squaring the operator is what buys definiteness when Ahat is indefinite --
%   V'Ahat V has no definite sign there, so the coarse solve would break.  Part A2
%   is the price of that trick, and it is why the SPD family must not pay it.
%
%   Part B -- the same sweep at realistic size, per family, with iteration counts
%   and the undeflated baseline drawn as a line, so the crossing point where the
%   coarse space starts costing more than it saves is visible.  ILDL runs MINRES
%   on the split operator; ichol runs PCG on K with B = C^-T P C^-1, each its own
%   production path.  Counts are therefore comparable WITHIN a panel, not across.
%
%   Writes figures/cost_vs_angle.png.
%
%   See also: deflated_spectrum, coarse_correction, two_level_solve_local.

    if nargin < 1 || isempty(opts), opts = struct(); end
    p   = add_paths();
    tau = getdef(opts, 'tau', 0.5);
    rng(0);

    V = struct([]);  D = struct();
    KINDS = {'indef', 'spd'};

    %% ---- Part A: the 2x2 closed forms -----------------------------------
    l1 = 0.01;  l2 = 1.0;
    th = [0, logspace(-5, log10(pi/2 - 1e-6), 60)];   % log grid: the thresholds
                                                      % sit at 1e-2 and 1e-1, and
                                                      % a linear grid misses both
    minmu = zeros(numel(th), 2);  maxmu = zeros(numel(th), 2);
    kap   = zeros(numel(th), 2);  err   = zeros(1, 2);
    for j = 1:2
        for t = 1:numel(th)
            mu = model_roots(l1, l2, tau, th(t), KINDS{j});
            o  = deflated_spectrum(diag([l1 l2]), [cos(th(t)); sin(th(t))], tau, KINDS{j});
            err(j) = max(err(j), norm(sort(real(mu)) - sort(o.lam), inf) / max(abs(o.lam)));
            minmu(t, j) = min(abs(o.lam));
            maxmu(t, j) = max(abs(o.lam));
            kap(t, j)   = o.kappa;
        end
    end
    D.model = array2table([th(:), minmu, maxmu, kap], 'VariableNames', ...
        {'theta', 'min_abs_lam_indef', 'min_abs_lam_spd', ...
                  'max_abs_lam_indef', 'max_abs_lam_spd', 'kappa_indef', 'kappa_spd'});

    thr  = [atan(abs(l1/l2)), atan(sqrt(l1/l2))];     % the predicted onsets
    th_half = zeros(1, 2);
    for j = 1:2
        i_h = find(minmu(:, j) < minmu(1, j)/2, 1);
        th_half(j) = nanget(th, i_h);
    end

    for j = 1:2
        V = [V, vrec(sprintf('exp6A/%s', KINDS{j}), ...
                     'Thm 4.1 closed form matches the assembled operator', ...
                     'max rel. eigenvalue error over theta', err(j), '< 1e-12', ...
                     err(j) < 1e-12)]; %#ok<AGROW>
    end
    % theta = 0: the captured mode lands on sqrt(tau)*sign(l1) (indef) or tau (spd).
    pr0 = {sort(abs([sqrt(tau)*sign(l1), l2])), sort(abs([tau, l2]))};
    lbl = {'sqrt(tau) sign(l1)', 'tau'};
    for j = 1:2
        V = [V, vrec(sprintf('exp6A/%s', KINDS{j}), ...
                     sprintf('Thm 4.1 theta=0 gives {%s, l2}', lbl{j}), ...
                     'deflated |eigenvalues| at theta=0', ...
                     sprintf('%.4g, %.4g (pred %.4g, %.4g)', ...
                             minmu(1,j), maxmu(1,j), pr0{j}(1), pr0{j}(2)), ...
                     'equal', abs(minmu(1,j) - pr0{j}(1)) < 1e-10 && ...
                              abs(maxmu(1,j) - pr0{j}(2)) < 1e-10)]; %#ok<AGROW>
        V = [V, vrec(sprintf('exp6A/%s', KINDS{j}), ...
                     'Cor 4.2 degradation sets in at the predicted onset', ...
                     'half-loss angle | predicted onset', ...
                     sprintf('%.4g | %.4g', th_half(j), thr(j)), ...
                     'same order (within 30x)', ...
                     ~isnan(th_half(j)) && th_half(j) < 30*thr(j) && ...
                                           th_half(j) > thr(j)/30)]; %#ok<AGROW>
    end
    V = [V, vrec('exp6A', 'the SPD form tolerates a larger coarse-space angle', ...
                 'half-loss angle spd / indef  at l1/l2 = 1e-2', ...
                 th_half(2) / th_half(1), '> 1', th_half(2) > th_half(1))];
    fprintf(['[exp6A] closed-form err: indef %.1e spd %.1e; half-loss angle: ' ...
             'indef %.2e (onset %.2e), spd %.2e (onset %.2e), ratio %.2f\n'], ...
            err(1), err(2), th_half(1), thr(1), th_half(2), thr(2), ...
            th_half(2)/th_half(1));

    %% ---- Part A2: how the usable angle scales with l1/l2 -----------------
    % The headline of the SPD-vs-indefinite comparison, and the one measurement
    % here that is a controlled slope rather than a single number.  Uses the
    % closed form, which Part A has just validated against the assembled
    % operator, so this costs nothing.
    rs   = logspace(-6, -1, 6);
    thf  = logspace(-7, log10(pi/2 - 1e-6), 2000);
    Hs   = zeros(numel(rs), 2);
    for ir = 1:numel(rs)
        for j = 1:2
            m = arrayfun(@(t) min(abs(model_roots(rs(ir)*l2, l2, tau, t, KINDS{j}))), thf);
            i_h = find(m < m(1)/2, 1);
            Hs(ir, j) = nanget(thf, i_h);
        end
    end
    D.scaling = array2table([rs(:), Hs], 'VariableNames', ...
        {'ratio_l1_l2', 'half_loss_indef', 'half_loss_spd'});
    sl = zeros(1, 2);
    for j = 1:2
        cf = polyfit(log10(rs(:)), log10(Hs(:, j)), 1);  sl(j) = cf(1);
    end
    pred_sl = [1.0, 0.5];
    for j = 1:2
        V = [V, vrec(sprintf('exp6A2/%s', KINDS{j}), ...
                     'Cor 4.2 usable angle scales as predicted in l1/l2', ...
                     'log-log slope of half-loss angle vs l1/l2', ...
                     sprintf('%.4f', sl(j)), sprintf('%.1f', pred_sl(j)), ...
                     abs(sl(j) - pred_sl(j)) < 0.1)]; %#ok<AGROW>
    end
    fprintf(['[exp6A2] half-loss angle vs l1/l2 over 5 decades: slope indef ' ...
             '%.4f (pred 1.0), spd %.4f (pred 0.5)\n'], sl(1), sl(2));
    disp(D.scaling);

    %% ---- Part B: the real operators, real solvers, iterations ------------
    % Each family with ITS OWN system, chart and coarse correction, and the EXACT
    % coarse space rotated away from the truth by a controlled angle.  Using the
    % real systems rather than synthetic diagonals keeps the iteration counts
    % meaningful.
    fams  = {'ildl', 'ichol'};
    kk    = getdef(opts, 'k', 20);
    thB   = [0 logspace(-5, log10(pi/2 - 1e-3), 12)];
    D.sweep = table();
    it_none = zeros(1, 2);  it_exact = zeros(1, 2);  it_worst = zeros(1, 2);
    for f = 1:2
        cs  = make_case(fams{f}, 1, opts);
        n   = cs.n;
        U   = pencil_subspace(cs.A, cs.M, kk, opts);
        Vex = orth_trunc(cs.C' * U);
        Wc  = randn(n, kk);
        Wc  = orth_trunc(Wc - Vex * (Vex' * Wc));      % an orthogonal complement
        b   = ones(n, 1);

        [~, ~, ~, itn] = two_level_solve_local(cs.A, b, 1e-8, 3000, cs, [], tau);
        rowsB = zeros(numel(thB), 5);
        for t = 1:numel(thB)
            Vt = orth_trunc(Vex * cos(thB(t)) + Wc * sin(thB(t)));
            [~, ~, ~, it] = two_level_solve_local(cs.A, b, 1e-8, 3000, cs, Vt, tau);
            rowsB(t, :) = [f, thB(t), gap(Vt, Vex), it, itn];
        end
        it_none(f)  = itn;
        it_exact(f) = rowsB(1, 4);
        it_worst(f) = max(rowsB(:, 4));

        tb = array2table(rowsB, 'VariableNames', ...
             {'family_idx', 'theta', 'gap_to_truth', 'iterations', 'iterations_none'});
        tb.family = repmat(string(fams{f}), height(tb), 1);
        tb.kind   = repmat(string(cs.defl_kind), height(tb), 1);
        D.sweep   = [D.sweep; tb];

        worse = any(rowsB(:, 4) > itn);
        i_w   = find(rowsB(:, 4) > itn, 1);
        mono  = all(diff(rowsB(:, 4)) >= -2);          % non-decreasing up to noise
        % Obs 4.3 is stated for the indefinite family and is MEASURED to be
        % specific to it: with the SPD coarse correction the worst angle merely
        % gives back the no-coarse-space count, never exceeds it.  The two
        % families therefore get opposite assertions, and both fail loudly if the
        % other behaviour ever appears.  This is direct evidence for the
        % explanation README section 4 offers for its own unresolved gap -- that
        % the excess lives in the redistribution of the INTERIOR of an indefinite
        % spectrum, a structure the SPD case does not have.
        if strcmpi(cs.defl_kind, 'indef')
            V = [V, vrec(sprintf('exp6B/%s', fams{f}), ...
                         'Obs 4.3 an inexact coarse space can be worse than none', ...
                         'its(no coarse space) | first theta that loses | its there', ...
                         sprintf('%d | %.2e | %d', itn, nanget(thB, i_w), ...
                                 nanget(rowsB(:,4), i_w)), ...
                         'a losing theta exists', worse)]; %#ok<AGROW>
        else
            V = [V, vrec(sprintf('exp6B/%s', fams{f}), ...
                         'Obs 4.3 does NOT occur for the SPD coarse correction', ...
                         'its(no coarse space) | worst its over theta', ...
                         sprintf('%d | %d', itn, max(rowsB(:, 4))), ...
                         'worst <= none', ~worse)]; %#ok<AGROW>
        end
        V = [V, vrec(sprintf('exp6B/%s', fams{f}), ...
                     'cost is monotone in the angle to the truth', ...
                     'its non-decreasing in theta (tol 2)', mono, 'true', mono)]; %#ok<AGROW>
        i_h = find(rowsB(:, 4) > 1.5 * rowsB(1, 4), 1);
        V = [V, vrec(sprintf('exp6B/%s', fams{f}), ...
                     'degradation with angle is gradual, not a cliff', ...
                     'exact its | first theta with 1.5x | no-coarse-space its', ...
                     sprintf('%d | %.2e | %d', rowsB(1,4), nanget(thB, i_h), itn), ...
                     'the failure mode is a saturated angle, not a small one', NaN)]; %#ok<AGROW>
        fprintf(['[exp6B/%s] (%s, %s) no coarse space %d its; exact %d its; ' ...
                 'worst %d its\n'], fams{f}, cs.defl_kind, ...
                solver_name(cs.defl_kind), itn, rowsB(1,4), max(rowsB(:,4)));
    end
    disp('[exp6B] deflation quality vs coarse-space angle');
    disp(D.sweep);

    %% ---- figure ----------------------------------------------------------
    fh = figure('Visible', 'off');
    tl = tiledlayout(fh, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile(tl);
    loglog(th(2:end), minmu(2:end, 1), '-',  'LineWidth', 2, ...
           'DisplayName', 'indef:  \surd\tau V(V''A^2V)^{-1/2}V''');  hold on;
    loglog(th(2:end), minmu(2:end, 2), '-',  'LineWidth', 2, ...
           'DisplayName', 'spd:  \tau V(V''AV)^{-1}V''');
    yline(abs(l1), 'k--', 'LineWidth', 1.5, 'DisplayName', '|\lambda_1|, no deflation');
    xline(thr(1),  'r:',  'LineWidth', 1.5, 'DisplayName', '\theta = |\lambda_1/\lambda_2|');
    xline(thr(2),  'm:',  'LineWidth', 1.5, 'DisplayName', '\theta = \surd(\lambda_1/\lambda_2)');
    xlabel('coarse-space angle \theta  (rad)');
    ylabel('smallest |eigenvalue| of G A_{hat}');
    title(sprintf('2\\times2 model: \\lambda = (%.2g, %.2g), \\tau = %.2g', l1, l2, tau));
    legend('Location', 'southwest');  hold off;

    nexttile(tl);
    loglog(rs, Hs(:, 1), '-o', 'LineWidth', 2, ...
           'DisplayName', sprintf('indef, slope %.2f', sl(1)));  hold on;
    loglog(rs, Hs(:, 2), '-o', 'LineWidth', 2, ...
           'DisplayName', sprintf('spd, slope %.2f', sl(2)));
    loglog(rs, rs,       'k--', 'LineWidth', 1, 'DisplayName', '\lambda_1/\lambda_2');
    loglog(rs, sqrt(rs), 'k:',  'LineWidth', 1, 'DisplayName', '\surd(\lambda_1/\lambda_2)');
    xlabel('\lambda_1/\lambda_2');  ylabel('half-loss angle  (rad)');
    title('the usable angle, over five decades');
    legend('Location', 'northwest');  hold off;

    for f = 1:2
        nexttile(tl);
        sub = D.sweep(D.sweep.family_idx == f, :);
        semilogx(max(sub.theta, 1e-6), sub.iterations, '-o', 'LineWidth', 2, ...
                 'DisplayName', sprintf('two-level (%s)', solver_name(sub.kind(1))));
        hold on;
        yline(it_none(f), 'k--', 'LineWidth', 1.5, 'DisplayName', 'no coarse space');
        xlabel('coarse-space angle \theta  (rad)');  ylabel('iterations');
        title(sprintf('%s (%s): exact %d, none %d', fams{f}, sub.kind(1), ...
                      it_exact(f), it_none(f)));
        legend('Location', 'northwest');  hold off;
    end

    save_figure(fh, 'cost_vs_angle');
    writetable(D.model,   fullfile(p.outDir, 'exp6_model.csv'));
    writetable(D.scaling, fullfile(p.outDir, 'exp6_scaling.csv'));
    writetable(D.sweep,   fullfile(p.outDir, 'exp6_sweep.csv'));
end

%==========================================================================
function mu = model_roots(l1, l2, tau, theta, kind)
%MODEL_ROOTS  The two eigenvalues of G*Ahat for the 2x2 model of Thm 4.1.
    c = cos(theta);  s = sin(theta);
    switch kind
        case 'indef'                        % E built on Ahat^2, half power
            E  = l1^2*c^2 + l2^2*s^2;
            be = sqrt(tau / E);
            al = l1*c^2 + l2*s^2;           % alpha = v'*Ahat*v, distinct from E
            tr = l1 + l2 + (be - 1)*al;
        case 'spd'                          % E built on Ahat, first power
            E  = l1*c^2 + l2*s^2;
            be = tau / E;
            tr = l1 + l2 + tau - E;         % alpha = E here, so (be-1)*al = tau - E
        otherwise
            error('exp6:kind', 'unknown kind "%s"', kind);
    end
    mu = roots([1, -tr, be*l1*l2]);
end

function s = solver_name(kind)
    if strcmpi(kind, 'spd'), s = 'PCG'; else, s = 'MINRES'; end
end

function v = nanget(x, i)
    if isempty(i), v = NaN; else, v = x(i); end
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
