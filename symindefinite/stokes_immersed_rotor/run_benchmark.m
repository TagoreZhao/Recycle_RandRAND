% RUN_BENCHMARK  Stokes-immersed-rotor benchmark (simplified deal.II step-70).
%
% Backward-Euler unsteady Stokes in a 2D channel with a moving immersed rigid
% solid enforced by distributed Lagrange multipliers.  Each time step solves a
% SYMMETRIC INDEFINITE saddle-point (KKT) system whose coupling block C(t_n)
% changes because the solid moves.  Solver comparison per step:
%   backslash (ground truth) vs MINRES (unpreconditioned) vs MINRES (SPD
%   block-diagonal preconditioner).
%
% NOTE: this benchmark intentionally departs from the suite's SPD contract.
% The PCG/ICHOL/AMG/deflation zoo (solve_deflate_M_P, RAND_EIGS) does NOT
% apply to an indefinite system, so the iteration columns are
% minres_unprec_its / minres_blk_its rather than ichol/amg/chol.

%% ===================== 1. Setup / params ==================================
thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
addpath(repoRoot);
addpath(thisFileDir);
import src.discretization.*
import src.stokes.*
rng(1);

params.dt          = 0.02;
params.Tstep       = 61;        % Tmax = 1.2
params.SOLVER_TOL  = 1e-8;
params.SOLVER_MAXIT = 4000;
params.h0          = 0.05;

% Channel geometry
x1 = 0; x2 = 4; y1 = 0; y2 = 1;
Lyc = y2 - y1;
Uin = 1.0;                      % peak inflow velocity

%% ===================== 2. Mesh ============================================
fprintf('[stokes_immersed_rotor] building channel mesh (h0=%.3f) ...\n', params.h0);
msh = build_channel_mesh_pde(params.h0, x1, x2, y1, y2, {'rect_right'});
N = msh.N;
fprintf('  nodes: %d  velocity DOFs: %d  pressure DOFs: %d\n', N, 2*N, N);

%% ===================== 3. BCs + case list + SMOKE_TEST ====================
% Velocity Dirichlet: parabolic inflow on the left, no-slip on top/bottom,
% natural (do-nothing) outflow on the right.
left = find(msh.rect_left);
walls = unique([find(msh.rect_top); find(msh.rect_bottom)]);
bnodes = unique([left; walls]);
yv = msh.p(bnodes, 2);
uxv = zeros(numel(bnodes), 1);
isleft = ismember(bnodes, left);
uxv(isleft) = Uin * 4 .* yv(isleft) .* (Lyc - yv(isleft)) / Lyc^2;  % parabola, 0 at walls
veldofs = [bnodes; N + bnodes];
velvals = [uxv; zeros(numel(bnodes), 1)];
velbc_fun = @(t) struct('dofs', veldofs, 'vals', velvals);   % steady inflow

[~, pin_node] = max(msh.p(:, 1));   % pin pressure at the outflow corner

geo = struct('x1', x1, 'x2', x2, 'y1', y1, 'y2', y2, ...
             'xc', (x1+x2)/2, 'yc', (y1+y2)/2, ...
             'h0', params.h0, 'Tmax', params.dt * params.Tstep);

all_cases = define_motion_list(params.dt);
all_names = cellfun(@(c) c.name, all_cases, 'UniformOutput', false);
case_names = {'bar_rotating', 'disk_translating', 'disk_static'};

if evalin('base', 'exist(''SMOKE_TEST'',''var'') && logical(SMOKE_TEST)')
    fprintf('[SMOKE_TEST] Overriding params for fast end-to-end check.\n');
    params.Tstep = 3;
    case_names = case_names(1);   % single (stress) case
end

results_root = fullfile(thisFileDir, 'test_tentative_approx');
if ~exist(results_root, 'dir'), mkdir(results_root); end

%% ===================== 4. Loop over cases =================================
num_cases = numel(case_names);
all_stats = cell(num_cases, 1);
for k = 1:num_cases
    cname = case_names{k};
    idx   = find(strcmp(all_names, cname), 1);
    mcase = all_cases{idx}.factory(geo);

    cfg = struct();
    cfg.mesh       = msh;
    cfg.nu         = mcase.nu;
    cfg.h0         = params.h0;
    cfg.velbc_fun  = velbc_fun;
    cfg.motion_fun = mcase.motion_fun;
    cfg.pin_node   = pin_node;
    cfg.pin_val    = 0;
    cfg.case_name  = cname;
    cfg.geometry   = 'stokes_immersed_rotor';

    run_dir = fullfile(results_root, cname);
    if ~exist(run_dir, 'dir'), mkdir(run_dir); end

    fprintf('\n========== Case %d/%d: %s ==========\n', k, num_cases, cname);
    st = solve_stokes_immersed(cfg, params, run_dir);
    st.mean_nnz_per_row = nnz(assemble_stokes_blocks(msh).A2) / (2*N);
    st.case_name = cname;
    all_stats{k} = st;

    % --- coefficient movie for the stress case ---
    if mcase.is_stress
        write_coefficient_movie(msh, mcase.motion_fun, params, run_dir);
    end
end

%% ===================== 5. CSV + plots =====================================
case_col={}; geom_col={}; ts_col=[]; un_its=[]; blk_its=[];
un_flag=[]; blk_flag=[]; relres_col=[]; flag_col=[]; diffF_col=[];
bs_relres=[]; blk_err=[]; constr=[]; nC_col=[];
for k = 1:num_cases
    st = all_stats{k};
    ns = numel(st.minres_blk_its);
    tcol = (1:ns)';
    case_col = [case_col; repmat({st.case_name}, ns, 1)];          %#ok<AGROW>
    geom_col = [geom_col; repmat({'stokes_immersed_rotor'}, ns, 1)]; %#ok<AGROW>
    ts_col   = [ts_col;   tcol];                                    %#ok<AGROW>
    un_its   = [un_its;   st.minres_unprec_its(:)];                 %#ok<AGROW>
    blk_its  = [blk_its;  st.minres_blk_its(:)];                    %#ok<AGROW>
    un_flag  = [un_flag;  st.minres_unprec_flag(:)];               %#ok<AGROW>
    blk_flag = [blk_flag; st.minres_blk_flag(:)];                  %#ok<AGROW>
    relres_col = [relres_col; st.minres_blk_relres(:)];           %#ok<AGROW>
    flag_col   = [flag_col;   st.minres_blk_flag(:)];             %#ok<AGROW>
    diffF_col  = [diffF_col;  st.coupling_change(:)];             %#ok<AGROW>
    bs_relres  = [bs_relres;  st.backslash_relres(:)];           %#ok<AGROW>
    blk_err    = [blk_err;    st.minres_blk_err(:)];             %#ok<AGROW>
    constr     = [constr;     st.constraint_res(:)];             %#ok<AGROW>
    nC_col     = [nC_col;     st.nC(:)];                          %#ok<AGROW>
end

T = table(case_col, geom_col, ts_col, un_its, blk_its, un_flag, blk_flag, ...
          relres_col, flag_col, diffF_col, bs_relres, blk_err, constr, nC_col, ...
    'VariableNames', {'case_name','geometry','timestep', ...
        'minres_unprec_its','minres_blk_its','minres_unprec_flag','minres_blk_flag', ...
        'relres','flag','diffF','backslash_relres','minres_blk_err','constraint_res','nC'});
writetable(T, fullfile(results_root, 'all_results.csv'));
fprintf('\nWrote %s\n', fullfile(results_root, 'all_results.csv'));

% --- per-case plots ---
for k = 1:num_cases
    st = all_stats{k};
    ns = numel(st.minres_blk_its);
    tax = (1:ns)' * params.dt;
    run_dir = fullfile(results_root, st.case_name);

    f1 = figure('Visible','off','Position',[100 100 1100 400]);
    subplot(1,3,1);
    safe = @(v) max(v(:),1);
    semilogy(tax, safe(st.minres_unprec_its),'-o','MarkerSize',3,'LineWidth',1.2); hold on;
    semilogy(tax, safe(st.minres_blk_its),'-s','MarkerSize',3,'LineWidth',1.2);
    xlabel('t'); ylabel('MINRES iterations'); grid on;
    legend('unpreconditioned','block precond','Location','best');
    title(sprintf('%s: MINRES iterations', st.case_name),'Interpreter','none');
    subplot(1,3,2);
    plot(tax, st.coupling_change,'.-','LineWidth',1.2);
    xlabel('t'); ylabel('||\DeltaC||_F/||C||_F'); grid on;
    title('per-step coupling change');
    subplot(1,3,3);
    semilogy(tax, max(st.minres_blk_err,1e-16),'-','LineWidth',1.2); hold on;
    semilogy(tax, max(st.constraint_res,1e-16),'--','LineWidth',1.2);
    xlabel('t'); ylabel('relative'); grid on;
    legend('MINRES(blk) vs backslash','constraint ||Cu-g||/||g||','Location','best');
    title('accuracy');
    saveas(f1, fullfile(run_dir, 'summary.png'));
    close(f1);
end

% --- run config ---
cfg_out.params = params;
cfg_out.geometry = 'stokes_immersed_rotor';
cfg_out.case_names = case_names;
save(fullfile(results_root, 'run_config.mat'), 'cfg_out');
jstr = jsonencode(cfg_out);
fid = fopen(fullfile(results_root, 'run_config.json'),'w');
if fid > 0, fwrite(fid, jstr); fclose(fid); end

fprintf('\n[stokes_immersed_rotor] done.\n');

%==========================================================================
%  Local function
%==========================================================================
function write_coefficient_movie(msh, motion_fun, params, run_dir)
    mdir = fullfile(run_dir, 'coefficient_movie');
    if ~exist(mdir, 'dir'), mkdir(mdir); end
    nframes = 8;
    Tmax = params.dt * params.Tstep;
    tt = linspace(params.dt, Tmax, nframes);
    for i = 1:nframes
        mot = motion_fun(tt(i));
        fh = figure('Visible','off','Position',[100 100 800 240]);
        triplot(msh.t, msh.p(:,1), msh.p(:,2), 'Color', [0.85 0.85 0.85]); hold on;
        plot(mot.X(:,1), mot.X(:,2), 'r.', 'MarkerSize', 8);
        axis equal tight; xlabel('x'); ylabel('y');
        title(sprintf('immersed solid, t = %.3f', tt(i)));
        saveas(fh, fullfile(mdir, sprintf('frame_%02d.png', i)));
        close(fh);
    end
end
