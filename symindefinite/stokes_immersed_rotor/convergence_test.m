% CONVERGENCE_TEST  Verification for the Stokes-immersed-rotor benchmark.
%
% This benchmark deliberately departs from the suite's SPD contract: the
% per-step KKT matrix is SYMMETRIC INDEFINITE (the user asked for exactly
% that, a MATLAB step-70).  So the SPD sanity check is replaced by a symmetry
% + indefiniteness check, and the order verification is done with the method
% of manufactured solutions (MMS) on the UNCONSTRAINED unsteady Stokes solver:
%
%   Part A  Spatial order  (steady MMS, dt -> inf): velocity L2 order ~ 2.
%   Part B  Temporal order (transient MMS, fine h): backward-Euler order ~ 1.
%   Part C  Immersed constraint residual ||C u - g|| / ||g||  (should be tiny).
%   Part D  Symmetry + indefiniteness of the KKT (both eigenvalue signs).
%   Part E  Per-step coupling change for the moving "bar_rotating" stress case
%           + sparsity diagnostic.
%
% Exact field (divergence-free, on the unit square), my sign convention for
% the symmetric saddle [A B'; B -eps L] (so the momentum strong form is
% u_t - nu*Lap(u) - grad(p) = f):
%   ux =  g(t)*pi*sin(pi x)cos(pi y)
%   uy = -g(t)*pi*cos(pi x)sin(pi y)
%   p  =  g(t)*cos(pi x)cos(pi y)

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
addpath(repoRoot);
import src.discretization.*
import src.stokes.*

outDir = fullfile(thisFileDir, 'convergence_out');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ---- Exact solution handles ------------------------------------------------
pifac = pi;
ux = @(x,y,g)  g .* pifac .* sin(pifac*x) .* cos(pifac*y);
uy = @(x,y,g) -g .* pifac .* cos(pifac*x) .* sin(pifac*y);
pe = @(x,y,g)  g .* cos(pifac*x) .* cos(pifac*y);
% derivatives used to build the forcing
utx = @(x,y,gp)  gp .* pifac .* sin(pifac*x) .* cos(pifac*y);
uty = @(x,y,gp) -gp .* pifac .* cos(pifac*x) .* sin(pifac*y);
dpx = @(x,y,g) -g .* pifac .* sin(pifac*x) .* cos(pifac*y);   % dp/dx
dpy = @(x,y,g) -g .* pifac .* cos(pifac*x) .* sin(pifac*y);   % dp/dy
% Lap(ux) = -2 pi^2 ux, Lap(uy) = -2 pi^2 uy
fx = @(x,y,g,gp,nu) utx(x,y,gp) - nu.*(-2*pifac^2).*ux(x,y,g) - dpx(x,y,g);
fy = @(x,y,g,gp,nu) uty(x,y,gp) - nu.*(-2*pifac^2).*uy(x,y,g) - dpy(x,y,g);

nu = 1.0;

%% ===================== Part A: spatial order (steady MMS) =================
fprintf('\n===== Part A: spatial convergence (steady MMS, velocity L2) =====\n');
h0_list = [0.10 0.07 0.05 0.035];
errU_space = zeros(numel(h0_list), 1);
errP_space = zeros(numel(h0_list), 1);
dofs_space = zeros(numel(h0_list), 1);
gA = 1.0; gpA = 0.0;                   % steady: g = const, g' = 0
dtA = 1e8;                             % dt -> inf  => solves the steady FEM system
for k = 1:numel(h0_list)
    h0 = h0_list(k);
    msh = build_channel_mesh_pde(h0, 0, 1, 0, 1, {});   % unit square, all-Dirichlet
    [uh, ph] = mms_step(msh, nu, dtA, gA, gpA, gA, gpA, ux, uy, pe, fx, fy, h0);
    N = msh.N;
    ue = [ux(msh.p(:,1),msh.p(:,2),gA); uy(msh.p(:,1),msh.p(:,2),gA)];
    pex = pe(msh.p(:,1),msh.p(:,2),gA);
    Dp = msh.D;
    eU = uh - ue;
    errU_space(k) = sqrt(abs(eU(1:N)'*(Dp*eU(1:N)) + eU(N+1:2*N)'*(Dp*eU(N+1:2*N))));
    eP = ph - pex;
    errP_space(k) = sqrt(abs(eP'*(Dp*eP)));
    dofs_space(k) = 3*N;
    fprintf('  h0=%.3f  dofs=%6d  ||eU||_L2=%.3e  ||eP||_L2=%.3e\n', ...
        h0, 3*N, errU_space(k), errP_space(k));
end
ordU_space = diff(log(errU_space)) ./ diff(log(h0_list(:)));
ordP_space = diff(log(errP_space)) ./ diff(log(h0_list(:)));
fprintf('  observed velocity spatial orders: %s\n', num2str(ordU_space', '%.2f '));
fprintf('  observed pressure spatial orders: %s\n', num2str(ordP_space', '%.2f '));

%% ===================== Part B: temporal order (transient MMS) =============
% Measure backward-Euler order against a SAME-MESH, fine-dt reference so the
% (fixed) P1 spatial error cancels exactly and the pure temporal order is
% visible.  Comparing against the exact field instead would bury the temporal
% signal under the spatial-error floor once dt is small.  Every dt (and dt_ref)
% divides Tmax so each run lands exactly on t = Tmax.
fprintf('\n===== Part B: temporal convergence (backward Euler, velocity L2) =====\n');
h0_fixed = 0.04;
Tmax_B   = 0.4;
dt_list  = [0.10 0.05 0.025 0.0125];
dt_ref   = dt_list(end) / 10;          % 1.25e-3, divides Tmax_B
gfun  = @(t) cos(t);
gpfun = @(t) -sin(t);
mshB = build_channel_mesh_pde(h0_fixed, 0, 1, 0, 1, {});
NB = mshB.N; DpB = mshB.D;
uref = mms_transient(mshB, nu, dt_ref, round(Tmax_B/dt_ref), gfun, gpfun, ux, uy, pe, fx, fy, h0_fixed);
errU_time = zeros(numel(dt_list), 1);
for k = 1:numel(dt_list)
    dt = dt_list(k);
    nsteps = round(Tmax_B / dt);
    uh = mms_transient(mshB, nu, dt, nsteps, gfun, gpfun, ux, uy, pe, fx, fy, h0_fixed);
    eU = uh - uref;                    % same mesh: spatial error cancels
    errU_time(k) = sqrt(abs(eU(1:NB)'*(DpB*eU(1:NB)) + eU(NB+1:2*NB)'*(DpB*eU(NB+1:2*NB))));
    fprintf('  dt=%.4f  ||eU(T)-uref||_L2=%.3e\n', dt, errU_time(k));
end
ordU_time = diff(log(errU_time)) ./ diff(log(dt_list(:)));
fprintf('  observed velocity temporal orders: %s\n', num2str(ordU_time', '%.2f '));

%% ===================== Part C/D/E: KKT diagnostics ========================
fprintf('\n===== Part C/D/E: KKT symmetry / indefiniteness / coupling =====\n');
addpath(thisFileDir);                  % for define_motion_list
h0d  = 0.06;
mshD = build_channel_mesh_pde(h0d, 0, 4, 0, 1, {'rect_right'});
N = mshD.N;
blk = assemble_stokes_blocks(mshD);
dt = 0.05;
Avel = blk.M2/dt + nu*blk.A2;
eps_stab = h0d^2 / (12*nu);
TR = triangulation(mshD.t, mshD.p);

geo = struct('x1',0,'x2',4,'y1',0,'y2',1,'xc',2,'yc',0.5,'h0',h0d,'Tmax',1.0);
cases = define_motion_list(dt);
% pick the bar_rotating stress case
sidx = find(cellfun(@(c) strcmp(c.name,'bar_rotating'), cases), 1);
mcase = cases{sidx}.factory(geo);
motion_fun = mcase.motion_fun;

% --- Sparsity of fluid block (Part E) ---
mean_nnz = nnz(Avel) / size(Avel,1);
fprintf('  fluid velocity-block mean nnz/row = %.2f\n', mean_nnz);

% --- Per-step coupling change + constraint residual over a short sweep ---
nprobe = 20;
tprobe = (1:nprobe) * dt;
coup_change = nan(nprobe,1);
C_prev = [];
for n = 1:nprobe
    mot = motion_fun(tprobe(n));
    [C, ~, ~] = assemble_coupling(TR, N, mot.X, mot.V);
    if ~isempty(C_prev) && size(C,1)==size(C_prev,1) && nnz(C_prev)>0
        coup_change(n) = norm(C - C_prev,'fro') / norm(C_prev,'fro');
    end
    C_prev = C;
end
med_change = median(coup_change(~isnan(coup_change)));
fprintf('  bar_rotating median per-step coupling change = %.3f (want >= 0.02)\n', med_change);

% --- Full KKT symmetry + indefiniteness on a coarse mesh (Part D) ---
mshS = build_channel_mesh_pde(0.12, 0, 4, 0, 1, {'rect_right'});
Ns = mshS.N;
blkS = assemble_stokes_blocks(mshS);
AvelS = blkS.M2/dt + nu*blkS.A2;
epsS = 0.12^2/(12*nu);
TRs = triangulation(mshS.t, mshS.p);
mot = motion_fun(0.3);
[Cs, gs, nCs] = assemble_coupling(TRs, Ns, mot.X, mot.V);
Zc = sparse(nCs, nCs);
Ks = [ AvelS, blkS.B', Cs'; ...
       blkS.B, -epsS*blkS.L, sparse(Ns,nCs); ...
       Cs, sparse(nCs,Ns), Zc ];
% velocity Dirichlet (left+top+bottom) + pressure pin to make it nonsingular
bm = mshS.rect_left | mshS.rect_top | mshS.rect_bottom;
bnodes = find(bm);
veldofs = [bnodes; Ns+bnodes];
[~, pin] = max(mshS.p(:,1));
ddofs = [veldofs; 2*Ns+pin];
Ks(ddofs,:) = 0; Ks(:,ddofs) = 0; Ks(ddofs,ddofs) = speye(numel(ddofs));
symres = norm(Ks - Ks','fro')/norm(Ks,'fro');
ev = eig(full(Ks));
lam_min = min(real(ev)); lam_max = max(real(ev));
fprintf('  coarse KKT: size=%d  sym_res=%.2e  lambda_min=%.3e  lambda_max=%.3e\n', ...
    size(Ks,1), symres, lam_min, lam_max);

% --- Constraint reproduction via a genuine solve on the coarse system ---
b = zeros(size(Ks,1),1);
b(end-nCs+1:end) = gs;                 % lambda rows carry the rigid-body velocity g
x = Ks \ b;
u = x(1:2*Ns);
constraint_residual = norm(Cs*u - gs)/max(norm(gs),eps);
fprintf('  immersed constraint residual ||C u - g||/||g|| = %.2e\n', constraint_residual);

%% ===================== Figure pack =======================================
fh = figure('Visible','off','Position',[100 100 1200 700]);
subplot(2,3,1);
loglog(h0_list, errU_space,'o-','LineWidth',1.5); hold on;
loglog(h0_list, errU_space(1)*(h0_list/h0_list(1)).^2,'k--');
xlabel('h0'); ylabel('||e_u||_{L2}'); title('Spatial (velocity), slope 2 ref'); grid on;
subplot(2,3,2);
loglog(dt_list, errU_time,'s-','LineWidth',1.5); hold on;
loglog(dt_list, errU_time(1)*(dt_list/dt_list(1)).^1,'k--');
xlabel('dt'); ylabel('||e_u(T)||_{L2}'); title('Temporal (velocity), slope 1 ref'); grid on;
subplot(2,3,3);
loglog(h0_list, errP_space,'o-','LineWidth',1.5);
xlabel('h0'); ylabel('||e_p||_{L2}'); title('Spatial (pressure)'); grid on;
subplot(2,3,4);
plot(tprobe, coup_change,'.-','LineWidth',1.2);
yline(0.02,'r--'); xlabel('t'); ylabel('||\DeltaC||_F/||C||_F');
title(sprintf('Coupling change (median %.3f)', med_change)); grid on;
subplot(2,3,5);
plot(real(ev),'.'); yline(0,'r-'); xlabel('index'); ylabel('eig(K)');
title(sprintf('KKT spectrum (\\lambda_{min}=%.1e)', lam_min)); grid on;
subplot(2,3,6); axis off;
txt = {
    sprintf('velocity spatial order (mean last 3): %.2f', mean(ordU_space(end-2:end)))
    sprintf('velocity temporal order (mean last 3): %.2f', mean(ordU_time(end-2:end)))
    sprintf('KKT symmetry residual: %.1e', symres)
    sprintf('KKT indefinite: lambda_min=%.1e < 0 < lambda_max=%.1e', lam_min, lam_max)
    sprintf('constraint residual: %.1e', constraint_residual)
    sprintf('median coupling change: %.3f', med_change)
    sprintf('fluid mean nnz/row: %.2f', mean_nnz)
    };
text(0.02,0.95,txt,'VerticalAlignment','top','FontSize',10,'Interpreter','none');
saveas(fh, fullfile(outDir,'convergence_summary.png'));
close(fh);
fprintf('\nFigure pack written to %s\n', fullfile(outDir,'convergence_summary.png'));

%% ===================== Pass/fail asserts ==================================
fprintf('\n===== PASS/FAIL =====\n');
ok = true;
assert_print = @(cond,msg) fprintf('  [%s] %s\n', ternary(cond,'PASS','FAIL'), msg);

c1 = all(diff(errU_space) < 0) && mean(ordU_space(end-2:end)) > 1.7 && mean(ordU_space(end-2:end)) < 2.3;
assert_print(c1, sprintf('velocity spatial order ~2 (got %.2f)', mean(ordU_space(end-2:end)))); ok = ok && c1;
c2 = all(diff(errU_time) < 0) && mean(ordU_time(end-2:end)) > 0.8 && mean(ordU_time(end-2:end)) < 1.3;
assert_print(c2, sprintf('velocity temporal order ~1 (got %.2f)', mean(ordU_time(end-2:end)))); ok = ok && c2;
c3 = symres < 1e-12;
assert_print(c3, sprintf('KKT symmetric (sym_res=%.1e)', symres)); ok = ok && c3;
c4 = (lam_min < 0) && (lam_max > 0);
assert_print(c4, 'KKT indefinite (eigenvalues of both signs)'); ok = ok && c4;
c5 = constraint_residual < 1e-8;
assert_print(c5, sprintf('immersed constraint satisfied (res=%.1e)', constraint_residual)); ok = ok && c5;
c6 = med_change >= 0.02;
assert_print(c6, sprintf('stress-case coupling moves (median=%.3f >= 0.02)', med_change)); ok = ok && c6;

if ok
    fprintf('\nALL CHECKS PASSED.\n');
else
    error('convergence_test:fail', 'One or more convergence/diagnostic checks failed.');
end

%==========================================================================
%  Local functions
%==========================================================================
function [uh, ph] = mms_step(msh, nu, dt, gprev, gpprev, gnew, gpnew, ux, uy, pe, fx, fy, h0) %#ok<INUSL>
% One backward-Euler MMS step from the exact previous state (used for the
% steady spatial test, where dt -> inf collapses to the steady FEM solve).
    import src.stokes.*
    N = msh.N; p = msh.p;
    blk = assemble_stokes_blocks(msh);
    eps_stab = h0^2/(12*nu);
    Avel = blk.M2/dt + nu*blk.A2;
    K = [Avel, blk.B'; blk.B, -eps_stab*blk.L];
    uprev = [ux(p(:,1),p(:,2),gprev); uy(p(:,1),p(:,2),gprev)];
    fnod  = [fx(p(:,1),p(:,2),gnew,gpnew,nu); fy(p(:,1),p(:,2),gnew,gpnew,nu)];
    pexn  = pe(p(:,1),p(:,2),gnew);
    rhsU  = (blk.M2/dt)*uprev + blk.M2*fnod;
    rhsP  = -eps_stab*(blk.L*pexn);                 % stabilization consistency correction
    b = [rhsU; rhsP];
    [uh, ph] = solve_with_bc(K, b, msh, N, ux, uy, pe, gnew);
end

function uh = mms_transient(msh, nu, dt, nsteps, gfun, gpfun, ux, uy, pe, fx, fy, h0)
% Backward-Euler MMS time loop, IC = exact at t=0, returns velocity at Tmax.
    import src.stokes.*
    N = msh.N; p = msh.p;
    blk = assemble_stokes_blocks(msh);
    eps_stab = h0^2/(12*nu);
    Avel = blk.M2/dt + nu*blk.A2;
    K0 = [Avel, blk.B'; blk.B, -eps_stab*blk.L];
    uprev = [ux(p(:,1),p(:,2),gfun(0)); uy(p(:,1),p(:,2),gfun(0))];
    uh = uprev;
    for n = 1:nsteps
        tn = n*dt;
        gn = gfun(tn); gpn = gpfun(tn);
        fnod = [fx(p(:,1),p(:,2),gn,gpn,nu); fy(p(:,1),p(:,2),gn,gpn,nu)];
        pexn = pe(p(:,1),p(:,2),gn);
        rhsU = (blk.M2/dt)*uprev + blk.M2*fnod;
        rhsP = -eps_stab*(blk.L*pexn);
        b = [rhsU; rhsP];
        [uh, ~] = solve_with_bc(K0, b, msh, N, ux, uy, pe, gn);
        uprev = uh;
    end
end

function [uh, ph] = solve_with_bc(K, b, msh, N, ux, uy, pe, g)
    import src.stokes.*
    p = msh.p;
    bnodes = msh.Bdry;
    veldofs = [bnodes; N+bnodes];
    velvals = [ux(p(bnodes,1),p(bnodes,2),g); uy(p(bnodes,1),p(bnodes,2),g)];
    [K, b] = apply_dirichlet_sym(K, b, veldofs, velvals);
    [~, pin] = min(p(:,1)+p(:,2));
    [K, b] = apply_dirichlet_sym(K, b, 2*N+pin, pe(p(pin,1),p(pin,2),g));
    x = K \ b;
    uh = x(1:2*N);
    ph = x(2*N+(1:N));
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
