% VARVISC_CONVERGENCE_TEST  Verification for the variable-viscosity Stokes-
% immersed-rotor benchmark (symindefinite/stokes_varvisc_rotor).
%
% Like the parent (stokes_immersed_rotor) this departs from the suite's
% SPD contract: the per-step KKT matrix is SYMMETRIC INDEFINITE, so the SPD
% sanity check is replaced by symmetry + indefiniteness, and order
% verification uses the method of manufactured solutions on the UNCONSTRAINED
% unsteady variable-viscosity Stokes solver:
%
%   Part A  Spatial order  (steady MMS, dt -> inf, smooth nu(x,y)):
%           velocity L2 order ~ 2.
%   Part B  Temporal order (transient MMS, nu(x,y,t) time-varying, fine h):
%           backward-Euler order ~ 1.  Per-step reassembly of A2(nu_e) and
%           Lp_eps(nu_e) is exercised inside this loop.
%   Part C  Immersed constraint residual ||C u - g|| / ||g||  (tiny).
%   Part D  Symmetry + indefiniteness of the var-nu KKT (both eigen-signs).
%   Part E  Stress-case invariants on the moving 100:1 nu field:
%           contrast >= 50, median fluid-block diffK >= 0.02, blob centroid
%           displacement >= 2*h0 per production step, mean nnz/row <= 12,
%           median coupling change >= 0.02, and the LOW-RANK REFUTATION:
%           the svd of the per-step stiffness difference needs rank
%           r90 = O(N) to capture 90% of its Frobenius mass (the parent's
%           exploitable border update has rank <= 2*nC ~ 40).
%
% MMS detail: the viscous weak form is grad-grad, i.e. the strong operator is
% -div(nu grad u) = -nu*Lap(u) - grad(nu).grad(u), so the forcing includes
% the grad(nu).grad(u) term.  Sign convention inherited from the parent
% (symmetric saddle => discrete p = -physical p):
%     u_t - nu*Lap(u) - grad(nu).grad(u) - grad(p) = f
% Exact fields (divergence-free, unit square):
%   ux =  g(t)*pi*sin(pi x)cos(pi y)
%   uy = -g(t)*pi*cos(pi x)sin(pi y)
%   p  =  g(t)*cos(pi x)cos(pi y)
%   nu =  1 + 0.5*a(t)*sin(pi x)sin(pi y)      (a = 1 steady, cos(t) transient)

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
addpath(repoRoot);
addpath(thisFileDir);                  % varvisc_define_case_list
import src.discretization.*
import src.stokes.*

outDir = fullfile(thisFileDir, 'convergence_out');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ---- Exact solution handles ------------------------------------------------
pf = pi;
ux = @(x,y,g)  g .* pf .* sin(pf*x) .* cos(pf*y);
uy = @(x,y,g) -g .* pf .* cos(pf*x) .* sin(pf*y);
pe = @(x,y,g)  g .* cos(pf*x) .* cos(pf*y);
% time derivatives
utx = @(x,y,gp)  gp .* pf .* sin(pf*x) .* cos(pf*y);
uty = @(x,y,gp) -gp .* pf .* cos(pf*x) .* sin(pf*y);
% pressure gradient (of the DISCRETE pressure; momentum term is -grad p)
dpx = @(x,y,g) -g .* pf .* sin(pf*x) .* cos(pf*y);
dpy = @(x,y,g) -g .* pf .* cos(pf*x) .* sin(pf*y);
% velocity gradients (for the grad(nu).grad(u) term)
duxdx = @(x,y,g)  g .* pf^2 .* cos(pf*x) .* cos(pf*y);
duxdy = @(x,y,g) -g .* pf^2 .* sin(pf*x) .* sin(pf*y);
duydx = @(x,y,g)  g .* pf^2 .* sin(pf*x) .* sin(pf*y);
duydy = @(x,y,g) -g .* pf^2 .* cos(pf*x) .* cos(pf*y);
% smooth MMS viscosity, amplitude factor a (steady: a=1; transient: a=cos t)
nuf  = @(x,y,a) 1 + 0.5 * a .* sin(pf*x) .* sin(pf*y);
nufx = @(x,y,a) 0.5 * a * pf .* cos(pf*x) .* sin(pf*y);
nufy = @(x,y,a) 0.5 * a * pf .* sin(pf*x) .* cos(pf*y);
% forcing: f = u_t - nu*Lap(u) - grad(nu).grad(u) - grad(p);  Lap(u) = -2 pi^2 u
fx = @(x,y,g,gp,a) utx(x,y,gp) + 2*pf^2 .* nuf(x,y,a) .* ux(x,y,g) ...
    - (nufx(x,y,a) .* duxdx(x,y,g) + nufy(x,y,a) .* duxdy(x,y,g)) - dpx(x,y,g);
fy = @(x,y,g,gp,a) uty(x,y,gp) + 2*pf^2 .* nuf(x,y,a) .* uy(x,y,g) ...
    - (nufx(x,y,a) .* duydx(x,y,g) + nufy(x,y,a) .* duydy(x,y,g)) - dpy(x,y,g);

%% ===================== Part A: spatial order (steady MMS) =================
fprintf('\n===== Part A: spatial convergence (steady MMS, variable nu) =====\n');
h0_list = [0.10 0.07 0.05 0.035];
errU_space = zeros(numel(h0_list), 1);
errP_space = zeros(numel(h0_list), 1);
gA = 1.0; gpA = 0.0; aA = 1.0;         % steady: g = 1, nu amplitude = 1
dtA = 1e8;                             % dt -> inf  => steady FEM system
for k = 1:numel(h0_list)
    h0 = h0_list(k);
    msh = build_channel_mesh_pde(h0, 0, 1, 0, 1, {});   % unit square, all-Dirichlet
    [uh, ph] = mms_step(msh, h0, dtA, gA, gpA, aA, ux, uy, pe, fx, fy, nuf);
    N = msh.N;
    ue = [ux(msh.p(:,1),msh.p(:,2),gA); uy(msh.p(:,1),msh.p(:,2),gA)];
    pex = pe(msh.p(:,1),msh.p(:,2),gA);
    Dp = msh.D;
    eU = uh - ue;
    errU_space(k) = sqrt(abs(eU(1:N)'*(Dp*eU(1:N)) + eU(N+1:2*N)'*(Dp*eU(N+1:2*N))));
    eP = ph - pex;
    errP_space(k) = sqrt(abs(eP'*(Dp*eP)));
    fprintf('  h0=%.3f  dofs=%6d  ||eU||_L2=%.3e  ||eP||_L2=%.3e\n', ...
        h0, 3*N, errU_space(k), errP_space(k));
end
ordU_space = diff(log(errU_space)) ./ diff(log(h0_list(:)));
ordP_space = diff(log(errP_space)) ./ diff(log(h0_list(:)));
fprintf('  observed velocity spatial orders: %s\n', num2str(ordU_space', '%.2f '));
fprintf('  observed pressure spatial orders: %s\n', num2str(ordP_space', '%.2f '));

%% ===================== Part B: temporal order (transient MMS) =============
% Same-mesh fine-dt reference so the fixed P1 spatial error cancels and the
% pure backward-Euler order is visible.  nu(x,y,t) is time-varying here, so
% Avel and Lp_eps are REASSEMBLED EVERY STEP — this loop regression-tests the
% production per-step assembly path.
fprintf('\n===== Part B: temporal convergence (transient MMS, nu(x,y,t)) =====\n');
h0_fixed = 0.04;
Tmax_B   = 0.4;
dt_list  = [0.10 0.05 0.025 0.0125];
dt_ref   = dt_list(end) / 10;          % 1.25e-3, divides Tmax_B
gfun  = @(t) cos(t);
gpfun = @(t) -sin(t);
afun  = @(t) cos(t);                   % nu = 1 + 0.5*cos(t)*sin(pi x)sin(pi y) in [0.5, 1.5]
mshB = build_channel_mesh_pde(h0_fixed, 0, 1, 0, 1, {});
NB = mshB.N; DpB = mshB.D;
uref = mms_transient(mshB, h0_fixed, dt_ref, round(Tmax_B/dt_ref), gfun, gpfun, afun, ux, uy, pe, fx, fy, nuf);
errU_time = zeros(numel(dt_list), 1);
for k = 1:numel(dt_list)
    dt = dt_list(k);
    nsteps = round(Tmax_B / dt);
    uh = mms_transient(mshB, h0_fixed, dt, nsteps, gfun, gpfun, afun, ux, uy, pe, fx, fy, nuf);
    eU = uh - uref;                    % same mesh: spatial error cancels
    errU_time(k) = sqrt(abs(eU(1:NB)'*(DpB*eU(1:NB)) + eU(NB+1:2*NB)'*(DpB*eU(NB+1:2*NB))));
    fprintf('  dt=%.4f  ||eU(T)-uref||_L2=%.3e\n', dt, errU_time(k));
end
ordU_time = diff(log(errU_time)) ./ diff(log(dt_list(:)));
fprintf('  observed velocity temporal orders: %s\n', num2str(ordU_time', '%.2f '));

%% ===================== Part C/D/E: KKT diagnostics ========================
fprintf('\n===== Part C/D/E: var-nu KKT symmetry / indefiniteness / motion =====\n');
% Production values the gates are calibrated against
h0_prod = 0.05;  dt_prod = 0.02;  Tstep_prod = 61;
Tmax_prod = dt_prod * (Tstep_prod - 1);

geo = struct('x1',0,'x2',4,'y1',0,'y2',1,'xc',2,'yc',0.5, ...
             'h0',0.10,'Tmax',Tmax_prod);
cases = varvisc_define_case_list(dt_prod);
sidx  = find(cellfun(@(c) strcmp(c.name,'bar_rotating_nu_orbiting'), cases), 1);
mcase = cases{sidx}.factory(geo);

% --- Part E: per-step matrix change / contrast / coupling on coarse channel ---
h0E  = 0.10;
mshE = build_channel_mesh_pde(h0E, 0, 4, 0, 1, {'rect_right'});
NE   = mshE.N;
blkE = assemble_stokes_blocks(mshE);
TRe  = triangulation(mshE.t, mshE.p);
nprobe = 12;
tprobe = (1:nprobe) * dt_prod;
diffK_probe = nan(nprobe,1);
contrast_probe = zeros(nprobe,1);
coup_change = nan(nprobe,1);
F_prev = []; C_prev = [];
mean_nnz = NaN;
for n = 1:nprobe
    nu_e = mcase.nu_fun(mshE.cent(:,1), mshE.cent(:,2), tprobe(n));
    K1nu = coef_stiffness(mshE, nu_e);
    ZN   = sparse(NE, NE);
    Avel = blkE.M2/dt_prod + [K1nu, ZN; ZN, K1nu];
    Lp_eps = coef_stiffness(mshE, h0E^2 ./ (12*nu_e));
    F = [Avel, blkE.B'; blkE.B, -Lp_eps];
    contrast_probe(n) = max(nu_e) / min(nu_e);
    if ~isempty(F_prev)
        diffK_probe(n) = norm(F - F_prev,'fro') / norm(F_prev,'fro');
    end
    F_prev = F;
    if n == 1, mean_nnz = nnz(Avel) / (2*NE); end
    mot = mcase.motion_fun(tprobe(n));
    [C, ~, ~] = assemble_coupling(TRe, NE, mot.X, mot.V);
    if ~isempty(C_prev) && size(C,1)==size(C_prev,1) && nnz(C_prev)>0
        coup_change(n) = norm(C - C_prev,'fro') / norm(C_prev,'fro');
    end
    C_prev = C;
end
med_diffK   = median(diffK_probe(~isnan(diffK_probe)));
min_contrast = min(contrast_probe);
med_coup    = median(coup_change(~isnan(coup_change)));
fprintf('  median fluid-block diffK   = %.4f (want >= 0.02)\n', med_diffK);
fprintf('  min nu contrast            = %.1f  (want >= 50)\n', min_contrast);
fprintf('  median coupling change     = %.4f (want >= 0.02)\n', med_coup);
fprintf('  fluid velocity-block mean nnz/row = %.2f (want <= 12)\n', mean_nnz);

% --- Blob-centroid displacement per production step (analytic path) ---
nrev = 3; omega = 2*pi*nrev / Tmax_prod; R_orb = 0.35;
blob_disp = omega * dt_prod * R_orb;
fprintf('  blob displacement/step     = %.4f (want >= 2*h0_prod = %.3f)\n', ...
    blob_disp, 2*h0_prod);

% --- Low-rank refutation: svd of the per-step stiffness difference ---
% Taken on a mesh that RESOLVES the striation texture (~5.6 cells per
% wavelength at h0=0.07); the h0E=0.10 probe mesh above under-samples it
% and under-reports the rank.
h0_svd  = 0.07;
mshSvd  = build_channel_mesh_pde(h0_svd, 0, 4, 0, 1, {'rect_right'});
Nsvd    = mshSvd.N;
tmid = round(nprobe/2) * dt_prod;
nu_a = mcase.nu_fun(mshSvd.cent(:,1), mshSvd.cent(:,2), tmid);
nu_b = mcase.nu_fun(mshSvd.cent(:,1), mshSvd.cent(:,2), tmid + dt_prod);
dK1  = coef_stiffness(mshSvd, nu_b) - coef_stiffness(mshSvd, nu_a);
sv   = svd(full(dK1));
cums = sqrt(cumsum(sv.^2));
r90  = find(cums >= 0.9 * cums(end), 1);
r99  = find(cums >= 0.99 * cums(end), 1);
parent_rank_bound = 40;                % parent's border update: rank <= 2*nC ~ 40
fprintf('  low-rank refutation: r90 = %d, r99 = %d of N = %d (r90 = %.0f%% of N; parent ~%d)\n', ...
    r90, r99, Nsvd, 100*r90/Nsvd, parent_rank_bound);

% --- Part D: full var-nu KKT symmetry + indefiniteness on a coarse mesh ---
mshS = build_channel_mesh_pde(0.12, 0, 4, 0, 1, {'rect_right'});
Ns = mshS.N;
blkS = assemble_stokes_blocks(mshS);
tD = 0.3;
nu_eS = mcase.nu_fun(mshS.cent(:,1), mshS.cent(:,2), tD);
K1S = coef_stiffness(mshS, nu_eS);
ZS  = sparse(Ns, Ns);
AvelS = blkS.M2/dt_prod + [K1S, ZS; ZS, K1S];
LpS = coef_stiffness(mshS, 0.12^2 ./ (12*nu_eS));
TRs = triangulation(mshS.t, mshS.p);
mot = mcase.motion_fun(tD);
[Cs, gs, nCs] = assemble_coupling(TRs, Ns, mot.X, mot.V);
Ks = [ AvelS, blkS.B', Cs'; ...
       blkS.B, -LpS, sparse(Ns,nCs); ...
       Cs, sparse(nCs,Ns), sparse(nCs,nCs) ];
bm = mshS.rect_left | mshS.rect_top | mshS.rect_bottom;
bnodes = find(bm);
veldofs = [bnodes; Ns+bnodes];
[~, pin] = max(mshS.p(:,1));
ddofs = [veldofs; 2*Ns+pin];
Ks(ddofs,:) = 0; Ks(:,ddofs) = 0; Ks(ddofs,ddofs) = speye(numel(ddofs));
symres = norm(Ks - Ks','fro')/norm(Ks,'fro');
ev = eig(full(Ks));
lam_min = min(real(ev)); lam_max = max(real(ev));
fprintf('  coarse var-nu KKT: size=%d  sym_res=%.2e  lam_min=%.3e  lam_max=%.3e\n', ...
    size(Ks,1), symres, lam_min, lam_max);

% --- Part C: constraint reproduction via a genuine coarse solve ---
b = zeros(size(Ks,1),1);
b(end-nCs+1:end) = gs;
x = Ks \ b;
u = x(1:2*Ns);
constraint_residual = norm(Cs*u - gs)/max(norm(gs),eps);
fprintf('  immersed constraint residual ||C u - g||/||g|| = %.2e\n', constraint_residual);

%% ===================== Figure pack =======================================
fh = figure('Visible','off','Position',[100 100 1400 900]);
subplot(3,3,1);
loglog(h0_list, errU_space,'o-','LineWidth',1.5); hold on;
loglog(h0_list, errU_space(1)*(h0_list/h0_list(1)).^2,'k--');
xlabel('h0'); ylabel('||e_u||_{L2}'); title('Spatial (velocity), slope 2 ref'); grid on;
subplot(3,3,2);
loglog(dt_list, errU_time,'s-','LineWidth',1.5); hold on;
loglog(dt_list, errU_time(1)*(dt_list/dt_list(1)).^1,'k--');
xlabel('dt'); ylabel('||e_u(T)||_{L2}'); title('Temporal (velocity), slope 1 ref'); grid on;
subplot(3,3,3);
loglog(h0_list, errP_space,'o-','LineWidth',1.5);
xlabel('h0'); ylabel('||e_p||_{L2}'); title('Spatial (pressure)'); grid on;
subplot(3,3,4);
plot(tprobe, diffK_probe,'.-','LineWidth',1.2); hold on;
plot(tprobe, coup_change,'.--','LineWidth',1.0);
yline(0.02,'r--'); xlabel('t'); ylabel('rel. change');
legend('fluid block \DeltaF','coupling \DeltaC','Location','best');
title(sprintf('Per-step change (median diffK %.3f)', med_diffK)); grid on;
subplot(3,3,5);
semilogy(sv / sv(1), '-','LineWidth',1.2); hold on;
xline(r90, 'r--'); xline(parent_rank_bound, 'k:');
xlabel('index'); ylabel('\sigma_k/\sigma_1');
title(sprintf('svd(\\DeltaK1\\nu): r90=%d of N=%d', r90, Nsvd)); grid on;
legend('singular values','r90','parent rank ~40','Location','best');
subplot(3,3,6);
plot(real(ev),'.'); yline(0,'r-'); xlabel('index'); ylabel('eig(K)');
title(sprintf('KKT spectrum (\\lambda_{min}=%.1e)', lam_min)); grid on;
subplot(3,3,7);
nu_plot = mcase.nu_fun(mshSvd.cent(:,1), mshSvd.cent(:,2), tmid);
patch('Faces', mshSvd.t, 'Vertices', mshSvd.p, ...
      'FaceVertexCData', log10(nu_plot), 'FaceColor', 'flat', 'EdgeColor', 'none');
mot = mcase.motion_fun(tmid);
hold on; plot(mot.X(:,1), mot.X(:,2), 'r.', 'MarkerSize', 8);
axis equal tight; colorbar; title(sprintf('log_{10}\\nu(x, t=%.2f) + rotor', tmid));
subplot(3,3,8);
plot(tprobe, contrast_probe,'.-','LineWidth',1.2); yline(50,'r--');
xlabel('t'); ylabel('max\nu/min\nu'); title('Viscosity contrast'); grid on;
subplot(3,3,9); axis off;
txt = {
    sprintf('velocity spatial order (mean last 3): %.2f', mean(ordU_space(end-2:end)))
    sprintf('velocity temporal order (mean last 3): %.2f', mean(ordU_time(end-2:end)))
    sprintf('KKT symmetry residual: %.1e', symres)
    sprintf('KKT indefinite: %.1e < 0 < %.1e', lam_min, lam_max)
    sprintf('constraint residual: %.1e', constraint_residual)
    sprintf('median diffK: %.3f   median coupling: %.3f', med_diffK, med_coup)
    sprintf('min contrast: %.0f   blob disp/step: %.3f', min_contrast, blob_disp)
    sprintf('r90 = %d, r99 = %d of N = %d (parent rank ~40)', r90, r99, Nsvd)
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
assert_print(c3, sprintf('var-nu KKT symmetric (sym_res=%.1e)', symres)); ok = ok && c3;
c4 = (lam_min < 0) && (lam_max > 0);
assert_print(c4, 'var-nu KKT indefinite (eigenvalues of both signs)'); ok = ok && c4;
c5 = constraint_residual < 1e-8;
assert_print(c5, sprintf('immersed constraint satisfied (res=%.1e)', constraint_residual)); ok = ok && c5;
c6 = med_coup >= 0.02;
assert_print(c6, sprintf('coupling moves (median=%.3f >= 0.02)', med_coup)); ok = ok && c6;
c7 = med_diffK >= 0.02;
assert_print(c7, sprintf('fluid block moves (median diffK=%.3f >= 0.02)', med_diffK)); ok = ok && c7;
c8 = min_contrast >= 50;
assert_print(c8, sprintf('viscosity contrast (min=%.0f >= 50)', min_contrast)); ok = ok && c8;
c9 = blob_disp >= 2*h0_prod;
assert_print(c9, sprintf('blob moves >= 2*h0/step (%.3f >= %.3f)', blob_disp, 2*h0_prod)); ok = ok && c9;
c10 = mean_nnz <= 12;
assert_print(c10, sprintf('sparsity (mean nnz/row=%.2f <= 12)', mean_nnz)); ok = ok && c10;
% r90 scales O(N) here (dense-in-pattern update) vs O(1) for the parent's
% border update; 0.10*N is a pre-calibration floor — tighten it to the
% measured value minus margin once the sandbox numbers are in.
c11 = (r90 >= 0.10*Nsvd) && (r90 > parent_rank_bound);
assert_print(c11, sprintf('update NOT low-rank (r90=%d >= 0.10N=%.0f and > %d)', ...
    r90, 0.10*Nsvd, parent_rank_bound)); ok = ok && c11;

if ok
    fprintf('\nALL CHECKS PASSED.\n');
else
    error('varvisc_convergence_test:fail', 'One or more convergence/diagnostic checks failed.');
end

%==========================================================================
%  Local functions
%==========================================================================
function Kc = coef_stiffness(msh, coef_e)
% P1 stiffness with an elementwise coefficient via unit-triplet rescale.
    Kc = sparse(msh.Itrip, msh.Jtrip, msh.Vunit .* repelem(coef_e, 9), msh.N, msh.N);
    Kc = (Kc + Kc') / 2;
end

function [uh, ph] = mms_step(msh, h0, dt, g, gp, a, ux, uy, pe, fx, fy, nuf)
% One backward-Euler MMS step from the exact previous state (steady spatial
% test: dt -> inf collapses to the steady variable-nu FEM solve).
    import src.stokes.*
    N = msh.N; p = msh.p;
    blk = assemble_stokes_blocks(msh);
    nu_e = nuf(msh.cent(:,1), msh.cent(:,2), a);
    K1nu = coef_stiffness(msh, nu_e);
    ZN   = sparse(N, N);
    Avel = blk.M2/dt + [K1nu, ZN; ZN, K1nu];
    Lp_eps = coef_stiffness(msh, h0^2 ./ (12*nu_e));
    K = [Avel, blk.B'; blk.B, -Lp_eps];
    uprev = [ux(p(:,1),p(:,2),g); uy(p(:,1),p(:,2),g)];
    fnod  = [fx(p(:,1),p(:,2),g,gp,a); fy(p(:,1),p(:,2),g,gp,a)];
    pexn  = pe(p(:,1),p(:,2),g);
    rhsU  = (blk.M2/dt)*uprev + blk.M2*fnod;
    rhsP  = -(Lp_eps*pexn);               % stabilization consistency correction
    b = [rhsU; rhsP];
    [uh, ph] = solve_with_bc(K, b, msh, N, ux, uy, pe, g);
end

function uh = mms_transient(msh, h0, dt, nsteps, gfun, gpfun, afun, ux, uy, pe, fx, fy, nuf)
% Backward-Euler MMS time loop with TIME-VARYING nu: Avel and Lp_eps are
% reassembled every step (the production per-step assembly path).
    import src.stokes.*
    N = msh.N; p = msh.p;
    blk = assemble_stokes_blocks(msh);
    ZN  = sparse(N, N);
    uprev = [ux(p(:,1),p(:,2),gfun(0)); uy(p(:,1),p(:,2),gfun(0))];
    uh = uprev;
    for n = 1:nsteps
        tn = n*dt;
        gn = gfun(tn); gpn = gpfun(tn); an = afun(tn);
        nu_e = nuf(msh.cent(:,1), msh.cent(:,2), an);
        K1nu = coef_stiffness(msh, nu_e);
        Avel = blk.M2/dt + [K1nu, ZN; ZN, K1nu];
        Lp_eps = coef_stiffness(msh, h0^2 ./ (12*nu_e));
        K = [Avel, blk.B'; blk.B, -Lp_eps];
        fnod = [fx(p(:,1),p(:,2),gn,gpn,an); fy(p(:,1),p(:,2),gn,gpn,an)];
        pexn = pe(p(:,1),p(:,2),gn);
        rhsU = (blk.M2/dt)*uprev + blk.M2*fnod;
        rhsP = -(Lp_eps*pexn);
        b = [rhsU; rhsP];
        [uh, ~] = solve_with_bc(K, b, msh, N, ux, uy, pe, gn);
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
