% PROFILE_COMPONENTS  Per-component cost breakdown for the stokes_immersed_rotor
% benchmark.  Builds one representative KKT system (bar_rotating, step ~12) and
% times each piece the per-step loop pays, separating ONE-TIME costs (deflation
% V builds, decomposition) from PER-STEP costs (ILDL factor, the 5 split MINRES
% solves, the baseline solves).  Mirrors the exact settings of run_benchmark.m.

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
addpath(repoRoot); addpath(thisFileDir);
import src.discretization.*
import src.stokes.*
import src.precond.*
rng(1);

% ---- params identical to run_benchmark.m --------------------------------
dt = 0.02; h0 = 0.05; tol = 1e-8; maxit = 4000;
DEFL = struct('sm_eig',500,'lg_eig',0,'q',2,'tau',0.5, ...
              'ildl_mode','nofill','droptol',1e-3,'cheb_degree',4);
PROFILE_STEP = 12;                      % match the printed step

% ---- mesh + constant blocks ---------------------------------------------
x1=0; x2=4; y1=0; y2=1; Lyc=y2-y1; Uin=1.0;
msh = build_channel_mesh_pde(h0, x1, x2, y1, y2, {'rect_right'});
N = msh.N; nU = 2*N; nP = N;
blk  = assemble_stokes_blocks(msh);
mlist = define_motion_list(dt);
mnames = cellfun(@(c) c.name, mlist, 'UniformOutput', false);
geo = struct('x1',x1,'x2',x2,'y1',y1,'y2',y2,'xc',(x1+x2)/2,'yc',(y1+y2)/2, ...
             'h0',h0,'Tmax',dt*61);
mcase = mlist{find(strcmp(mnames,'bar_rotating'),1)}.factory(geo);
nu = mcase.nu;
Avel = blk.M2/dt + nu*blk.A2;  Avel = (Avel+Avel')/2;
Bdiv = blk.B; Lp = blk.L; Dp = blk.Dp;
eps_stab = h0^2/(12*nu);
TR = triangulation(msh.t, msh.p);
[~,pin_node] = max(msh.p(:,1));

% BCs (steady parabolic inflow)
left = find(msh.rect_left);
walls = unique([find(msh.rect_top); find(msh.rect_bottom)]);
bnodes = unique([left; walls]); yv = msh.p(bnodes,2);
uxv = zeros(numel(bnodes),1); isleft = ismember(bnodes,left);
uxv(isleft) = Uin*4.*yv(isleft).*(Lyc-yv(isleft))/Lyc^2;
veldofs = [bnodes; N+bnodes]; velvals = [uxv; zeros(numel(bnodes),1)];

% context for the KKT builder
ctx = struct('TR',TR,'N',N,'nU',nU,'nP',nP,'mcase',mcase,'Avel',Avel, ...
    'Bdiv',Bdiv,'Lp',Lp,'eps_stab',eps_stab,'blk',blk,'dt',dt, ...
    'veldofs',veldofs,'velvals',velvals,'pin_node',pin_node);

% ---- advance ground-truth state to PROFILE_STEP-1 (cheap backslash) -----
u_prev = zeros(nU,1);
for n = 1:PROFILE_STEP-1
    [K,b,~] = build_kkt(n*dt, u_prev, ctx);
    x = K\b; u_prev = x(1:nU);
end
[K,b,nC] = build_kkt(PROFILE_STEP*dt, u_prev, ctx);
ntot = size(K,1); mit = min(maxit, ntot);
fprintf('\n=== KKT at step %d: n=%d (nU=%d nP=%d nC=%d) ===\n', ...
        PROFILE_STEP, ntot, nU, nP, nC);

R = struct('name',{},'sec',{},'kind',{},'note',{});

% ---- PER-STEP: ILDL factor (full ldl + mask) ----------------------------
ildl_opts = struct('mode','nofill');
tP = timeit_n(@() make_ildl_precond(K, ildl_opts), 3);
P  = make_ildl_precond(K, ildl_opts);
R = addrow(R,'ILDL build (make_ildl_precond)', tP, 'per-step', sprintf('fill=%.1fx nnzL=%d', P.fill_ratio, P.nnzL));

% isolate the raw ldl() inside it
As = (K+K')/2;
Assp = sparse(As);
tldl = timeit_n(@() ldl_call(Assp), 3);
R = addrow(R,'   - raw ldl() factorization', tldl, 'per-step', 'subset of ILDL build');

% ---- ONE-TIME: decomposition(K) for gaussian/sjlt -----------------------
tdec = timeit_n(@() decomposition(K), 2);
dA = decomposition(K);
R = addrow(R,'decomposition(K)  [gaussian/sjlt]', tdec, 'one-time', 'refresh=Inf');

% ---- ONE-TIME: deflation V builds, per method ---------------------------
for m = {'exact','polynomial','gaussian','sjlt'}
    meth = m{1}; o = DEFL; o.method = meth;
    if any(strcmp(meth,{'gaussian','sjlt'}))
        f = @() build_deflation_V(K, P, o, dA);
    else
        f = @() build_deflation_V(K, P, o, []);
    end
    tv = timeit_n(f, 1);
    R = addrow(R, sprintf('build_deflation_V (%s)',meth), tv, 'one-time', sprintf('sm_eig=%d', DEFL.sm_eig));
end

% Build the exact V for the split-solve timing
oe = DEFL; oe.method = 'exact';
Vexact = build_deflation_V(K, P, oe, []);

% ---- PER-STEP: the 5 split MINRES solves --------------------------------
tnone = timeit_n(@() two_level_split_solve(K,b,tol,mit,P,[],DEFL.tau), 1);
[~,~,~,itnone] = two_level_split_solve(K,b,tol,mit,P,[],DEFL.tau);
R = addrow(R,'split solve: ildl_nofill (V=[])', tnone, 'per-step', sprintf('%d its', itnone));

tdef = timeit_n(@() two_level_split_solve(K,b,tol,mit,P,Vexact,DEFL.tau), 1);
[~,~,~,itdef] = two_level_split_solve(K,b,tol,mit,P,Vexact,DEFL.tau);
R = addrow(R,'split solve: two_level (V=exact)', tdef, 'per-step', sprintf('%d its (x4 methods)', itdef));

% ---- PER-STEP: baselines ------------------------------------------------
tbs = timeit_n(@() K\b, 2);
R = addrow(R,'backslash K\\b (ground truth)', tbs, 'per-step', '');

tun = timeit_n(@() minres(K,b,tol,mit), 1);
[~,~,~,itun] = minres(K,b,tol,mit);
R = addrow(R,'minres_unprec', tun, 'per-step', sprintf('%d its', itun));

Au_bc = make_au_bc(Avel, veldofs);
Lc = ichol(Au_bc, struct('type','nofill'));
Rp = chol((Dp+Dp')/2);
bj = @(r) blkprec(r,nU,nP,nC,Lc,Rp,nu);
tbj = timeit_n(@() minres(K,b,tol,mit,bj), 1);
[~,~,~,itbj] = minres(K,b,tol,mit,bj);
R = addrow(R,'minres block_jacobi', tbj, 'per-step', sprintf('%d its', itbj));

% ================= report ================================================
[~,ord] = sort([R.sec],'descend');
fprintf('\n%-42s %10s  %-9s  %s\n','COMPONENT','sec','kind','note');
fprintf('%s\n', repmat('-',1,92));
for i = ord
    fprintf('%-42s %10.3f  %-9s  %s\n', R(i).name, R(i).sec, R(i).kind, R(i).note);
end

% ---- project full-run cost (3 cases x 60 steps) -------------------------
isper = strcmp({R.kind},'per-step');
% exclude the 'raw ldl' subset row (already inside ILDL build); count the split
% solve x4 (the 4 two-level methods reuse it; ildl_nofill is its own row)
perstep = 0;
for i=1:numel(R)
    if ~isper(i), continue; end
    if startsWith(strtrim(R(i).name),'- raw'), continue; end
    mult = 1;
    if contains(R(i).name,'two_level (V=exact)'), mult = 4; end
    perstep = perstep + R(i).sec*mult;
end
onetime = sum([R(strcmp({R.kind},'one-time')).sec]);
nsteps = 60; ncases = 3;
fprintf('\n--- projected full run (%d cases x %d steps) ---\n', ncases, nsteps);
fprintf('per-step total : %.2f s/step  x %d x %d = %.1f min\n', ...
        perstep, nsteps, ncases, perstep*nsteps*ncases/60);
fprintf('one-time total : %.2f s/case  x %d        = %.1f min\n', ...
        onetime, ncases, onetime*ncases/60);
fprintf('GRAND TOTAL (approx)            = %.1f min\n', ...
        (perstep*nsteps*ncases + onetime*ncases)/60);

% ===================== local functions ===================================
function t = timeit_n(f, reps)
    ts = zeros(reps,1);
    for i=1:reps, tt=tic; f(); ts(i)=toc(tt); end
    t = median(ts);
end

function R = addrow(R, nm, s, kind, note)
    R(end+1) = struct('name',nm,'sec',s,'kind',kind,'note',note);
end

function ldl_call(A)
    [~,~,~] = ldl(A,'vector');
end

function [K,b,nC] = build_kkt(tcur, u_prev, ctx)
    import src.stokes.*
    mot = ctx.mcase.motion_fun(tcur);
    [C,gvec,nC] = assemble_coupling(ctx.TR, ctx.N, mot.X, mot.V);
    Zf = @(a,b) sparse(a,b);
    K = [ ctx.Avel, ctx.Bdiv', C'; ...
          ctx.Bdiv, -ctx.eps_stab*ctx.Lp, Zf(ctx.nP,nC); ...
          C, Zf(nC,ctx.nP), Zf(nC,nC) ];
    rhsU = (ctx.blk.M2/ctx.dt) * u_prev;
    b = [rhsU; zeros(ctx.nP,1); gvec];
    [K,b] = apply_dirichlet_sym(K, b, ctx.veldofs, ctx.velvals);
    [K,b] = apply_dirichlet_sym(K, b, ctx.nU + ctx.pin_node, 0);
end

function Au = make_au_bc(Avel, dofs)
    Au = Avel; Au(dofs,:) = 0; Au(:,dofs) = 0;
    Au(dofs,dofs) = speye(numel(dofs)); Au = (Au+Au')/2;
end

function y = blkprec(r,nU,nP,nC,Lc,Rp,nu)
    ru=r(1:nU); rp=r(nU+(1:nP)); rl=r(nU+nP+(1:nC));
    y = [Lc'\(Lc\ru); nu*(Rp\(Rp'\rp)); rl];
end
