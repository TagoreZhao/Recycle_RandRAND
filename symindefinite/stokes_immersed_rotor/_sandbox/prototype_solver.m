% PROTOTYPE_SOLVER  Dev-time scratch for the Stokes-immersed-rotor benchmark.
%
% Self-contained backward-Euler time loop that assembles the symmetric
% indefinite KKT system each step and solves it with backslash only (no
% preconditioner) — the ground-truth established before MINRES was added in
% the durable solver +src/+stokes/solve_stokes_immersed.m.
%
% Kept for reference / debugging; the production path is run_benchmark.m.
% Run from this folder:  cd bench/stokes_immersed_rotor/_sandbox; prototype_solver

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(fileparts(thisFileDir)));
addpath(repoRoot);
addpath(fileparts(thisFileDir));     % for define_motion_list
import src.discretization.*
import src.stokes.*

nu = 1.0; dt = 0.02; nsteps = 10; h0 = 0.06;
x1=0; x2=4; y1=0; y2=1; Ly=y2-y1; Uin=1.0;

msh = build_channel_mesh_pde(h0, x1, x2, y1, y2, {'rect_right'});
N = msh.N;
blk = assemble_stokes_blocks(msh);
Avel = blk.M2/dt + nu*blk.A2;
eps_stab = h0^2/(12*nu);
TR = triangulation(msh.t, msh.p);

% inflow BC
left  = find(msh.rect_left);
walls = unique([find(msh.rect_top); find(msh.rect_bottom)]);
bnodes = unique([left; walls]);
yv = msh.p(bnodes,2);
uxv = zeros(numel(bnodes),1);
isleft = ismember(bnodes,left);
uxv(isleft) = Uin*4.*yv(isleft).*(Ly-yv(isleft))/Ly^2;
veldofs = [bnodes; N+bnodes]; velvals = [uxv; zeros(numel(bnodes),1)];
[~, pin] = max(msh.p(:,1));

geo = struct('x1',x1,'x2',x2,'y1',y1,'y2',y2,'xc',(x1+x2)/2,'yc',(y1+y2)/2,'h0',h0,'Tmax',dt*nsteps);
cases = define_motion_list(dt);
motion_fun = cases{1}.factory(geo).motion_fun;   % bar_rotating

u_prev = zeros(2*N,1);
for n = 1:nsteps
    t = n*dt;
    mot = motion_fun(t);
    [C, gvec, nC] = assemble_coupling(TR, N, mot.X, mot.V);
    K = [ Avel, blk.B', C'; blk.B, -eps_stab*blk.L, sparse(N,nC); C, sparse(nC,N), sparse(nC,nC) ];
    b = [ (blk.M2/dt)*u_prev; zeros(N,1); gvec ];
    [K,b] = apply_dirichlet_sym(K, b, veldofs, velvals);
    [K,b] = apply_dirichlet_sym(K, b, 2*N+pin, 0);
    x = K\b;
    u_prev = x(1:2*N);
    symres = norm(K-K','fro')/norm(K,'fro');
    fprintf('step %2d  nC=%4d  sym=%.1e  ||Ku-b||/||b||=%.1e  ||Cu-g||/||g||=%.1e\n', ...
        n, nC, symres, norm(K*x-b)/norm(b), norm(C*x(1:2*N)-gvec)/max(norm(gvec),eps));
end
fprintf('prototype OK\n');
