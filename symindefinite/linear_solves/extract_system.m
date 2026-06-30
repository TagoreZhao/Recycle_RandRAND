% EXTRACT_SYSTEM  Extract one symmetric-indefinite Stokes KKT pair (A, b) and
% save it for preconditioner experiments in this folder.
%
% Assembles a single per-step KKT system of the immersed-rotor Stokes benchmark
% (the same matrix +src/+stokes/solve_stokes_immersed.m builds each time step)
% at a fixed bar_rotating snapshot, applies the symmetric Dirichlet + pressure-pin
% elimination, and writes A (=K), b and a meta struct to stokes_kkt_system.mat.
%
% The inflow Dirichlet values are lifted into the RHS by apply_dirichlet_sym, so
% b is genuinely nonzero and (A, b) is a meaningful linear solve.
%
% Run this once; then run test_ildl_minres.m.
%
% See also: make_ildl_precond, test_ildl_minres,
%           src.stokes.solve_stokes_immersed, run_spectrum_spy.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
addpath(repoRoot);
addpath(thisFileDir);
addpath(fullfile(fileparts(thisFileDir), 'stokes_immersed_rotor'));  % define_motion_list
import src.discretization.*
import src.stokes.*
rng(1);

% ---- parameters (benchmark-representative; see run_benchmark.m) -----------
h0     = 0.05;          % benchmark mesh resolution (n = 3N + nC ~ 5.8k)
nu     = 1.0;
dt     = 0.02;
Tmax   = 1.2;
t_snap = 0.30;          % fixed bar_rotating snapshot

fprintf('[extract_system] assembling KKT snapshot (h0=%.3g, t=%.3g)...\n', h0, t_snap);
S = assemble_kkt_snapshot(h0, t_snap, nu, dt, Tmax);
A = S.K;                 % symmetric indefinite KKT matrix
b = S.b;                 % nonzero RHS (inflow Dirichlet lifting)

% ---- properties ----------------------------------------------------------
n        = size(A, 1);
sym_res  = norm(A - A', 'fro') / max(norm(A, 'fro'), eps);
ev_small = eigs(A, 1, 'smallestreal', struct('maxit', 500));
ev_large = eigs(A, 1, 'largestreal',  struct('maxit', 500));

fprintf('[extract_system] n=%d  nU=%d  nP=%d  nC=%d  nnz(A)=%d\n', ...
        n, S.nU, S.nP, S.nC, nnz(A));
fprintf('[extract_system] symmetry residual ||A-A''||/||A|| = %.2e\n', sym_res);
fprintf('[extract_system] eig range: lambda_min=%.3e  lambda_max=%.3e  (indefinite: %d)\n', ...
        ev_small, ev_large, ev_small < 0 && ev_large > 0);

% ---- save ----------------------------------------------------------------
meta = struct('n', n, 'nU', S.nU, 'nP', S.nP, 'nC', S.nC, 'N', S.N, ...
              'h0', h0, 't_snap', t_snap, 'nu', nu, 'dt', dt, ...
              'case_name', 'bar_rotating', 'sym_res', sym_res, ...
              'lambda_min', ev_small, 'lambda_max', ev_large);
matFile = fullfile(thisFileDir, 'stokes_kkt_system.mat');
save(matFile, 'A', 'b', 'meta');
fprintf('[extract_system] saved %s\n', matFile);

%==========================================================================
%  Local functions
%==========================================================================
function S = assemble_kkt_snapshot(h0, t_snap, nu, dt, Tmax)
%ASSEMBLE_KKT_SNAPSHOT  Standalone symmetric-indefinite KKT (A=K) and RHS b at
% the bar_rotating snapshot t_snap.  Mirrors solve_stokes_immersed.m assembly
% and run_spectrum_spy.m:assemble_kkt_snapshot, but keeps the lifted RHS b.
    import src.discretization.*
    import src.stokes.*

    x1 = 0; x2 = 4; y1 = 0; y2 = 1; Lyc = y2 - y1; Uin = 1.0;

    msh  = build_channel_mesh_pde(h0, x1, x2, y1, y2, {'rect_right'});
    N    = msh.N;  nU = 2*N;  nP = N;
    blk  = assemble_stokes_blocks(msh);
    Avel = blk.M2/dt + nu*blk.A2;  Avel = (Avel + Avel')/2;   % SPD velocity block
    eps_stab = h0^2 / (12*nu);

    % --- bar_rotating coupling at t_snap ---
    geo = struct('x1',x1,'x2',x2,'y1',y1,'y2',y2, ...
                 'xc',(x1+x2)/2,'yc',(y1+y2)/2,'h0',h0,'Tmax',Tmax);
    cases = define_motion_list(dt);
    sidx  = find(cellfun(@(c) strcmp(c.name,'bar_rotating'), cases), 1);
    mcase = cases{sidx}.factory(geo);
    mot   = mcase.motion_fun(t_snap);
    TR    = triangulation(msh.t, msh.p);
    [C, gvec, nC] = assemble_coupling(TR, N, mot.X, mot.V);

    % --- full symmetric indefinite KKT ---
    Z = @(a,b) sparse(a,b);
    K = [ Avel ,  blk.B'        ,  C'        ; ...
          blk.B, -eps_stab*blk.L,  Z(nP,nC) ; ...
          C    ,  Z(nP,nC)'     ,  Z(nC,nC)  ];

    % --- RHS: rest-state velocity (u_prev = 0) + coupling target gvec ---
    b = [ zeros(nU,1); zeros(nP,1); gvec ];

    % --- velocity Dirichlet (parabolic inflow + no-slip walls), pressure pin ---
    left  = find(msh.rect_left);
    walls = unique([find(msh.rect_top); find(msh.rect_bottom)]);
    bnodes = unique([left; walls]);
    yv  = msh.p(bnodes, 2);
    uxv = zeros(numel(bnodes), 1);
    isl = ismember(bnodes, left);
    uxv(isl) = Uin * 4 .* yv(isl) .* (Lyc - yv(isl)) / Lyc^2;
    veldofs = [bnodes; N + bnodes];
    velvals = [uxv; zeros(numel(bnodes), 1)];
    [K, b] = apply_dirichlet_sym(K, b, veldofs, velvals);
    [~, pin_node] = max(msh.p(:, 1));            % outflow corner
    [K, b] = apply_dirichlet_sym(K, b, nU + pin_node, 0);

    K = (K + K')/2;                              % enforce exact symmetry
    S.K = K;  S.b = b;  S.n = size(K,1);
    S.nU = nU;  S.nP = nP;  S.nC = nC;  S.N = N;  S.h0 = h0;
end
