function solvers = define_solver_list()
%DEFINE_SOLVER_LIST  MINRES solver/preconditioner registry for the
% Stokes-immersed-rotor benchmark (simplified deal.II step-70).
%
%   solvers = define_solver_list()
%
%   Returns a cell array of solver structs.  Each per-step KKT system is
%   SYMMETRIC INDEFINITE and is solved with MINRES; a solver entry differs only
%   in the (SPD) preconditioner it applies.  This is the extensibility seam:
%   adding a preconditioner is a single struct appended here — the engine
%   (solve_stokes_immersed) and the driver (run_benchmark, make_paper_summary_table)
%   pick it up automatically for CSV columns, plots and the summary table.
%
%   Solver-struct fields:
%     .key    short id used for CSV column names and output filenames
%             (must be a valid MATLAB field name), e.g. 'minres_unprec'.
%     .label  display name for plot legends/titles.
%     .build  @(pc) -> Papply   preconditioner-apply function handle passed as
%             MINRES's 5th argument, or [] for the unpreconditioned solve.
%             pc is a context struct the engine fills with reusable ingredients:
%               pc.Lc  ichol factor of the (BC-eliminated) velocity block Avel
%               pc.Rp  chol  factor of the pressure mass matrix Dp
%               pc.nu  kinematic viscosity
%               pc.nU  velocity DOF count (2N)
%               pc.nP  pressure DOF count (N)
%               pc.nC  number of coupling rows at the current step (per step)
%
%   Mirrors define_motion_list (the geometry/motion registry).  Per the project
%   convention, this preconditioner registry stays LOCAL to the benchmark (it
%   changes/grows and is not geometry-persistent); the +src engine is kept
%   preconditioner-agnostic.

solvers = {};

solvers{end+1} = struct( ...
    'key',   'minres_unprec', ...
    'label', 'MINRES (unpreconditioned)', ...
    'build', @(pc) []);

solvers{end+1} = struct( ...
    'key',   'block_jacobi', ...
    'label', 'MINRES (block Jacobi)', ...
    'build', @(pc) @(r) block_precond(r, pc.nU, pc.nP, pc.nC, pc.Lc, pc.Rp, pc.nu));

solvers = solvers(:);
end

%==========================================================================
%  Preconditioner-apply helpers
%==========================================================================

function y = block_precond(r, nU, nP, nC, Lc, Rp, nu)
%BLOCK_PRECOND  Apply the SPD block-diagonal ("block Jacobi") preconditioner
% P^{-1} r for the Stokes KKT system.
%   Pu   ~ Avel                          (applied via ichol factor Lc)
%   Pp   ~ (1/nu) * pressure mass        (applied via chol factor Rp) -> nu*M^{-1}
%   Plam = I
    ru = r(1:nU);
    rp = r(nU + (1:nP));
    rl = r(nU + nP + (1:nC));

    yu = Lc' \ (Lc \ ru);
    yp = nu * (Rp \ (Rp' \ rp));
    yl = rl;

    y = [yu; yp; yl];
end
