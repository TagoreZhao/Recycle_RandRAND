function Mapply = amg_precond(A, opts)
% AMG_PRECOND  Ruge-Stuben AMG preconditioner as a function handle.
%
%   Mapply = amg_precond(A)
%   Mapply = amg_precond(A, opts)
%
%   Hierarchy (coarsening + smoothers) is built ONCE at this call. The
%   returned handle runs exactly one V-cycle from a zero initial guess
%   per column of its argument, so Mapply is a fixed linear operator —
%   suitable for pcg(A, b, tol, maxit, Mapply).
%
%   The returned handle accepts either a column vector r or a multi-column
%   matrix R; columns are V-cycled independently so it works as a drop-in
%   operator for subspace_iter, deflation_P_apply, etc.
%
%   opts (optional struct; defaults shown):
%     .level                  (3)
%     .relax_it               (2)
%     .relax_para             (1)
%     .post_smoothing         (1)
%     .pc_type                (1)    smoother selection:
%                                      1 = Jacobi            (symmetric — safe for PCG)
%                                      2 = Gauss-Seidel      (not symmetric — PCG may stall)
%                                      3 = ILU               (broken: luinc removed in R2013a+)
%                                      4 = line Jacobi       (tridiagonal; symmetric — safe for PCG)
%                                      5 = line Gauss-Seidel (not symmetric — PCG may stall)
%     .connection_threshold   (0.25)

  import src.solver.AMG_Wu.*

  if nargin < 2 || isempty(opts), opts = struct; end
  if ~isfield(opts,'level'),                opts.level = 3;                end
  if ~isfield(opts,'relax_it'),             opts.relax_it = 2;             end
  if ~isfield(opts,'relax_para'),           opts.relax_para = 1;           end
  if ~isfield(opts,'post_smoothing'),       opts.post_smoothing = 1;       end
  if ~isfield(opts,'pc_type'),              opts.pc_type = 1;              end
  if ~isfield(opts,'connection_threshold'), opts.connection_threshold = 0.25; end

  % ---- Setup (runs once) ----
  [A_stack, P_stack, m_cindx, l_cindx, lpindx] = ...
      amg_corsening(A, opts.level, opts.connection_threshold);
  MPC = multi_pcond(A_stack, lpindx, m_cindx, l_cindx, ...
                    opts.level, opts.relax_para, opts.pc_type);

  % ---- Capture scalars into the closure ----
  n              = size(A, 1);
  max_level      = opts.level - 1;
  relax_it       = opts.relax_it;
  relax_para     = opts.relax_para;
  post_smoothing = opts.post_smoothing;

  % ---- Apply: one V-cycle per column, loops internally so the handle is
  % matrix-capable (needed by subspace_iter, etc.).
  Mapply = @(R) apply_cols(R, A_stack, MPC, P_stack, m_cindx, lpindx, ...
                           relax_it, relax_para, post_smoothing, max_level, n);
end

function Y = apply_cols(R, A_stack, MPC, P_stack, m_cindx, lpindx, ...
                        relax_it, relax_para, post_smoothing, max_level, n)
  import src.solver.AMG_Wu.*
  k = size(R, 2);
  Y = zeros(n, k);
  for j = 1:k
    rj = R(:, j);
    if issparse(rj), rj = full(rj); end
    [Y(:, j), ~, ~, ~, ~] = MG(A_stack, MPC, zeros(n,1), rj, ...
                               relax_it, relax_para, post_smoothing, ...
                               1e-30, 0, max_level, max_level, 1, ...
                               P_stack, m_cindx, lpindx);
  end
end
