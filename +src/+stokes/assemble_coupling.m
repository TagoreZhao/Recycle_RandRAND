function [C, gvec, nC] = assemble_coupling(TR, N, X, Vpts)
%ASSEMBLE_COUPLING  Distributed-Lagrange-multiplier coupling for an immersed
% rigid solid (the deal.II step-70 mechanism, simplified).
%
%   [C, GVEC, NC] = ASSEMBLE_COUPLING(TR, N, X, VPTS)
%
%   Enforces, weakly at a set of moving Lagrange points X_k on the solid,
%   that the fluid velocity equals the prescribed rigid-body velocity:
%
%       u_h(X_k) = Vpts_k          (one constraint per velocity component)
%
%   Each point contributes two rows to C (one for ux, one for uy); each row
%   interpolates the host-triangle P1 velocity DOFs with barycentric weights,
%   so C has ~3 nonzeros per row and MOVES as X_k(t) moves.  This is the only
%   block of the KKT system that changes from step to step.
%
%   Inputs:
%     TR    - triangulation(msh.t, msh.p), built once and reused per step
%     N     - velocity nodes per component (msh.N); x-DOF of node j is j,
%             y-DOF is N+j
%     X     - K x 2 Lagrange-point coordinates at the current time
%     Vpts  - K x 2 prescribed rigid-body velocity at those points
%
%   Outputs:
%     C     - nC x 2N sparse coupling matrix (rows = [ux constraints; uy])
%     gvec  - nC x 1 right-hand side ( [Vpts_x ; Vpts_y] over valid points )
%     nC    - number of constraint rows (= 2 * number of in-domain points)
%
%   Points that fall outside the fluid mesh (pointLocation -> NaN) are
%   dropped so C never references a missing triangle.

    ti = pointLocation(TR, X);          % K x 1 host triangle, NaN if outside
    valid = ~isnan(ti);

    Xv   = X(valid, :);
    tiv  = ti(valid);
    Vv   = Vpts(valid, :);
    Kv   = size(Xv, 1);

    if Kv == 0
        C    = sparse(0, 2*N);
        gvec = zeros(0, 1);
        nC   = 0;
        return;
    end

    tri_nodes = TR.ConnectivityList(tiv, :);          % Kv x 3 global node ids
    w         = cartesianToBarycentric(TR, tiv, Xv);  % Kv x 3 weights

    rowsK = repmat((1:Kv)', 1, 3);                    % Kv x 3 constraint rows

    Cx = sparse(rowsK(:), tri_nodes(:),     w(:), Kv, 2*N);   % ux rows
    Cy = sparse(rowsK(:), tri_nodes(:) + N, w(:), Kv, 2*N);   % uy rows

    C    = [Cx; Cy];
    gvec = [Vv(:,1); Vv(:,2)];
    nC   = 2 * Kv;
end
