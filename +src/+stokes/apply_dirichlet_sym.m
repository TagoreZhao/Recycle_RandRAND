function [K, b] = apply_dirichlet_sym(K, b, dofs, vals)
%APPLY_DIRICHLET_SYM  Symmetric elimination of Dirichlet DOFs.
%   [K, b] = APPLY_DIRICHLET_SYM(K, b, DOFS, VALS)
%
%   Enforces x(DOFS) = VALS on the linear system K x = b while PRESERVING
%   the symmetry of K (essential: the immersed-Stokes KKT system is solved
%   with MINRES, which requires a symmetric operator).
%
%   The standard "lift to RHS then zero row+column" procedure:
%     b      <- b - K(:,DOFS) * VALS     (move known columns to the RHS)
%     b(DOFS) <- VALS
%     K(DOFS,:) <- 0 ;  K(:,DOFS) <- 0 ;  K(DOFS,DOFS) <- I
%
%   Inputs:
%     K    - n x n sparse symmetric matrix
%     b    - n x 1 right-hand side
%     dofs - column/row indices to constrain
%     vals - prescribed values (same length as dofs)

    dofs = dofs(:);
    vals = vals(:);
    nd   = numel(dofs);
    if nd == 0
        return;
    end

    b = b - K(:, dofs) * vals;
    b(dofs) = vals;

    K(dofs, :) = 0;
    K(:, dofs) = 0;
    K(dofs, dofs) = speye(nd);
end
