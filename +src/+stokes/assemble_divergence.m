function B = assemble_divergence(msh)
%ASSEMBLE_DIVERGENCE  P1-P1 discrete divergence operator for 2D Stokes.
%   B = ASSEMBLE_DIVERGENCE(MSH)
%
%   Builds the (N x 2N) sparse matrix B with
%
%       B(k, (j,c)) = \int_Omega psi_k  d(phi_j)/d(x_c)  dOmega
%
%   for P1 pressure test functions psi_k and P1 velocity basis phi_j, where
%   the velocity DOFs are ordered [ux ; uy] (x-component of node j is column
%   j, y-component is column N+j).  This is the genuinely new vector-FEM
%   piece for the immersed-Stokes benchmark; everything else reuses the
%   scalar P1 assembly already in +src/+discretization.
%
%   Continuity equation:  B * [ux; uy] = 0  (weakly divergence-free).
%   The momentum block uses B' (the pressure gradient), so the global Stokes
%   saddle [A B'; B -eps*L] is symmetric by construction.
%
%   Derivation (P1 on a triangle, element e, area A_e):
%     d(phi_j)/dx = b_j/(2 A_e),  d(phi_j)/dy = c_j/(2 A_e)   (constant on e)
%     \int_e psi_k = A_e/3
%   so the element contribution is  b_j/6  (x) and  c_j/6  (y), independent of
%   A_e.  Here, matching tri_stiff_loc.m,
%     b = [y2-y3 ; y3-y1 ; y1-y2]   (x-derivative coefficients)
%     c = [x3-x2 ; x1-x3 ; x2-x1]   (y-derivative coefficients)
%
%   Input:
%     msh - mesh struct from assemble_fem_struct (uses .p, .t, .N, .M)
%   Output:
%     B   - N x 2N sparse divergence matrix

    p = msh.p;  t = msh.t;  N = msh.N;

    x1 = p(t(:,1),1); y1 = p(t(:,1),2);
    x2 = p(t(:,2),1); y2 = p(t(:,2),2);
    x3 = p(t(:,3),1); y3 = p(t(:,3),2);

    % Per-element gradient coefficients (M x 3), same convention as tri_stiff_loc.
    b = [y2 - y3, y3 - y1, y1 - y2];   % x-derivative coefficients
    c = [x3 - x2, x1 - x3, x2 - x1];   % y-derivative coefficients

    % Local 3x3 -> 9 expansion: k = pressure test node, j = velocity node.
    Iloc = [1 1 1 2 2 2 3 3 3];   % k
    Jloc = [1 2 3 1 2 3 1 2 3];   % j

    rows  = t(:, Iloc);                 % M x 9 pressure rows (node k)
    cols  = t(:, Jloc);                 % M x 9 velocity nodes (node j)
    valx  = b(:, Jloc) / 6;             % M x 9 x-component contributions
    valy  = c(:, Jloc) / 6;             % M x 9 y-component contributions

    I = [rows(:);        rows(:)       ];
    J = [cols(:);        cols(:) + N   ];   % x-DOFs then y-DOFs
    V = [valx(:);        valy(:)       ];

    B = sparse(I, J, V, N, 2*N);
end
