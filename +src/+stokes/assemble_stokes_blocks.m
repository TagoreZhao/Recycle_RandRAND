function blk = assemble_stokes_blocks(msh)
%ASSEMBLE_STOKES_BLOCKS  Time-independent fluid blocks for P1-P1 Stokes.
%   BLK = ASSEMBLE_STOKES_BLOCKS(MSH)
%
%   Assembles, ONCE, every block of the Stokes saddle-point operator that
%   does not change as the immersed solid moves.  Only the coupling block
%   C(t_n) (see assemble_coupling.m) is rebuilt per time step.
%
%   Velocity DOFs are ordered [ux ; uy] (2N total); pressure is P1 (N DOFs).
%
%   Returned struct fields:
%     blk.N    - velocity nodes per component (= msh.N)
%     blk.Np   - pressure DOFs (= msh.N, equal-order P1-P1)
%     blk.Dp   - N x N   scalar consistent mass (pressure mass / vel mass block)
%     blk.K1   - N x N   unit-kappa P1 stiffness (scalar Laplacian)
%     blk.M2   - 2N x 2N block-diagonal velocity mass
%     blk.A2   - 2N x 2N block-diagonal vector Laplacian (unit viscosity)
%     blk.B    - N x 2N  discrete divergence (continuity operator)
%     blk.L    - N x N   pressure-stabilization Laplacian (= K1)
%
%   The global per-step operator is
%       K = [ M2/dt + nu*A2 ,  B' ,  C' ;
%             B             , -eps*L,  0 ;
%             C             ,  0   ,  0 ]
%   which is SYMMETRIC (transpose-paired off-diagonals, symmetric diagonal
%   blocks) and INDEFINITE (zero (lam,lam) block plus negative -eps*L).

    import src.stokes.*

    N = msh.N;

    Dp = msh.D;                                            % scalar mass
    K1 = sparse(msh.Itrip, msh.Jtrip, msh.Vunit, N, N);   % unit P1 stiffness
    K1 = (K1 + K1') / 2;                                   % enforce exact symmetry

    Z = sparse(N, N);

    blk.N  = N;
    blk.Np = N;
    blk.Dp = Dp;
    blk.K1 = K1;
    blk.M2 = [Dp, Z; Z, Dp];
    blk.A2 = [K1, Z; Z, K1];
    blk.B  = assemble_divergence(msh);
    blk.L  = K1;
end
