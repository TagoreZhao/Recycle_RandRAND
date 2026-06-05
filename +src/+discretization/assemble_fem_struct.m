function mesh = assemble_fem_struct(p, t, Bdry, IN, bdry_masks)
%ASSEMBLE_FEM_STRUCT  Build the standard FEM mesh struct from raw mesh data.
%   MESH = ASSEMBLE_FEM_STRUCT(P, T, BDRY, IN, BDRY_MASKS)
%
%   Inputs:
%     p          - Nx2  node coordinates (from distmesh2d)
%     t          - Mx3  triangle connectivity (from distmesh2d)
%     Bdry       - column vector of Dirichlet boundary node indices
%     IN         - column vector of interior node indices
%     bdry_masks - struct whose fields are logical Nx1 masks copied to output
%
%   Output:
%     mesh - struct with all fields expected by SPDE experiment scripts:
%       .p, .t, .N, .M           mesh geometry
%       .Bdry, .IN, .numIN, .numB DOF partitioning
%       .D, .D_II, .D_IB         mass matrix and blocks
%       .Kunit_II, .Kunit_IB     unit-kappa stiffness blocks
%       .cent                     element centroids (Mx2)
%       .Itrip, .Jtrip, .Vunit   full stiffness triplets
%       .idxII, .idxIB            triplet index masks
%       .I_II, .J_II              mapped triplet indices for II block
%       .I_IB, .J_IB              mapped triplet indices for IB block
%       + all fields of bdry_masks

    import src.discretization.*

    N = size(p,1);
    M = size(t,1);
    numIN = numel(IN);
    numB  = numel(Bdry);

    % --- Precompute local index expansion (3x3 -> 9) ---
    Iloc = [1 1 1 2 2 2 3 3 3]';
    Jloc = [1 2 3 1 2 3 1 2 3]';

    % --- Triplets for mass (D) and unit-kappa stiffness (Kunit) ---
    Mi = zeros(9*M,1); Mj = zeros(9*M,1); Mv = zeros(9*M,1);
    Itrip = zeros(9*M,1); Jtrip = zeros(9*M,1); Vunit = zeros(9*M,1);

    for e = 1:M
        nod = t(e,:);
        xy  = p(nod,:);
        Mloc = tri_mass_loc(xy);     % 3x3
        Kloc = tri_stiff_loc(xy);    % 3x3 (unit kappa)

        idx = (e-1)*9 + (1:9);
        % mass triplets
        Mi(idx) = nod(Iloc);
        Mj(idx) = nod(Jloc);
        Mv(idx) = Mloc(:);
        % stiffness triplets
        Itrip(idx) = nod(Iloc);
        Jtrip(idx) = nod(Jloc);
        Vunit(idx) = Kloc(:);
    end

    % Assemble mass once
    D = sparse(Mi, Mj, Mv, N, N);

    % --- Element centroids (vectorized) ---
    cent = (p(t(:,1),:) + p(t(:,2),:) + p(t(:,3),:)) / 3;

    % --- Maps & masks for II / IB extraction from triplets ---
    INmask = false(N,1); INmask(IN) = true;
    Bmask  = false(N,1); Bmask(Bdry) = true;

    mapIN = zeros(N,1); mapIN(IN)   = 1:numIN;
    mapB  = zeros(N,1); mapB(Bdry)  = 1:numB;

    maskII = INmask(Itrip) & INmask(Jtrip);
    maskIB = INmask(Itrip) & Bmask(Jtrip);

    idxII = find(maskII);
    idxIB = find(maskIB);

    % Triplet-mapped indices
    I_II = mapIN(Itrip(idxII));  J_II = mapIN(Jtrip(idxII));
    I_IB = mapIN(Itrip(idxIB));  J_IB = mapB (Jtrip(idxIB));

    % Unit-kappa blocks
    Kunit_II = sparse(I_II, J_II, Vunit(idxII), numIN, numIN);
    Kunit_IB = sparse(I_IB, J_IB, Vunit(idxIB), numIN, numB);

    % --- Pack mesh struct ---
    mesh.p   = p;    mesh.t   = t;    mesh.N = N;   mesh.M = M;
    mesh.Bdry = Bdry; mesh.IN = IN;   mesh.numIN = numIN; mesh.numB = numB;

    mesh.D   = D;            mesh.D_II = D(IN,IN);   mesh.D_IB = D(IN,Bdry);
    mesh.Kunit_II = Kunit_II; mesh.Kunit_IB = Kunit_IB;
    mesh.cent = cent;

    % full triplets (for time-varying kappa scaling)
    mesh.Itrip = Itrip; mesh.Jtrip = Jtrip; mesh.Vunit = Vunit;

    % indices to pick II / IB entries from scaled triplets
    mesh.idxII = idxII; mesh.idxIB = idxIB;

    % TRIPLET indices for II / IB
    mesh.I_II = I_II; mesh.J_II = J_II;
    mesh.I_IB = I_IB; mesh.J_IB = J_IB;

    % --- Copy boundary masks ---
    fnames = fieldnames(bdry_masks);
    for k = 1:numel(fnames)
        mesh.(fnames{k}) = bdry_masks.(fnames{k});
    end
end
