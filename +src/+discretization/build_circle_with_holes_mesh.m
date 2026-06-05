function mesh = build_circle_with_holes_mesh(h0, p1x,p1y,R1, p2x,p2y,R2, p3x,p3y,R3, Rout, neumann)
%BUILD_CIRCLE_WITH_HOLES_MESH  Mesh a disk with three circular holes (DistMesh).
%   MESH = BUILD_CIRCLE_WITH_HOLES_MESH(H0, P1X,P1Y,R1, P2X,P2Y,R2, P3X,P3Y,R3, ROUT, NEUMANN)
%
%   Inputs:
%     h0      - mesh size
%     p1x,p1y,R1 - centre and radius of hole 1
%     p2x,p2y,R2 - centre and radius of hole 2
%     p3x,p3y,R3 - centre and radius of hole 3
%     Rout    - radius of outer circle
%     neumann - cell array of boundary names to treat as Neumann (default {})
%               Valid names: 'B1', 'B2', 'B3', 'Bout'
%
%   Output:
%     mesh - standard FEM struct (see assemble_fem_struct)
%            Boundary masks: mesh.B1, mesh.B2, mesh.B3, mesh.Bout

    if nargin < 11, neumann = {}; end

    % --- Geometry & mesh (DistMesh) ---
    import src.discretization.*   % imports functions in +src/+discretization
    fd = @(p) ddiff( ddiff( ddiff(dcircle(p,0,0,Rout),dcircle(p,p1x,p1y,R1)), ...
                             dcircle(p,p2x,p2y,R2)), dcircle(p,p3x,p3y,R3));
    [p,t] = distmesh2d(fd,@huniform,h0,[-2,-2;2,2],[]);
    N = size(p,1);

    % --- Boundary & interior sets ---
    B1   = (sum((p-[p1x,p1y]).^2,2) <= R1^2 + h0/100);
    B2   = (sum((p-[p2x,p2y]).^2,2) <= R2^2 + h0/100);
    B3   = (sum((p-[p3x,p3y]).^2,2) <= R3^2 + h0/100);
    Bout = (sum(p.^2,2)            >= Rout^2 - h0/100);

    % --- Exclude Neumann segments from Dirichlet boundary ---
    Bdry_mask = B1 | B2 | B3 | Bout;
    seg_map = struct('B1', B1, 'B2', B2, 'B3', B3, 'Bout', Bout);
    for i = 1:numel(neumann)
        if isfield(seg_map, neumann{i})
            Bdry_mask = Bdry_mask & ~seg_map.(neumann{i});
        end
    end
    Bdry = find(Bdry_mask);
    IN   = setdiff((1:N).', Bdry);

    % --- Delegate to shared assembly ---
    bdry_masks.B1 = B1; bdry_masks.B2 = B2; bdry_masks.B3 = B3; bdry_masks.Bout = Bout;
    mesh = assemble_fem_struct(p, t, Bdry, IN, bdry_masks);
end
