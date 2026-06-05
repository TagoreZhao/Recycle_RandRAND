function mesh = build_polygon_mesh(h0, pv, neumann)
%BUILD_POLYGON_MESH  Mesh a polygonal domain (distmesh2d Example 3).
%   MESH = BUILD_POLYGON_MESH(H0, PV, NEUMANN)
%
%   Inputs:
%     h0      - mesh size  (default 0.1)
%     pv      - (K+1)x2 closed polygon vertices, first==last
%               (default: distmesh2d example polygon)
%     neumann - cell array of boundary names to treat as Neumann (default {})
%               Valid names: 'polygon_bdry', 'edge1', 'edge2', ..., 'edgeK'
%
%   Output:
%     mesh - standard FEM struct (see assemble_fem_struct)
%            Boundary masks: mesh.polygon_bdry, mesh.edge1, ..., mesh.edgeK

    if nargin < 1 || isempty(h0), h0 = 0.1; end
    if nargin < 2 || isempty(pv)
        pv = [-0.4 -0.5; 0.4 -0.2; 0.4 -0.7; 1.5 -0.4; 0.9 0.1;
               1.6  0.8; 0.5  0.5; 0.2  1.0; 0.1  0.4;-0.7  0.7;-0.4 -0.5];
    end
    if nargin < 3, neumann = {}; end

    import src.discretization.*

    bbox = [min(pv(:,1))-0.1, min(pv(:,2))-0.1;
            max(pv(:,1))+0.1, max(pv(:,2))+0.1];
    [p, t] = distmesh2d(@dpoly, @huniform, h0, bbox, pv, pv);
    N = size(p, 1);

    tol = h0/10;
    nvs = size(pv, 1) - 1;   % number of edges

    % --- Per-edge boundary masks ---
    % Compute distance from each node to each segment
    seg_dist = dsegment(p, pv);  % N x nvs

    bdry_masks = struct();
    edge_masks = false(N, nvs);
    for k = 1:nvs
        edge_masks(:, k) = seg_dist(:, k) < tol;
        bdry_masks.(sprintf('edge%d', k)) = edge_masks(:, k);
    end

    % Overall polygon boundary
    polygon_bdry = any(edge_masks, 2);
    bdry_masks.polygon_bdry = polygon_bdry;

    % --- DOF partitioning ---
    Bdry_mask = polygon_bdry;
    neu = lower(neumann);
    if any(strcmp(neu, 'polygon_bdry'))
        Bdry_mask = false(N, 1);
    else
        for k = 1:nvs
            if any(strcmp(neu, sprintf('edge%d', k)))
                Bdry_mask = Bdry_mask & ~edge_masks(:, k);
            end
        end
    end
    Bdry = find(Bdry_mask);
    IN   = setdiff((1:N).', Bdry);

    mesh = assemble_fem_struct(p, t, Bdry, IN, bdry_masks);
end
