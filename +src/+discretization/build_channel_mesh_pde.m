function mesh = build_channel_mesh_pde(h0, x1, x2, y1, y2, neumann)
%BUILD_CHANNEL_MESH_PDE  Mesh a rectangular channel (no hole) with MATLAB's
% PDE Toolbox.  Companion to build_rect_with_hole_mesh_pde for the immersed-
% Stokes benchmark, where the obstacle is NOT cut out of the mesh — it is an
% immersed solid enforced by Lagrange multipliers on a fictitious domain.
%
%   MESH = BUILD_CHANNEL_MESH_PDE(H0, X1, X2, Y1, Y2, NEUMANN)
%
%   Produces the standard struct from assemble_fem_struct, with boundary
%   masks:
%     .rect_left  (inflow)   .rect_right (outflow)
%     .rect_bottom (wall)    .rect_top   (wall)
%
%   Inputs:
%     h0      - target mesh edge length (Hmax)
%     x1,x2   - x extent (default 0..4)
%     y1,y2   - y extent (default 0..1)
%     neumann - cell array of boundary names treated as Neumann / natural
%               (removed from the Dirichlet set). Default {'rect_right'} so the
%               outflow is a natural (do-nothing) boundary.

    if nargin < 1 || isempty(h0), h0 = 0.05; end
    if nargin < 2 || isempty(x1), x1 = 0;   end
    if nargin < 3 || isempty(x2), x2 = 4;   end
    if nargin < 4 || isempty(y1), y1 = 0;   end
    if nargin < 5 || isempty(y2), y2 = 1;   end
    if nargin < 6, neumann = {'rect_right'}; end

    import src.discretization.*

    % --- Geometry via decsg (single rectangle) ---
    R = [3; 4; x1; x2; x2; x1; y1; y1; y2; y2];
    gd = R;
    ns = char('R')';
    sf = 'R';
    g = decsg(gd, sf, ns);

    % --- Generate triangular mesh ---
    model = createpde();
    geometryFromEdges(model, g);
    pdeMesh = generateMesh(model, 'Hmax', h0, 'GeometricOrder', 'linear');

    p = pdeMesh.Nodes';     % N x 2
    t = pdeMesh.Elements';  % M x 3
    N = size(p, 1);

    % Ensure CCW orientation (FEM stiffness assumes positive area).
    v1 = p(t(:,2), :) - p(t(:,1), :);
    v2 = p(t(:,3), :) - p(t(:,1), :);
    cross_z = v1(:,1) .* v2(:,2) - v1(:,2) .* v2(:,1);
    flip = cross_z < 0;
    if any(flip)
        t(flip, [2,3]) = t(flip, [3,2]);
    end

    tol = h0 / 10;

    rect_left   = abs(p(:,1) - x1) < tol;
    rect_right  = abs(p(:,1) - x2) < tol;
    rect_bottom = abs(p(:,2) - y1) < tol;
    rect_top    = abs(p(:,2) - y2) < tol;

    all_bdry = rect_left | rect_right | rect_bottom | rect_top;
    neu = lower(neumann);
    if any(strcmp(neu, 'rect_left')),   all_bdry = all_bdry & ~rect_left;   end
    if any(strcmp(neu, 'rect_right')),  all_bdry = all_bdry & ~rect_right;  end
    if any(strcmp(neu, 'rect_bottom')), all_bdry = all_bdry & ~rect_bottom; end
    if any(strcmp(neu, 'rect_top')),    all_bdry = all_bdry & ~rect_top;    end
    Bdry = find(all_bdry);
    IN   = setdiff((1:N).', Bdry);

    bdry_masks.rect_left   = rect_left;
    bdry_masks.rect_right  = rect_right;
    bdry_masks.rect_bottom = rect_bottom;
    bdry_masks.rect_top    = rect_top;
    mesh = assemble_fem_struct(p, t, Bdry, IN, bdry_masks);
end
