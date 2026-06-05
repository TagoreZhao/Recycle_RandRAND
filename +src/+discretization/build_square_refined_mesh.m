function mesh = build_square_refined_mesh(h0, neumann)
%BUILD_SQUARE_REFINED_MESH  Mesh the unit square with adaptive refinement (distmesh2d Example 5).
%   MESH = BUILD_SQUARE_REFINED_MESH(H0, NEUMANN)
%
%   Size function has a point source at the origin and a line source from
%   (0.3,0.7) to (0.7,0.5), producing strong local refinement.
%
%   Inputs:
%     h0      - mesh size  (default 0.01)
%     neumann - cell array of boundary names to treat as Neumann (default {})
%               Valid names: 'left','right','bottom','top','square_bdry'
%
%   Output:
%     mesh - standard FEM struct (see assemble_fem_struct)
%            Boundary masks: mesh.left, .right, .bottom, .top, .square_bdry

    if nargin < 1 || isempty(h0), h0 = 0.01; end
    if nargin < 2, neumann = {}; end

    import src.discretization.*

    fd = @(p) drectangle(p, 0, 1, 0, 1);
    fh = @(p) min(min(0.01 + 0.3*abs(dcircle(p,0,0,0)), ...
                      0.025 + 0.3*abs(dpoly(p,[0.3,0.7; 0.7,0.5]))), 0.15);
    pfix = [0,0; 1,0; 0,1; 1,1];
    [p, t] = distmesh2d(fd, fh, h0, [0,0; 1,1], pfix);
    N = size(p, 1);

    tol = h0/5;   % slightly looser for very fine mesh

    % --- Boundary detection ---
    left   = abs(p(:,1))     < tol;
    right  = abs(p(:,1) - 1) < tol;
    bottom = abs(p(:,2))     < tol;
    top    = abs(p(:,2) - 1) < tol;
    square_bdry = left | right | bottom | top;

    % --- DOF partitioning ---
    Bdry_mask = square_bdry;
    neu = lower(neumann);
    if any(strcmp(neu, 'square_bdry'))
        Bdry_mask = false(N, 1);
    else
        if any(strcmp(neu, 'left')),   Bdry_mask = Bdry_mask & ~left;   end
        if any(strcmp(neu, 'right')),  Bdry_mask = Bdry_mask & ~right;  end
        if any(strcmp(neu, 'bottom')), Bdry_mask = Bdry_mask & ~bottom; end
        if any(strcmp(neu, 'top')),    Bdry_mask = Bdry_mask & ~top;    end
    end
    Bdry = find(Bdry_mask);
    IN   = setdiff((1:N).', Bdry);

    bdry_masks.left = left;
    bdry_masks.right = right;
    bdry_masks.bottom = bottom;
    bdry_masks.top = top;
    bdry_masks.square_bdry = square_bdry;
    mesh = assemble_fem_struct(p, t, Bdry, IN, bdry_masks);
end
