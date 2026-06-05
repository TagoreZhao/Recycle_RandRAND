function mesh = build_rect_with_hole_mesh(h0, x1, x2, y1, y2, cx, cy, cr, neumann)
%BUILD_RECT_WITH_HOLE_MESH  Mesh a rectangle with a circular hole (distmesh2d Example 2).
%   MESH = BUILD_RECT_WITH_HOLE_MESH(H0, X1, X2, Y1, Y2, CX, CY, CR, NEUMANN)
%
%   Inputs:
%     h0      - mesh size        (default 0.05)
%     x1,x2   - x extent         (default -1, 1)
%     y1,y2   - y extent         (default -1, 1)
%     cx,cy   - hole centre      (default 0, 0)
%     cr      - hole radius      (default 0.5)
%     neumann - cell array of boundary names to treat as Neumann (default {})
%               Valid names: 'rect_left','rect_right','rect_bottom','rect_top','circle_hole'
%
%   Output:
%     mesh - standard FEM struct (see assemble_fem_struct)
%            Boundary masks: mesh.rect_left, .rect_right, .rect_bottom,
%                            .rect_top, .circle_hole

    if nargin < 1 || isempty(h0), h0 = 0.05; end
    if nargin < 2 || isempty(x1), x1 = -1;   end
    if nargin < 3 || isempty(x2), x2 =  1;   end
    if nargin < 4 || isempty(y1), y1 = -1;   end
    if nargin < 5 || isempty(y2), y2 =  1;   end
    if nargin < 6 || isempty(cx), cx =  0;   end
    if nargin < 7 || isempty(cy), cy =  0;   end
    if nargin < 8 || isempty(cr), cr =  0.5; end
    if nargin < 9, neumann = {}; end

    import src.discretization.*

    fd = @(p) ddiff(drectangle(p,x1,x2,y1,y2), dcircle(p,cx,cy,cr));
    fh = @(p) 0.05 + 0.3*dcircle(p,cx,cy,cr);
    pfix = [x1,y1; x1,y2; x2,y1; x2,y2];
    [p, t] = distmesh2d(fd, fh, h0, [x1,y1; x2,y2], pfix);
    N = size(p, 1);

    tol = h0/10;

    % --- Boundary detection ---
    rect_left   = abs(p(:,1) - x1) < tol;
    rect_right  = abs(p(:,1) - x2) < tol;
    rect_bottom = abs(p(:,2) - y1) < tol;
    rect_top    = abs(p(:,2) - y2) < tol;
    circle_hole = sqrt((p(:,1)-cx).^2 + (p(:,2)-cy).^2) <= cr + tol;

    % --- DOF partitioning ---
    all_bdry = rect_left | rect_right | rect_bottom | rect_top | circle_hole;
    % Remove Neumann segments from Dirichlet set
    neu = lower(neumann);
    if any(strcmp(neu, 'rect_left')),   all_bdry = all_bdry & ~rect_left;   end
    if any(strcmp(neu, 'rect_right')),  all_bdry = all_bdry & ~rect_right;  end
    if any(strcmp(neu, 'rect_bottom')), all_bdry = all_bdry & ~rect_bottom; end
    if any(strcmp(neu, 'rect_top')),    all_bdry = all_bdry & ~rect_top;    end
    if any(strcmp(neu, 'circle_hole')), all_bdry = all_bdry & ~circle_hole; end
    Bdry = find(all_bdry);
    IN   = setdiff((1:N).', Bdry);

    bdry_masks.rect_left   = rect_left;
    bdry_masks.rect_right  = rect_right;
    bdry_masks.rect_bottom = rect_bottom;
    bdry_masks.rect_top    = rect_top;
    bdry_masks.circle_hole = circle_hole;
    mesh = assemble_fem_struct(p, t, Bdry, IN, bdry_masks);
end
