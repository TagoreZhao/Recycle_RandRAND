function mesh = build_unit_circle_mesh(h0, R, neumann)
%BUILD_UNIT_CIRCLE_MESH  Mesh a circular domain (distmesh2d Example 1).
%   MESH = BUILD_UNIT_CIRCLE_MESH(H0, R, NEUMANN)
%
%   Inputs:
%     h0      - mesh size  (default 0.2)
%     R       - radius     (default 1)
%     neumann - cell array of boundary names to treat as Neumann (default {})
%               Valid names: 'circle'
%
%   Output:
%     mesh - standard FEM struct (see assemble_fem_struct)
%            Boundary mask: mesh.circle

    if nargin < 1 || isempty(h0), h0 = 0.2; end
    if nargin < 2 || isempty(R),  R  = 1;   end
    if nargin < 3, neumann = {}; end

    import src.discretization.*

    fd = @(p) sqrt(sum(p.^2, 2)) - R;
    [p, t] = distmesh2d(fd, @huniform, h0, [-R,-R; R,R], []);
    N = size(p, 1);

    % --- Boundary detection ---
    circle = sqrt(sum(p.^2, 2)) >= R - h0/10;

    % --- DOF partitioning ---
    Bdry_mask = circle;
    if any(strcmpi(neumann, 'circle'))
        Bdry_mask = false(N, 1);  % no Dirichlet
    end
    Bdry = find(Bdry_mask);
    IN   = setdiff((1:N).', Bdry);

    bdry_masks.circle = circle;
    mesh = assemble_fem_struct(p, t, Bdry, IN, bdry_masks);
end
