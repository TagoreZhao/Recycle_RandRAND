function mesh = build_ellipse_mesh(h0, a, b, neumann)
%BUILD_ELLIPSE_MESH  Mesh an elliptical domain (distmesh2d Example 4).
%   MESH = BUILD_ELLIPSE_MESH(H0, A, B, NEUMANN)
%
%   Inputs:
%     h0      - mesh size       (default 0.2)
%     a       - x semi-axis     (default 2)
%     b       - y semi-axis     (default 1)
%     neumann - cell array of boundary names to treat as Neumann (default {})
%               Valid names: 'ellipse'
%
%   Output:
%     mesh - standard FEM struct (see assemble_fem_struct)
%            Boundary mask: mesh.ellipse

    if nargin < 1 || isempty(h0), h0 = 0.2; end
    if nargin < 2 || isempty(a),  a  = 2;   end
    if nargin < 3 || isempty(b),  b  = 1;   end
    if nargin < 4, neumann = {}; end

    import src.discretization.*

    fd = @(p) p(:,1).^2/a^2 + p(:,2).^2/b^2 - 1;
    [p, t] = distmesh2d(fd, @huniform, h0, [-a,-b; a,b], []);
    N = size(p, 1);

    % --- Boundary detection ---
    ellipse = (p(:,1).^2/a^2 + p(:,2).^2/b^2) >= 1 - h0/(10*min(a,b));

    % --- DOF partitioning ---
    Bdry_mask = ellipse;
    if any(strcmpi(neumann, 'ellipse'))
        Bdry_mask = false(N, 1);
    end
    Bdry = find(Bdry_mask);
    IN   = setdiff((1:N).', Bdry);

    bdry_masks.ellipse = ellipse;
    mesh = assemble_fem_struct(p, t, Bdry, IN, bdry_masks);
end
