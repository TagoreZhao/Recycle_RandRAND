function Kloc = tri_stiff_loc_surface(xy)
%TRI_STIFF_LOC_SURFACE  Unit-kappa Laplace-Beltrami stiffness for a P1
%   surface triangle embedded in 3D.
%
%   Kloc = tri_stiff_loc_surface(xy)
%
%   Input:
%     xy - 3x3 matrix, rows are vertex coordinates (x_i, y_i, z_i)
%
%   Output:
%     Kloc - 3x3 symmetric positive semi-definite element stiffness matrix
%
%   Uses the cotangent formula:
%     e_i = edge opposite vertex i
%     Kloc(i,j) = dot(e_i, e_j) / (4*A)
%   where A is the triangle area computed via cross product.
%
%   On a flat triangle (z=0), this matches tri_stiff_loc exactly.

    % Edges opposite each vertex
    e1 = xy(3,:) - xy(2,:);   % opposite vertex 1
    e2 = xy(1,:) - xy(3,:);   % opposite vertex 2
    e3 = xy(2,:) - xy(1,:);   % opposite vertex 3

    % Triangle area via cross product
    n = cross(e3, -e2);       % = cross(v2-v1, v3-v1)
    A = norm(n) / 2;

    % Cotangent stiffness: Kloc(i,j) = dot(e_i, e_j) / (4*A)
    edges = [e1; e2; e3];     % 3x3
    Kloc = (edges * edges') / (4 * A);
end
