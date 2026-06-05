function Mloc = tri_mass_loc_surface(xy)
%TRI_MASS_LOC_SURFACE  Consistent P1 mass matrix for a surface triangle
%   embedded in 3D.
%
%   Mloc = tri_mass_loc_surface(xy)
%
%   Input:
%     xy - 3x3 matrix, rows are vertex coordinates (x_i, y_i, z_i)
%
%   Output:
%     Mloc - 3x3 symmetric positive definite element mass matrix
%
%   Formula:  Mloc = (A/12) * [2 1 1; 1 2 1; 1 1 2]
%   where A is the triangle area computed via cross product.
%
%   On a flat triangle (z=0), this matches tri_mass_loc exactly.

    % Triangle area via cross product
    v1 = xy(2,:) - xy(1,:);
    v2 = xy(3,:) - xy(1,:);
    A = norm(cross(v1, v2)) / 2;

    Mloc = (A / 12) * [2 1 1; 1 2 1; 1 1 2];
end
