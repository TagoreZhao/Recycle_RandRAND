function Mloc = tri_mass_loc(xy)
% Consistent mass for linear P1 triangle
% xy: 3x2, rows are (x_i,y_i)
    x1=xy(1,1); y1=xy(1,2);
    x2=xy(2,1); y2=xy(2,2);
    x3=xy(3,1); y3=xy(3,2);
    A = abs(det([1 x1 y1; 1 x2 y2; 1 x3 y3]))/2;
    Mloc = (A/12)*[2 1 1; 1 2 1; 1 1 2];
end