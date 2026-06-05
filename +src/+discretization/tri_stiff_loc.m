function Kloc = tri_stiff_loc(xy)
% Unit-kappa stiffness for linear P1 triangle
    x1=xy(1,1); y1=xy(1,2);
    x2=xy(2,1); y2=xy(2,2);
    x3=xy(3,1); y3=xy(3,2);
    A = abs(det([1 x1 y1; 1 x2 y2; 1 x3 y3]))/2;
    b = [y2 - y3; y3 - y1; y1 - y2];
    c = [x3 - x2; x1 - x3; x2 - x1];
    Kloc = (b*b' + c*c')/(4*A);
end
