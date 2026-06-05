function mesh = build_naca0012_mesh(hlead, htrail, hmax, circx, circr, neumann)
%BUILD_NACA0012_MESH  Mesh around a NACA0012 airfoil (distmesh2d Example 6).
%   MESH = BUILD_NACA0012_MESH(HLEAD, HTRAIL, HMAX, CIRCX, CIRCR, NEUMANN)
%
%   Inputs:
%     hlead   - element size at leading edge  (default 0.01)
%     htrail  - element size at trailing edge (default 0.04)
%     hmax    - max element size in far field (default 2)
%     circx   - x-centre of far-field circle (default 2)
%     circr   - radius of far-field circle   (default 4)
%     neumann - cell array of boundary names to treat as Neumann (default {})
%               Valid names: 'airfoil', 'farfield'
%
%   Output:
%     mesh - standard FEM struct (see assemble_fem_struct)
%            Boundary masks: mesh.airfoil, mesh.farfield

    if nargin < 1 || isempty(hlead),  hlead  = 0.01; end
    if nargin < 2 || isempty(htrail), htrail = 0.04; end
    if nargin < 3 || isempty(hmax),   hmax   = 2;    end
    if nargin < 4 || isempty(circx),  circx  = 2;    end
    if nargin < 5 || isempty(circr),  circr  = 4;    end
    if nargin < 6, neumann = {}; end

    import src.discretization.*

    % NACA0012 thickness coefficients
    a = .12/.2 * [0.2969, -0.1260, -0.3516, 0.2843, -0.1036];

    % Distance and size functions
    fd = @(p) ddiff(dcircle(p,circx,0,circr), ...
                    (abs(p(:,2)) - polyval([a(5:-1:2),0], p(:,1))).^2 - a(1)^2*p(:,1));
    fh = @(p) min(min(hlead + 0.3*dcircle(p,0,0,0), ...
                      htrail + 0.3*dcircle(p,1,0,0)), hmax);

    % Fixed nodes: far-field cardinal points, leading/trailing edge, refined trailing
    fixx  = 1 - htrail*cumsum(1.3.^(0:4)');
    fixx  = fixx(fixx > 0);
    fixy  = a(1)*sqrt(fixx) + polyval([a(5:-1:2),0], fixx);
    fix   = [[circx+[-1,1,0,0]*circr; 0,0,circr*[-1,1]]'; ...
             0,0; 1,0; fixx,fixy; fixx,-fixy];
    box   = [circx-circr, -circr; circx+circr, circr];
    h0    = min([hlead, htrail, hmax]);

    [p, t] = distmesh2d(fd, fh, h0, box, fix);
    N = size(p, 1);

    % --- Boundary detection ---
    % Airfoil surface: points where the airfoil level-set is near zero
    % and inside the chord range [0,1]
    airfoil_ls = (abs(p(:,2)) - polyval([a(5:-1:2),0], p(:,1))).^2 - a(1)^2*p(:,1);
    airfoil = (airfoil_ls >= -h0/10) & (p(:,1) >= -h0/10) & (p(:,1) <= 1+h0/10);

    % Far-field circle
    farfield = sqrt((p(:,1)-circx).^2 + p(:,2).^2) >= circr - h0*5;

    % --- DOF partitioning ---
    Bdry_mask = airfoil | farfield;
    neu = lower(neumann);
    if any(strcmp(neu, 'airfoil')),  Bdry_mask = Bdry_mask & ~airfoil;  end
    if any(strcmp(neu, 'farfield')), Bdry_mask = Bdry_mask & ~farfield; end
    Bdry = find(Bdry_mask);
    IN   = setdiff((1:N).', Bdry);

    bdry_masks.airfoil  = airfoil;
    bdry_masks.farfield = farfield;
    mesh = assemble_fem_struct(p, t, Bdry, IN, bdry_masks);
end
