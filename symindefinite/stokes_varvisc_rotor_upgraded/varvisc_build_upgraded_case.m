function [cfg, descriptor] = varvisc_build_upgraded_case(case_name, h0, dt, Tmax)
%VARVISC_BUILD_UPGRADED_CASE Build either promoted variable-viscosity case.
% Geometry and physical parameters are the promoted values from the verified
% stokes_varvisc_rotor upgrade; spectral and validation machinery is omitted.

    import src.discretization.*
    case_name = string(case_name);
    coupling_radius = 0.12;

    switch case_name
        case "current_channel_ar4"
            x1 = 0; x2 = 4; y1 = 0; y2 = 1;
            msh = build_channel_mesh_pde(h0, x1, x2, y1, y2, {'rect_right'});
            N = msh.N;
            left = find(msh.rect_left);
            walls = unique([find(msh.rect_top); find(msh.rect_bottom)]);
            bnodes = unique([left; walls]);
            yv = msh.p(bnodes, 2);
            uxv = zeros(numel(bnodes), 1);
            isleft = ismember(bnodes, left);
            uxv(isleft) = 4 * yv(isleft) .* (1 - yv(isleft));
            veldofs = [bnodes; N + bnodes];
            velvals = [uxv; zeros(numel(bnodes), 1)];
            velbc_fun = @(~) struct('dofs', veldofs, 'vals', velvals);
            [~, pin_node] = max(msh.p(:, 1));
            physical = make_channel_rotor(Tmax, 2, 0.5);
            geometry = "channel_ar4";
            boundary = "parabolic inflow; no-slip top/bottom; natural outflow";

        case "mixer_circle_four_blade"
            msh = build_disk_mesh(h0, 1.0);
            N = msh.N;
            bnodes = msh.Bdry;
            veldofs = [bnodes; N + bnodes];
            velbc_fun = @(~) struct('dofs', veldofs, ...
                'vals', zeros(2 * numel(bnodes), 1));
            [~, pin_node] = max(msh.p(:, 1));
            physical = make_circle_mixer(h0, Tmax);
            geometry = "closed_circle";
            boundary = "no-slip circular wall";

        otherwise
            error('varvisc_build_upgraded_case:unknownCase', ...
                'Unknown upgraded case %s.', case_name);
    end

    coupling_fun = @(mesh, node_count, X, V) ...
        varvisc_assemble_regularized_coupling( ...
            mesh, node_count, X, V, coupling_radius);
    cfg = struct('mesh', msh, 'nu_fun', physical.nu_fun, 'h0', h0, ...
        'velbc_fun', velbc_fun, 'motion_fun', physical.motion_fun, ...
        'coupling_fun', coupling_fun, 'coupling_radius', coupling_radius, ...
        'pin_node', pin_node, 'pin_val', 0, 'case_name', char(case_name), ...
        'geometry', char(geometry));

    initial_motion = physical.motion_fun(0);
    descriptor = struct('case_name', char(case_name), ...
        'geometry', char(geometry), 'boundary_conditions', char(boundary), ...
        'h0', h0, 'dt', dt, 'Tmax', Tmax, ...
        'nu_lo', physical.nu_lo, 'nu_hi', physical.nu_hi, ...
        'viscosity_contrast', physical.nu_hi / physical.nu_lo, ...
        'marker_count', size(initial_motion.X, 1), ...
        'coupling_mode', 'finite_radius_average', ...
        'coupling_radius', coupling_radius, ...
        'coupling_quadrature', '4 equal-area radial rings x 16 angles');
end

function msh = build_disk_mesh(h0, radius)
    import src.discretization.*
    C = [1; 0; 0; radius; 0; 0; 0; 0; 0; 0];
    g = decsg(C, 'C', char('C')');
    model = createpde();
    geometryFromEdges(model, g);
    pde_mesh = generateMesh(model, 'Hmax', h0, 'GeometricOrder', 'linear');
    p = pde_mesh.Nodes';
    t = pde_mesh.Elements';
    v1 = p(t(:, 2), :) - p(t(:, 1), :);
    v2 = p(t(:, 3), :) - p(t(:, 1), :);
    flip = v1(:, 1) .* v2(:, 2) - v1(:, 2) .* v2(:, 1) < 0;
    t(flip, [2, 3]) = t(flip, [3, 2]);
    wall = abs(sqrt(sum(p.^2, 2)) - radius) < h0 / 5;
    Bdry = find(wall);
    IN = setdiff((1:size(p, 1))', Bdry);
    masks = struct('wall', wall, 'rect_left', false(size(wall)), ...
        'rect_right', false(size(wall)), 'rect_bottom', false(size(wall)), ...
        'rect_top', false(size(wall)));
    msh = src.discretization.assemble_fem_struct(p, t, Bdry, IN, masks);
end

function S = make_circle_mixer(h0, Tmax)
    radius = 0.62;
    s = radius * [-1, -0.5, 0, 0.5, 1]';
    base = [s, zeros(size(s)); zeros(size(s)), s];
    [~, keep] = unique(round(base / (0.01 * max(h0, eps))), 'rows', 'stable');
    base = base(sort(keep), :);
    omega = 2 * pi / Tmax;
    S.nu_lo = 0.02;
    S.nu_hi = 1.0;
    S.motion_fun = @(t) rotate_points(base, t, omega);
    S.nu_fun = @(x, y, t) mixer_viscosity( ...
        x, y, t, omega, S.nu_lo, S.nu_hi);
end

function S = make_channel_rotor(Tmax, xc, yc)
    base = [-0.35; 0; 0.35];
    omega = 2 * pi / Tmax;
    S.nu_lo = 0.04;
    S.nu_hi = 2.0;
    S.motion_fun = @(t) channel_bar_motion(base, t, omega, xc, yc);
    S.nu_fun = @(x, y, t) channel_viscosity( ...
        x, y, t, omega, S.nu_lo, S.nu_hi, xc, yc);
end

function out = channel_bar_motion(base, t, omega, xc, yc)
    th = omega * t;
    X = [xc + base * cos(th), yc + base * sin(th)];
    V = omega * [-base * sin(th), base * cos(th)];
    out = struct('X', X, 'V', V);
end

function nu = channel_viscosity(x, y, t, omega, nu_lo, nu_hi, xc, yc)
    th = omega * t;
    radius = 0.35;
    sigma = 0.30;
    S = zeros(size(x));
    for phase = [0, pi]
        cx = xc + radius * cos(th + phase);
        cy = yc + radius * sin(th + phase);
        S = max(S, exp(-((x - cx).^2 + (y - cy).^2) / sigma^2));
    end
    xr = cos(th) * (x - xc) + sin(th) * (y - yc);
    yr = -sin(th) * (x - xc) + cos(th) * (y - yc);
    envelope = exp(-((x - xc).^2 + (y - yc).^2) / 0.9^2);
    bands = 0.5 * (1 + sin(8 * xr) .* sin(8 * yr));
    indicator = min(1, S + 0.15 * bands .* envelope);
    nu = nu_hi * (nu_lo / nu_hi).^indicator;
end

function out = rotate_points(base, t, omega)
    th = omega * t;
    R = [cos(th), -sin(th); sin(th), cos(th)];
    X = base * R';
    V = omega * [-X(:, 2), X(:, 1)];
    out = struct('X', X, 'V', V);
end

function nu = mixer_viscosity(x, y, t, omega, nu_lo, nu_hi)
    th = omega * t;
    xr = cos(th) * x + sin(th) * y;
    yr = -sin(th) * x + cos(th) * y;
    env = exp(-(x.^2 + y.^2) / 0.75^2);
    bands = 0.5 * (1 + sin(6 * xr + 0.7) .* sin(5 * yr - 0.4));
    right = exp(-((xr - 0.55).^2 + yr.^2) / 0.30^2);
    left = 0.74 * exp(-((xr + 0.55).^2 + yr.^2) / 0.30^2);
    upper = 0.88 * exp(-(xr.^2 + (yr - 0.55).^2) / 0.30^2);
    lower = 0.63 * exp(-(xr.^2 + (yr + 0.55).^2) / 0.30^2);
    batch = 0.82 * exp(-((xr - 0.24).^2 + (yr - 0.17).^2) / 0.22^2);
    tips = max(max(right, left), max(max(upper, lower), batch));
    indicator = min(1, tips + 0.15 * bands .* env);
    nu = nu_hi * (nu_lo / nu_hi).^indicator;
end
