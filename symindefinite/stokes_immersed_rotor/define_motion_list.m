function cases = define_motion_list(dt) %#ok<INUSD>
%DEFINE_MOTION_LIST  Immersed-solid motion cases for the Stokes-immersed-rotor
% benchmark (simplified deal.II step-70).
%
%   cases = define_motion_list(dt)
%
%   Each case factory returns a STRUCT describing the immersed rigid solid and
%   its prescribed motion.  The solid is enforced on the fluid by distributed
%   Lagrange multipliers at a set of points X_k(t) that MOVE with the solid;
%   only the coupling block C(t_n) of the KKT system changes per step.
%
%   Factory signature:
%       @(geo) -> case_struct
%   where geo has fields .x1 .x2 .y1 .y2 .xc .yc .h0 .Tmax, and
%       case_struct.nu          - viscosity for this case
%       case_struct.motion_fun  - @(t) -> struct('X', K x 2, 'V', K x 2)
%       case_struct.is_stress   - logical, true for the moving stress case
%
%   Cases:
%     1. bar_rotating     (STRESS) - thin rigid bar spinning about the centre;
%                                     Lagrange points sweep across the mesh.
%     2. disk_translating          - rigid disk advecting down the channel.
%     3. disk_static      (baseline) - fixed disk obstacle (coupling constant).
%
%   dt is accepted for interface parity with define_kappa_list and is unused.

cases = {};

cases{end+1}.name = 'bar_rotating';
cases{end}.label  = 'Rotating rigid bar (immersed rotor)';
cases{end}.factory = @(geo) make_bar_rotating(geo);

cases{end+1}.name = 'disk_translating';
cases{end}.label  = 'Translating rigid disk';
cases{end}.factory = @(geo) make_disk_translating(geo);

cases{end+1}.name = 'disk_static';
cases{end}.label  = 'Static rigid disk (baseline)';
cases{end}.factory = @(geo) make_disk_static(geo);

cases = cases(:);
end

%==========================================================================
%  Case factories
%==========================================================================

function S = make_bar_rotating(geo)
    Lb    = 0.35 * (geo.y2 - geo.y1);          % bar half-length
    nrev  = 2;                                  % revolutions over [0,Tmax]
    omega = 2 * pi * nrev / geo.Tmax;
    nb    = max(8, ceil(2 * Lb / (1.5 * geo.h0)));
    s     = linspace(-Lb, Lb, nb)';            % arc-parameter along the bar

    S.nu        = 1.0;
    S.is_stress = true;
    S.motion_fun = @(t) bar_points(t, s, geo.xc, geo.yc, omega);
end

function out = bar_points(t, s, xc, yc, omega)
    th = omega * t;
    dir = [cos(th), sin(th)];
    X = [xc + s * dir(1), yc + s * dir(2)];     % K x 2 points along the bar
    % Rigid rotation velocity v = omega x r = omega*s*[-sin th, cos th]
    V = omega * [-s * sin(th), s * cos(th)];
    out = struct('X', X, 'V', V);
end

function S = make_disk_translating(geo)
    rd = 0.22 * (geo.y2 - geo.y1);
    x0 = geo.x1 + 0.6;        % start near inflow
    xend = geo.x2 - 0.6;      % finish near outflow
    vx = (xend - x0) / geo.Tmax;
    Xc0 = [x0, geo.yc];
    Pts = disk_sample(rd, geo.h0);              % K x 2 body-frame interior points

    S.nu        = 1.0;
    S.is_stress = false;
    S.motion_fun = @(t) disk_points(t, Pts, Xc0, vx);
end

function out = disk_points(t, Pts, Xc0, vx)
    cen = [Xc0(1) + vx * t, Xc0(2)];
    X = Pts + cen;                              % translate body points
    V = repmat([vx, 0], size(Pts, 1), 1);      % rigid translation velocity
    out = struct('X', X, 'V', V);
end

function S = make_disk_static(geo)
    rd  = 0.22 * (geo.y2 - geo.y1);
    cen = [geo.x1 + 0.35 * (geo.x2 - geo.x1), geo.yc];
    Pts = disk_sample(rd, geo.h0) + cen;

    S.nu        = 1.0;
    S.is_stress = false;
    S.motion_fun = @(t) struct('X', Pts, 'V', zeros(size(Pts, 1), 2));
end

%==========================================================================
%  Helpers
%==========================================================================

function Pts = disk_sample(rd, h0)
%DISK_SAMPLE  Interior sample points of a disk on a Cartesian grid clipped to
% the disk, spaced ~1.5*h0 so the coupling rows stay linearly independent.
    sp = 1.5 * h0;
    g  = -rd:sp:rd;
    [GX, GY] = meshgrid(g, g);
    inside = (GX.^2 + GY.^2) <= (0.95 * rd)^2;
    Pts = [GX(inside), GY(inside)];
    if isempty(Pts)
        Pts = [0, 0];                          % degenerate fallback
    end
end
