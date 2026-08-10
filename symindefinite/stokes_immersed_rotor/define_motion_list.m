function [cases, resolved] = define_motion_list(dt, motion_params) %#ok<INUSD>
%DEFINE_MOTION_LIST  Immersed-solid motion cases for the Stokes-immersed-rotor
% benchmark (simplified deal.II step-70).
%
%   cases            = define_motion_list(dt)
%   cases            = define_motion_list(dt, motion_params)
%   [cases, resolved] = define_motion_list(dt, motion_params)
%
%   RESOLVED is motion_params with every default filled in -- what the factories
%   actually used, which is what callers should record rather than the sparse
%   struct they passed in.
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
%       case_struct.motion_meta - the resolved geometry, for downstream metadata
%
%   Each registry entry also carries .params_tag: a filename-safe encoding of the
%   NON-DEFAULT knobs that case consumes, empty when everything is at its default.
%   build_stokes_sequence appends it to the cache tag so two sweep points cannot
%   collide on one filename.  The disks consume no knobs, so theirs is always ''.
%
%   MOTION_PARAMS (optional; mirrors define_solver_list's getdef convention):
%     .theta0       initial phase of the rotating bar, radians    (default 0)
%     .Lb_frac      bar half-length as a fraction of the channel
%                   height; the tip-to-wall gap is (0.5 - Lb_frac)
%                   times the height, so this is a CLEARANCE knob  (default 0.35)
%     .bar_pt_frac  Lagrange-point spacing along the bar, as a
%                   multiple of h0                                 (default 0.6)
%
%   THE DEFAULTS REPRODUCE THE PREVIOUS LITERALS BIT-FOR-BIT.  Each literal was
%   replaced by a variable holding the identical double in the identical
%   expression, so no floating-point reassociation occurs, and theta0 = 0 adds
%   exactly zero.  test_motion_params T1/T2 assert this by uint64 comparison and
%   across the exact h0 values where the ceil() in nb tips -- because a one-point
%   change in nb would silently change nC for all 37 callers.
%
%   Cases:
%     1. bar_rotating     (STRESS) - thin rigid bar spinning about the centre;
%                                     Lagrange points sweep across the mesh.
%     2. disk_translating          - rigid disk advecting down the channel.
%     3. disk_static      (baseline) - fixed disk obstacle (coupling constant).
%
%   dt is accepted for interface parity with define_kappa_list and is unused.
%
%   See also: define_solver_list, build_stokes_sequence, assert_coupling_feasible.

if nargin < 2 || isempty(motion_params)
    motion_params = struct();
end
[mp, bar_tag] = resolve_motion_params(motion_params);
resolved = mp;

cases = {};

cases{end+1}.name = 'bar_rotating';
cases{end}.label  = 'Rotating rigid bar (immersed rotor)';
cases{end}.factory    = @(geo) make_bar_rotating(geo, mp);
cases{end}.params_tag = bar_tag;

cases{end+1}.name = 'disk_translating';
cases{end}.label  = 'Translating rigid disk';
cases{end}.factory    = @(geo) make_disk_translating(geo);
cases{end}.params_tag = '';        % consumes no motion_params

cases{end+1}.name = 'disk_static';
cases{end}.label  = 'Static rigid disk (baseline)';
cases{end}.factory    = @(geo) make_disk_static(geo);
cases{end}.params_tag = '';        % consumes no motion_params

cases = cases(:);
end

%==========================================================================
%  Parameter resolution
%==========================================================================

function [mp, bar_tag] = resolve_motion_params(in)
%RESOLVE_MOTION_PARAMS  Defaults, validation, and the cache-tag suffix.
%   A LOCAL SUBFUNCTION on purpose: the defaults and the tag rule must have
%   exactly one home, and a subfunction cannot be shadowed by a stray copy on the
%   path -- which a separate file could be, and which assert_woodbury_helpers
%   would then have to pin as another anti-shadow target.

    DEF = struct('theta0', 0, 'Lb_frac', 0.35, 'bar_pt_frac', 0.6);

    known = fieldnames(DEF);
    given = fieldnames(in);
    bad   = given(~ismember(given, known));
    if ~isempty(bad)
        % The disks ignore these knobs entirely, so a typo would otherwise be
        % silently absorbed and the run would report the default geometry.
        error('define_motion_list:unknownMotionParam', ...
              ['unknown motion_params field "%s".  Known fields: %s.  A typo ' ...
               'here is silent -- the geometry would fall back to its default ' ...
               'and the run would look successful.'], ...
              bad{1}, strjoin(known', ', '));
    end

    mp = DEF;
    for k = 1:numel(known)
        f = known{k};
        if isfield(in, f) && ~isempty(in.(f))
            mp.(f) = in.(f);
        end
    end

    check_scalar(mp.theta0,      'theta0');
    check_scalar(mp.Lb_frac,     'Lb_frac');
    check_scalar(mp.bar_pt_frac, 'bar_pt_frac');

    if ~(mp.Lb_frac > 0 && mp.Lb_frac < 0.5)
        % Strictly below 0.5: at 0.5 the tip reaches the wall, leaves the mesh,
        % and assemble_coupling silently drops it -- which surfaces much later as
        % build_stokes_sequence's "nC changed at step k" assert, naming the wrong
        % cause entirely.
        error('define_motion_list:barLeavesChannel', ...
              ['Lb_frac = %.6g must lie strictly in (0, 0.5).  At 0.5 the bar ' ...
               'tip reaches the channel wall and leaves the mesh; the point is ' ...
               'then dropped by assemble_coupling and the failure resurfaces as ' ...
               'a confusing "nC changed" assert several steps later.'], mp.Lb_frac);
    end
    if ~(mp.bar_pt_frac > 0)
        error('define_motion_list:badPointSpacing', ...
              'bar_pt_frac = %.6g must be positive.', mp.bar_pt_frac);
    end

    % Tag only what differs from the default, so every existing cache filename is
    % unchanged and only sweep points get a suffix.
    bar_tag = '';
    bar_tag = [bar_tag tag_if(mp.Lb_frac,     DEF.Lb_frac,     'Lb')];
    bar_tag = [bar_tag tag_if(mp.theta0,      DEF.theta0,      'th')];
    bar_tag = [bar_tag tag_if(mp.bar_pt_frac, DEF.bar_pt_frac, 'bp')];
end

%==========================================================================
function check_scalar(v, name)
    if ~(isnumeric(v) && isscalar(v) && isfinite(v))
        error('define_motion_list:badMotionParam', ...
              'motion_params.%s must be a finite numeric scalar.', name);
    end
end

%==========================================================================
function s = tag_if(v, dflt, name)
%TAG_IF  '' when v is the default, else a filename-safe '_<name><value>'.
%   %.12g rather than num2str: num2str keeps 5 significant digits, so two nearby
%   sweep points (0.4990 and 0.4991) would alias onto one cache file.  '.'->'p'
%   and '-'->'m' so the result survives build_stokes_sequence's
%   regexprep(tag,'[^\w]','_') sanitiser as a distinguishable string.
    if v == dflt
        s = '';
        return;
    end
    s = sprintf('_%s%s', name, ...
                strrep(strrep(sprintf('%.12g', v), '.', 'p'), '-', 'm'));
end

%==========================================================================
%  Case factories
%==========================================================================

function S = make_bar_rotating(geo, mp)
    Lb    = mp.Lb_frac * (geo.y2 - geo.y1);     % bar half-length
    nrev  = 2;                                  % revolutions over [0,Tmax]
    omega = 2 * pi * nrev / geo.Tmax;
    nb    = max(8, ceil(2 * Lb / (mp.bar_pt_frac * geo.h0)));
    s     = linspace(-Lb, Lb, nb)';            % arc-parameter along the bar

    S.nu        = 1.0;
    S.is_stress = true;
    S.motion_fun = @(t) bar_points(t, s, geo.xc, geo.yc, omega, mp.theta0);
    S.motion_meta = struct('kind', 'bar_rotating', 'theta0', mp.theta0, ...
                           'omega', omega, 'nrev', nrev, 'Lb', Lb, 'nb', nb, ...
                           'Lb_frac', mp.Lb_frac, 'bar_pt_frac', mp.bar_pt_frac, ...
                           'wall_gap', (0.5 - mp.Lb_frac) * (geo.y2 - geo.y1));
end

function out = bar_points(t, s, xc, yc, omega, theta0)
    % NOTE build_stokes_sequence evaluates step n at t = n*dt, so step 1 is
    % t = dt, NOT t = 0.  To place a chosen angle on step 1, the offset must be
    % theta0 = target - omega*dt = target - 2*pi*nrev/Tstep (dt cancels).
    th = theta0 + omega * t;
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
    S.motion_meta = struct('kind', 'disk_translating', 'rd', rd, 'vx', vx, ...
                           'x0', x0, 'npts', size(Pts, 1), 'pt_frac', 0.95);
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
    S.motion_meta = struct('kind', 'disk_static', 'rd', rd, 'cen', cen, ...
                           'npts', size(Pts, 1), 'pt_frac', 0.95);
end

%==========================================================================
%  Helpers
%==========================================================================

function Pts = disk_sample(rd, h0)
%DISK_SAMPLE  Interior sample points of a disk on a Cartesian grid clipped to
% the disk.
%
% THE SPACING IS A HARD CONSTRAINT, NOT A PREFERENCE.  The disk is sampled on a
% 2-D grid, so halving the spacing quadruples the constraint count while the
% velocity DOFs available to constrain are capped by the fluid mesh.  Measured at
% h0 = 0.03 (nC / touched DOFs / cond(Cap) for the step 1 -> 5 update):
%
%     1.50h  138 / 334 / 3.6e2      the original calibration
%     1.00h  308 / 392 / 8.5e3
%     0.95h  340 / 392 / 1.6e9      <- here: ill conditioned, still well posed
%     0.90h  370 / 382 / 2.9e35     passes the row count but cond(K_1) = 4e20
%     0.86h  416 / 402              refused: more rows than DOFs, K exactly singular
%
% 0.95*h0 is chosen to drive cond(Cap) hard while leaving cond(K_1) at 7.9e6,
% essentially its shipped 7.4e6 -- so the OPERATOR stays as well posed as before
% and only the low-rank update degrades.  Below ~0.9*h0 the operator itself goes,
% which is a different (and uninteresting) failure.  assert_coupling_feasible
% enforces the floor; note it is a row count and cannot see the 0.90h case.
%
% The bar is a 1-D sample and has a much lower floor, so make_bar_rotating uses
% its own factor (motion_params.bar_pt_frac, default 0.6).
    sp = 0.95 * h0;
    g  = -rd:sp:rd;
    [GX, GY] = meshgrid(g, g);
    inside = (GX.^2 + GY.^2) <= (0.95 * rd)^2;
    Pts = [GX(inside), GY(inside)];
    if isempty(Pts)
        Pts = [0, 0];                          % degenerate fallback
    end
end
