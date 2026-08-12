function cases = varvisc_define_case_list(dt) %#ok<INUSD>
%DEFINE_CASE_LIST  Motion + viscosity cases for the variable-viscosity
% Stokes-immersed-rotor benchmark.
%
%   cases = varvisc_define_case_list(dt)
%
%   Each case pairs an immersed rigid-solid motion (distributed Lagrange
%   multipliers, as in bench/stokes_immersed_rotor) with a moving
%   high-contrast viscosity field nu(x,t).  Unlike the parent benchmark —
%   where only the coupling border C(t) changes per step (a rank <= 2*nC
%   structured update) — here the nu-scaled velocity stiffness and the
%   nu-weighted stabilization block change at every nonzero, every step.
%
%   Factory signature:
%       @(geo) -> case_struct
%   where geo has fields .x1 .x2 .y1 .y2 .xc .yc .h0 .Tmax, and
%       case_struct.nu_fun      - @(xc, yc, t) -> M x 1 element viscosities
%                                 (evaluated at element centroids)
%       case_struct.motion_fun  - @(t) -> struct('X', K x 2, 'V', K x 2)
%       case_struct.is_stress   - logical, true for the moving stress case
%       case_struct.nu_lo, .nu_hi - viscosity bounds (contrast = nu_hi/nu_lo)
%
%   Cases:
%     1. bar_rotating_nu_orbiting  (STRESS) - rotating rigid bar with two
%        log-Gaussian low-viscosity blobs riding the bar tips (100:1) PLUS
%        a high-frequency striation texture rotating rigidly with the rotor
%        (the filamentary field chaotic advection produces in real stirred
%        mixing).  The striations are what make the per-step matrix change
%        genuinely HIGH-RANK: two smooth blobs alone yield a numerically
%        compressible difference (r90 ~ 30), while the sign-oscillating
%        striation difference spreads Frobenius mass across O(N) singular
%        directions.  nrev = 3 so the blob centroids move >= 2*h0 per
%        production step.
%     2. disk_translating_nu_wake  - translating rigid disk with one
%        low-viscosity blob trailing in its wake (50:1, smooth).
%     3. disk_static_nu_const      (control) - static disk, nu == 1:
%        K(t) is CONSTANT, reproducing the parent's degenerate regime;
%        the frozen-vs-refreshed preconditioner gap must vanish here.
%
%   The viscosity profile is log-Gaussian with a clamped exponent so the
%   bounds are exact:
%       S(x,t)  = min(1, max_k exp(-||x - c_k(t)||^2 / sigma^2)
%                        + w_str * striation(x,t))
%       nu(x,t) = nu_hi * (nu_lo/nu_hi)^S(x,t)   in [nu_lo, nu_hi]
%
%   dt is accepted for interface parity with define_kappa_list and is unused.

cases = {};

cases{end+1}.name = 'bar_rotating_nu_orbiting';
cases{end}.label  = 'Rotating bar + orbiting low-viscosity blobs (100:1)';
cases{end}.factory = @(geo) make_bar_rotating_nu_orbiting(geo);

cases{end+1}.name = 'disk_translating_nu_wake';
cases{end}.label  = 'Translating disk + wake low-viscosity blob (50:1)';
cases{end}.factory = @(geo) make_disk_translating_nu_wake(geo);

cases{end+1}.name = 'disk_static_nu_const';
cases{end}.label  = 'Static disk, constant viscosity (control)';
cases{end}.factory = @(geo) make_disk_static_nu_const(geo);

cases = cases(:);
end

%==========================================================================
%  Case factories
%==========================================================================

function S = make_bar_rotating_nu_orbiting(geo)
% Rotating bar (as the parent's bar_rotating, but nrev = 3) plus two
% low-viscosity blobs orbiting in phase with the bar tips — physically,
% thermally-thinned fluid shed at the rotor tips.
    Lb    = 0.35 * (geo.y2 - geo.y1);          % bar half-length
    nrev  = 3;                                  % revolutions over [0,Tmax]
    omega = 2 * pi * nrev / geo.Tmax;
    nb    = max(8, ceil(2 * Lb / (1.5 * geo.h0)));
    s     = linspace(-Lb, Lb, nb)';

    nu_lo = 0.02;  nu_hi = 2.0;                % contrast 100:1
    sigma = 0.18;                               % blob width
    R_orb = Lb;                                 % blobs ride the bar tips
    phis  = [0, pi];                            % one blob per bar end
    kstr  = 16;                                 % striation wavenumber (lambda ~ 0.39)
    w_str = 0.5;                                % striation strength (10:1 stripes alone)
    sig_env = 0.9;                              % striation envelope radius (fills the
                                                % stirred channel core, decays at in/outflow)

    S.nu_lo     = nu_lo;
    S.nu_hi     = nu_hi;
    S.is_stress = true;
    S.motion_fun = @(t) bar_points(t, s, geo.xc, geo.yc, omega);
    S.nu_fun     = @(x, y, t) nu_orbiting_blobs_striated(x, y, t, ...
        geo.xc, geo.yc, R_orb, omega, phis, sigma, nu_lo, nu_hi, ...
        kstr, w_str, sig_env);
end

function S = make_disk_translating_nu_wake(geo)
% Translating rigid disk with a single low-viscosity blob trailing in its
% wake at a fixed offset.
    rd = 0.22 * (geo.y2 - geo.y1);
    x0 = geo.x1 + 0.6;
    xend = geo.x2 - 0.6;
    vx = (xend - x0) / geo.Tmax;
    Xc0 = [x0, geo.yc];
    Pts = disk_sample(rd, geo.h0);

    nu_lo = 0.04;  nu_hi = 2.0;                % contrast 50:1
    sigma = 0.18;
    wake_offset = [-0.4, 0];                    % blob trails the disk

    S.nu_lo     = nu_lo;
    S.nu_hi     = nu_hi;
    S.is_stress = false;
    S.motion_fun = @(t) disk_points(t, Pts, Xc0, vx);
    S.nu_fun     = @(x, y, t) nu_single_blob(x, y, ...
        Xc0(1) + vx * t + wake_offset(1), Xc0(2) + wake_offset(2), ...
        sigma, nu_lo, nu_hi);
end

function S = make_disk_static_nu_const(geo)
% Control case: constant viscosity, static solid — K(t) constant, so both
% the full-rank-update claim and the preconditioner-staleness gap have a
% built-in negative control.
    rd  = 0.22 * (geo.y2 - geo.y1);
    cen = [geo.x1 + 0.35 * (geo.x2 - geo.x1), geo.yc];
    Pts = disk_sample(rd, geo.h0) + cen;

    S.nu_lo     = 1.0;
    S.nu_hi     = 1.0;
    S.is_stress = false;
    S.motion_fun = @(t) struct('X', Pts, 'V', zeros(size(Pts, 1), 2));
    S.nu_fun     = @(x, y, t) ones(size(x));
end

%==========================================================================
%  Viscosity fields
%==========================================================================

function nu = nu_orbiting_blobs_striated(x, y, t, xc, yc, R_orb, omega, phis, ...
                                          sigma, nu_lo, nu_hi, kstr, w_str, sig_env)
% Log-Gaussian blobs orbiting the domain centre in phase with the rotor,
% plus a high-frequency striation texture rotating rigidly with the rotor
% (enveloped near the swept region).  The striations make the per-step
% stiffness difference genuinely high-rank: its sign oscillates on the
% striation wavelength, so the svd energy spreads over O(N) modes instead
% of the ~30 a smooth blob pair gives.  The exponent is clamped to [0,1]
% so nu stays in [nu_lo, nu_hi] exactly.
    S = zeros(size(x));
    th = omega * t;
    for k = 1:numel(phis)
        cxk = xc + R_orb * cos(th + phis(k));
        cyk = yc + R_orb * sin(th + phis(k));
        S = max(S, exp(-((x - cxk).^2 + (y - cyk).^2) / sigma^2));
    end
    % rotating-frame coordinates: the texture co-rotates with the bar
    xr =  cos(th) * (x - xc) + sin(th) * (y - yc);
    yr = -sin(th) * (x - xc) + cos(th) * (y - yc);
    env = exp(-((x - xc).^2 + (y - yc).^2) / sig_env^2);
    mix = 0.5 * (1 + sin(kstr * xr) .* sin(kstr * yr));
    S = min(1, S + w_str * mix .* env);
    nu = nu_hi * (nu_lo / nu_hi).^S;
end

function nu = nu_single_blob(x, y, cx, cy, sigma, nu_lo, nu_hi)
    S = exp(-((x - cx).^2 + (y - cy).^2) / sigma^2);
    nu = nu_hi * (nu_lo / nu_hi).^S;
end

%==========================================================================
%  Motion helpers (as in bench/stokes_immersed_rotor/define_motion_list.m)
%==========================================================================

function out = bar_points(t, s, xc, yc, omega)
    th = omega * t;
    dir = [cos(th), sin(th)];
    X = [xc + s * dir(1), yc + s * dir(2)];
    V = omega * [-s * sin(th), s * cos(th)];
    out = struct('X', X, 'V', V);
end

function out = disk_points(t, Pts, Xc0, vx)
    cen = [Xc0(1) + vx * t, Xc0(2)];
    X = Pts + cen;
    V = repmat([vx, 0], size(Pts, 1), 1);
    out = struct('X', X, 'V', V);
end

function Pts = disk_sample(rd, h0)
%DISK_SAMPLE  Interior sample points of a disk on a Cartesian grid clipped to
% the disk, spaced ~1.5*h0 so the coupling rows stay linearly independent.
    sp = 1.5 * h0;
    g  = -rd:sp:rd;
    [GX, GY] = meshgrid(g, g);
    inside = (GX.^2 + GY.^2) <= (0.95 * rd)^2;
    Pts = [GX(inside), GY(inside)];
    if isempty(Pts)
        Pts = [0, 0];
    end
end
