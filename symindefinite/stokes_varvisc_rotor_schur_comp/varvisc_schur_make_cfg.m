function [cfg, msh] = varvisc_schur_make_cfg(case_name, params, msh)
%VARVISC_SCHUR_MAKE_CFG  Build one variable-viscosity benchmark case.

    import src.discretization.*

    x1 = 0; x2 = 4; y1 = 0; y2 = 1;
    Lyc = y2 - y1;
    Uin = 1.0;

    if nargin < 3 || isempty(msh)
        msh = build_channel_mesh_pde(params.h0, x1, x2, y1, y2, {'rect_right'});
    end
    N = msh.N;

    left = find(msh.rect_left);
    walls = unique([find(msh.rect_top); find(msh.rect_bottom)]);
    bnodes = unique([left; walls]);
    yv = msh.p(bnodes, 2);
    uxv = zeros(numel(bnodes), 1);
    isleft = ismember(bnodes, left);
    uxv(isleft) = Uin * 4 .* yv(isleft) .* (Lyc - yv(isleft)) / Lyc^2;
    veldofs = [bnodes; N + bnodes];
    velvals = [uxv; zeros(numel(bnodes), 1)];
    [~, pin_node] = max(msh.p(:, 1));

    geo = struct('x1', x1, 'x2', x2, 'y1', y1, 'y2', y2, ...
        'xc', (x1+x2)/2, 'yc', (y1+y2)/2, 'h0', params.h0, ...
        'Tmax', params.dt * (params.Tstep - 1));
    cases = varvisc_define_case_list(params.dt);
    names = cellfun(@(c) c.name, cases, 'UniformOutput', false);
    hard_case = 'disk_static_nu_checkerboard_shift';
    if strcmp(case_name, hard_case)
        % Schur-local adversarial case: retain the static disk so C_n is
        % constant, but translate a smooth 100:1 checkerboard by half a
        % wavelength.  The endpoint fields exchange their high/low regions.
        idx = find(strcmp(names, 'disk_static_nu_const'), 1);
        mcase = cases{idx}.factory(geo);
        mcase.nu_lo = 0.02;
        mcase.nu_hi = 2.0;
        mcase.nu_fun = @(x,y,t) local_checkerboard_viscosity( ...
            x,y,t,params.dt,geo.Tmax,mcase.nu_lo,mcase.nu_hi);
        mcase.is_stress = true;
    else
        idx = find(strcmp(names, case_name), 1);
        if isempty(idx)
            available = [names, {hard_case}];
            error('varvisc_schur_make_cfg:unknownCase', ...
                  'Unknown case "%s". Available: %s.', ...
                  case_name, strjoin(available, ', '));
        end
        mcase = cases{idx}.factory(geo);
    end

    cfg = struct();
    cfg.mesh = msh;
    cfg.nu_fun = mcase.nu_fun;
    cfg.h0 = params.h0;
    cfg.velbc_fun = @(t) struct('dofs', veldofs, 'vals', velvals);
    cfg.motion_fun = mcase.motion_fun;
    cfg.pin_node = pin_node;
    cfg.pin_val = 0;
    cfg.case_name = case_name;
    cfg.geometry = 'stokes_varvisc_rotor_schur_comp';
    cfg.is_stress = mcase.is_stress;
    cfg.veldofs = veldofs;
    cfg.velvals = velvals;
end

function nu = local_checkerboard_viscosity(x,y,t,dt,Tmax,nu_lo,nu_hi)
%LOCAL_CHECKERBOARD_VISCOSITY  Smooth complementary log-viscosity texture.
%   The phase is zero at the first solved time t=dt and pi at t=Tmax.  A pi
%   phase shift negates q, so nu(dt).*nu(Tmax) = nu_lo*nu_hi pointwise.

    alpha = 2;
    wave_number = 8;
    duration = max(Tmax-dt, eps(max(Tmax,dt)));
    progress = min(max((t-dt)/duration,0),1);
    phase = pi*progress;
    z = sin(wave_number*x+phase).*sin(wave_number*y);
    q = tanh(alpha*z)/tanh(alpha);
    mix = 0.5*(1+q);
    nu = nu_hi*(nu_lo/nu_hi).^mix;
end
