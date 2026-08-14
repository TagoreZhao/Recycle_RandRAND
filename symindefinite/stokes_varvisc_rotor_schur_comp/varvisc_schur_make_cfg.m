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
    idx = find(strcmp(names, case_name), 1);
    if isempty(idx)
        error('varvisc_schur_make_cfg:unknownCase', ...
              'Unknown case "%s". Available: %s.', case_name, strjoin(names, ', '));
    end
    mcase = cases{idx}.factory(geo);

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
