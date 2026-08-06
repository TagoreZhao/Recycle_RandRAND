function [cfg, msh] = schur_make_cfg(case_name, params, msh)
%SCHUR_MAKE_CFG  Build the per-case cfg for the Schur-complement benchmark.
%   [CFG, MSH] = SCHUR_MAKE_CFG(CASE_NAME, PARAMS)
%   [CFG, MSH] = SCHUR_MAKE_CFG(CASE_NAME, PARAMS, MSH)   reuse an existing mesh
%
%   Mirrors the channel geometry, inflow parabola and pressure pin of the
%   sibling benchmark (symindefinite/stokes_immersed_rotor/run_benchmark.m) so
%   the two studies solve the SAME physical sequence and their numbers are
%   directly comparable.  The mesh is expensive, so the driver builds it once
%   and passes it back in.
%
%   CASE_NAME is one of the define_motion_list entries: 'bar_rotating',
%   'disk_translating', 'disk_static'.
%
%   See also: schur_context_init, define_motion_list.

    import src.discretization.*

    % Channel geometry (identical to the sibling benchmark)
    x1 = 0; x2 = 4; y1 = 0; y2 = 1;
    Lyc = y2 - y1;
    Uin = 1.0;                       % peak inflow velocity

    if nargin < 3 || isempty(msh)
        msh = build_channel_mesh_pde(params.h0, x1, x2, y1, y2, {'rect_right'});
    end
    N = msh.N;

    % --- Velocity Dirichlet: parabolic inflow left, no-slip walls, free outflow
    left   = find(msh.rect_left);
    walls  = unique([find(msh.rect_top); find(msh.rect_bottom)]);
    bnodes = unique([left; walls]);
    yv     = msh.p(bnodes, 2);
    uxv    = zeros(numel(bnodes), 1);
    isleft = ismember(bnodes, left);
    uxv(isleft) = Uin * 4 .* yv(isleft) .* (Lyc - yv(isleft)) / Lyc^2;
    veldofs = [bnodes; N + bnodes];
    velvals = [uxv; zeros(numel(bnodes), 1)];

    [~, pin_node] = max(msh.p(:, 1));       % pin pressure at the outflow corner

    geo = struct('x1', x1, 'x2', x2, 'y1', y1, 'y2', y2, ...
                 'xc', (x1 + x2) / 2, 'yc', (y1 + y2) / 2, ...
                 'h0', params.h0, 'Tmax', params.dt * params.Tstep);

    all_cases = define_motion_list(params.dt);
    all_names = cellfun(@(c) c.name, all_cases, 'UniformOutput', false);
    idx = find(strcmp(all_names, case_name), 1);
    if isempty(idx)
        error('schur_make_cfg:unknownCase', ...
              'Unknown motion case "%s". Available: %s.', ...
              case_name, strjoin(all_names, ', '));
    end
    mcase = all_cases{idx}.factory(geo);

    cfg = struct();
    cfg.mesh       = msh;
    cfg.nu         = mcase.nu;
    cfg.h0         = params.h0;
    cfg.velbc_fun  = @(t) struct('dofs', veldofs, 'vals', velvals);  % steady
    cfg.motion_fun = mcase.motion_fun;
    cfg.pin_node   = pin_node;
    cfg.pin_val    = 0;
    cfg.case_name  = case_name;
    cfg.geometry   = 'stokes_immersed_rotor_schur_comp';
    cfg.is_stress  = mcase.is_stress;
    cfg.veldofs    = veldofs;
    cfg.velvals    = velvals;
end
