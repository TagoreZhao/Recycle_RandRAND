function test_upgraded_varvisc_components()
%TEST_UPGRADED_VARVISC_COMPONENTS Promoted geometry/coupling regression.
    here = fileparts(mfilename('fullpath'));
    parent = fullfile(fileparts(here), 'stokes_varvisc_rotor');
    repo = fileparts(fileparts(here));
    addpath(repo, parent, here);

    names = ["current_channel_ar4", "mixer_circle_four_blade"];
    expected_rows = [6, 18];
    for k = 1:numel(names)
        [cfg, descriptor] = varvisc_build_upgraded_case(names(k), 0.20, 0.02, 1.2);
        mot = cfg.motion_fun(0.02);
        [C, g, nC] = cfg.coupling_fun(cfg.mesh, cfg.mesh.N, mot.X, mot.V);
        assert(nC == expected_rows(k));
        assert(isequal(size(C), [nC, 2 * cfg.mesh.N]));
        assert(numel(g) == nC && sprank(C) == nC);
        constant_velocity = ones(2 * cfg.mesh.N, 1);
        assert(norm(C * constant_velocity - ones(nC, 1), inf) < 1e-12);
        assert(descriptor.marker_count * 2 == nC);
    end

    [cfg, ~] = varvisc_build_upgraded_case( ...
        "mixer_circle_four_blade", 0.20, 0.02, 1.2);
    caught = false;
    try
        varvisc_assemble_regularized_coupling( ...
            cfg.mesh, cfg.mesh.N, [1.5, 0], [0, 0], 0.12);
    catch err
        caught = strcmp(err.identifier, ...
            'varvisc_assemble_regularized_coupling:supportOutside');
    end
    assert(caught, 'Out-of-domain regularized support was not rejected.');

    % The shared engine must retain its original point-coupling fallback when
    % no custom coupling assembler is supplied.
    [point_cfg, ~] = varvisc_build_upgraded_case( ...
        "current_channel_ar4", 0.20, 0.02, 1.2);
    point_cfg = rmfield(point_cfg, 'coupling_fun');
    point_params = struct('dt', 0.02, 'Tstep', 2, ...
        'SOLVER_TOL', 1e-8, 'SOLVER_MAXIT', 500, ...
        'solvers', {{struct('key', 'minres_unprec', ...
            'label', 'MINRES (unpreconditioned)', 'build', @(pc) [])}});
    point_stats = src.stokes.solve_stokes_varvisc(point_cfg, point_params, '');
    assert(point_stats.nC(1) == 6 && ...
        isfinite(point_stats.solver_its.minres_unprec(1)));
    fprintf('test_upgraded_varvisc_components: PASS\n');
end
