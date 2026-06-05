function mesh = build_sphere_mesh(h0, visualize_mesh, method)
%BUILD_SPHERE_MESH  Triangular surface mesh of the unit sphere + FEM struct.
%   MESH = BUILD_SPHERE_MESH(H0, VISUALIZE_MESH, METHOD)
%
%   Triangulates the surface of the unit sphere and assembles mass and
%   Laplace-Beltrami stiffness matrices via assemble_fem_struct_surface.
%   The sphere is a closed manifold: Bdry = [], IN = (1:N)'.
%
%   Two mesh-generation backends are available, selected by METHOD:
%     'pdetoolbox' (default) - mesh the solid ball with PDE Toolbox
%                              (multisphere + generateMesh) and extract the
%                              surface triangulation via freeBoundary. Fast.
%                              Requires Partial Differential Equation Toolbox.
%     'distmesh'             - the DistMesh surface generator
%                              (src.discretization.distmeshsurface). No
%                              toolbox dependency, but slower.
%   Both backends produce (p, t) and share the same surface FEM assembly.
%
%   Inputs:
%     h0             - target edge length (default 0.15)
%     visualize_mesh - show the DistMesh live plot (default false; ignored
%                      by the 'pdetoolbox' backend)
%     method         - 'pdetoolbox' (default) or 'distmesh'
%
%   Output:
%     mesh - struct from assemble_fem_struct_surface (fields .p, .t, .N, .M,
%            .D_II, .Vunit, .idxII, .I_II, .J_II, .cent, ...).

    import src.discretization.*

    if nargin < 1 || isempty(h0),             h0 = 0.15;             end
    if nargin < 2 || isempty(visualize_mesh), visualize_mesh = false; end
    if nargin < 3 || isempty(method),         method = 'pdetoolbox'; end

    switch lower(method)
        case 'pdetoolbox'
            [p, t] = sphere_surface_pdetoolbox(h0);
        case 'distmesh'
            [p, t] = sphere_surface_distmesh(h0, visualize_mesh);
        otherwise
            error('build_sphere_mesh:badMethod', ...
                  'method must be ''pdetoolbox'' or ''distmesh'', got ''%s''.', ...
                  method);
    end

    % Closed surface: all nodes are interior
    N    = size(p, 1);
    Bdry = [];
    IN   = (1:N)';

    mesh = assemble_fem_struct_surface(p, t, Bdry, IN);

    fprintf('  Sphere mesh (%s): N=%d, M=%d, numIN=%d, numB=%d\n', ...
        lower(method), mesh.N, mesh.M, mesh.numIN, mesh.numB);
end


function [p, t] = sphere_surface_pdetoolbox(h0)
%SPHERE_SURFACE_PDETOOLBOX  Unit-sphere surface tris via PDE Toolbox.
%   Meshes the solid unit ball, then extracts its boundary triangulation.
%   Handles both the fegeometry workflow and the createpde-model workflow,
%   depending on what multisphere returns in this MATLAB release.
    gm = multisphere(1);                       % unit-radius solid sphere geometry
    if isa(gm, 'fegeometry')
        fe    = generateMesh(gm, 'Hmax', h0, 'GeometricOrder', 'linear');
        femsh = fe.Mesh;
    else
        model = createpde;
        model.Geometry = gm;
        generateMesh(model, 'Hmax', h0, 'GeometricOrder', 'linear');
        femsh = model.Mesh;
    end

    nodes = double(femsh.Nodes');              % N_vol x 3
    elems = double(femsh.Elements');           % M_vol x 4 (linear tets)

    % Boundary triangulation: F indexes into P (already compacted to the
    % surface nodes). Orientation is irrelevant to mass/cotangent stiffness.
    TR     = triangulation(elems, nodes);
    [t, p] = freeBoundary(TR);

    % Project boundary nodes exactly onto the unit sphere (no-op when already
    % on the surface; guards against any geometric tolerance).
    p = p ./ sqrt(sum(p.^2, 2));
end


function [p, t] = sphere_surface_distmesh(h0, visualize_mesh)
%SPHERE_SURFACE_DISTMESH  Unit-sphere surface tris via DistMesh.
    import src.discretization.*

    if ~visualize_mesh
        prev_vis = get(0, 'DefaultFigureVisible');
        set(0, 'DefaultFigureVisible', 'off');
    end

    fd     = @(p) dsphere(p, 0, 0, 0, 1);
    [p, t] = distmeshsurface(fd, @huniform, h0, 1.1 * [-1,-1,-1; 1,1,1]);

    if ~visualize_mesh
        set(0, 'DefaultFigureVisible', prev_vis);
    end
end
