function vals = interp_tri_values(pf, tf, vf, qpts, fillValue)
%INTERP_TRI_VALUES  P1 (piecewise-linear) interpolation on a triangular mesh.
%
%   vals = interp_tri_values(pf, tf, vf, qpts)
%   vals = interp_tri_values(pf, tf, vf, qpts, fillValue)
%
% Inputs
%   pf        : (N x 2) node coordinates
%   tf        : (T x 3) triangle connectivity (indices into pf)
%   vf        : (N x 1) or (N x C) nodal values
%   qpts      : (Q x 2) query points
%   fillValue : scalar to use for points outside mesh (default: NaN)
%
% Output
%   vals      : (Q x 1) or (Q x C) interpolated values at qpts

    if nargin < 5 || isempty(fillValue)
        fillValue = NaN;
    end

    % Basic checks (lightweight)
    if size(pf,2) ~= 2 || size(qpts,2) ~= 2
        error('pf and qpts must be N×2 and Q×2 arrays of (x,y) coordinates.');
    end
    if size(tf,2) ~= 3
        error('tf must be T×3 triangle connectivity.');
    end
    if size(vf,1) ~= size(pf,1)
        error('vf must have the same number of rows as pf (nodal values).');
    end

    TR = triangulation(tf, pf);
    [ti, bc] = pointLocation(TR, qpts);   % ti: Q×1 triangle IDs, bc: Q×3 barycentric

    Q = size(qpts,1);
    C = size(vf,2);

    vals = fillValue * ones(Q, C);

    valid = ~isnan(ti);
    if ~any(valid)
        return;
    end

    % Connectivity for the containing triangles of valid points
    triNodes = TR.ConnectivityList;                 % T×3
    nodes = triNodes(ti(valid), :);                 % Qv×3

    % Gather nodal values for each query point's containing triangle
    % For vf (N×C), this becomes (Qv×3×C) after reshape.
    vnodes = vf(nodes(:), :);                       % (Qv*3)×C
    vnodes = reshape(vnodes, [], 3, C);             % Qv×3×C

    w = bc(valid, :);                               % Qv×3
    w = reshape(w, [], 3, 1);                       % Qv×3×1 for broadcasting

    % Weighted sum over the triangle's 3 vertices
    vals(valid, :) = squeeze(sum(w .* vnodes, 2));  % Qv×C
end
