function [C, gvec, nC] = varvisc_assemble_regularized_coupling( ...
        msh, N, X, Vpts, radius)
%VARVISC_ASSEMBLE_REGULARIZED_COUPLING Finite-radius velocity constraints.
% Each marker constrains a deterministic disk average evaluated with four
% equal-area radial rings and sixteen angular samples per ring.

    if radius <= 0 || ~isfinite(radius)
        error('varvisc_assemble_regularized_coupling:badRadius', ...
            'The coupling radius must be finite and positive.');
    end
    marker_count = size(X, 1);
    if size(X, 2) ~= 2 || ~isequal(size(Vpts), size(X))
        error('varvisc_assemble_regularized_coupling:badMarkerData', ...
            'X and Vpts must both be K-by-2.');
    end

    TR = triangulation(msh.t, msh.p);
    radial_count = 4;
    angular_count = 16;
    angle = (0:angular_count-1)' * (2 * pi / angular_count);
    offsets = zeros(radial_count * angular_count, 2);
    for ring = 1:radial_count
        radial = radius * sqrt((ring - 0.5) / radial_count);
        index = (ring - 1) * angular_count + (1:angular_count);
        offsets(index, :) = [radial * cos(angle), radial * sin(angle)];
    end

    rows = cell(marker_count, 1);
    cols = cell(marker_count, 1);
    values = cell(marker_count, 1);
    for k = 1:marker_count
        query = X(k, :) + offsets;
        ti = pointLocation(TR, query);
        valid = ~isnan(ti);
        if nnz(valid) < 0.95 * size(query, 1)
            error('varvisc_assemble_regularized_coupling:supportOutside', ...
                'Marker %d has too much coupling support outside the mesh.', k);
        end
        query = query(valid, :);
        tiv = ti(valid);
        nodes = TR.ConnectivityList(tiv, :);
        bary = cartesianToBarycentric(TR, tiv, query) / nnz(valid);
        rows{k} = repmat(k, numel(nodes), 1);
        cols{k} = nodes(:);
        values{k} = bary(:);
    end

    row = vertcat(rows{:});
    col = vertcat(cols{:});
    value = vertcat(values{:});
    Cx = sparse(row, col, value, marker_count, 2 * N);
    Cy = sparse(row, col + N, value, marker_count, 2 * N);
    C = [Cx; Cy];
    gvec = [Vpts(:, 1); Vpts(:, 2)];
    nC = 2 * marker_count;
end
