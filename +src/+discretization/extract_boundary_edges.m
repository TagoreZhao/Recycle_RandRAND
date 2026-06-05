function bdry_edges = extract_boundary_edges(t)
%EXTRACT_BOUNDARY_EDGES  Find boundary edges from triangle connectivity.
%   BDRY_EDGES = EXTRACT_BOUNDARY_EDGES(T) returns an E x 2 matrix of node
%   index pairs for edges that appear in exactly one triangle.
%
%   Inputs:
%     t - Mx3 triangle connectivity
%
%   Output:
%     bdry_edges - Ex2 matrix of boundary edge node pairs (sorted per row)

    edges = sort([t(:,[1 2]); t(:,[2 3]); t(:,[1 3])], 2);
    [unique_edges, ~, ic] = unique(edges, 'rows');
    counts = accumarray(ic, 1);
    bdry_edges = unique_edges(counts == 1, :);
end
