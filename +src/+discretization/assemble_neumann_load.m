function fN = assemble_neumann_load(p, bdry_edges, neu_mask, gN)
%ASSEMBLE_NEUMANN_LOAD  Assemble Neumann boundary load vector.
%   FN = ASSEMBLE_NEUMANN_LOAD(P, BDRY_EDGES, NEU_MASK, GN)
%
%   Assembles the load vector arising from Neumann BCs using the 1D P1
%   edge mass matrix: M_e = (|e|/6) * [2 1; 1 2].
%
%   Inputs:
%     p          - Nx2 node coordinates
%     bdry_edges - Ex2 boundary edge node pairs (from extract_boundary_edges)
%     neu_mask   - Nx1 logical mask: true at Neumann boundary nodes
%     gN         - Nx1 vector of g_N values at each node (only Neumann entries used)
%
%   Output:
%     fN - Nx1 load vector: (fN)_i = sum_e [M_e^(1D)] * [gN(j); gN(k)]
%          where sum runs over Neumann edges containing node i.

    N = size(p, 1);
    fN = zeros(N, 1);
    for e = 1:size(bdry_edges, 1)
        j = bdry_edges(e, 1);
        k = bdry_edges(e, 2);
        if ~neu_mask(j) || ~neu_mask(k), continue; end
        len_e = norm(p(j,:) - p(k,:));
        % 1D P1 edge mass matrix: (|e|/6) * [2 1; 1 2]
        Me = (len_e / 6) * [2 1; 1 2];
        local_load = Me * [gN(j); gN(k)];
        fN(j) = fN(j) + local_load(1);
        fN(k) = fN(k) + local_load(2);
    end
end
