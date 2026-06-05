function Cfun = make_Ctau_prec(V, Minv, A, tau)
%MAKE_CTAU_PREC  Return handle y = C_tau^{-1} x for use in pcg
%
%   Q = V * (V' * A * V)^{-1} * V'
%   C_tau^{-1} = tau*Q + (I - Q*A) * M^{-1} * (I - A*Q)
%
% Inputs
%   V    : n-by-k coarse basis matrix
%   Minv : function handle, z = Minv(r), applying M^{-1} to a vector r
%   A    : n-by-n matrix or function handle, y = A(x)
%   tau  : scalar
%
% Output
%   Cfun : function handle, y = Cfun(x), suitable for pcg(..., Cfun)

    [n, k] = size(V);

    % Build AV once
    if isa(A, 'function_handle')
        % Try block application first
        try
            AV = A(V);
            if ~isequal(size(AV), [n, k])
                error('A(V) returned wrong size.');
            end
        catch
            % Fall back to column-by-column
            AV = zeros(n, k, class(V));
            for j = 1:k
                AV(:, j) = A(V(:, j));
            end
        end
    else
        AV = A * V;
    end

    % Small coarse matrix G = V'AV
    G = V' * AV;

    % For PCG/SPD, enforce symmetry numerically
    G = 0.5 * (G + G');

    % Factor once and reuse
    Gdec = decomposition(G, 'chol');

    VT  = V';
    AVT = AV';

    Cfun = @apply_Ctau_inv;

    function y = apply_Ctau_inv(x)
        % s = (V'AV)^{-1} V'x
        s = Gdec \ (VT * x);

        % r = (I - AQ)x = x - AV s
        r = x - AV * s;

        % z = M^{-1} r
        z = Minv(r);

        % t = (V'AV)^{-1} (AV)' z = (V'AV)^{-1} V'Az
        t = Gdec \ (AVT * z);

        % y = tau*Qx + (I - QA)z
        y = z + V * (tau * s - t);
    end
end