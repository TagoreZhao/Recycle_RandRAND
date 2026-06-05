function [Papply, E, decE] = deflation_P_apply(V, A, tau, output_type, RAND_EIGS)
    if nargin < 3 || isempty(tau), tau = 1; end
    if tau <= 0, error('tau must be positive.'); end
    if ~(isnumeric(A) || isa(A,'function_handle'))
        error('A must be a numeric matrix or a function handle Afun(X)=A*X.');
    end
    if nargin < 4 || isempty(output_type), output_type = 'handle'; end
    if ~ismember(output_type, {'handle', 'matrix'})
        error('output_type must be ''handle'' or ''matrix''.');
    end
    if nargin < 5 || isempty(RAND_EIGS), RAND_EIGS = 1; end



    if RAND_EIGS == 2
        % Build coarse matrix E = V' A V
        AV = apply_A(A, V.V_fun(eye(V.l,V.l)));
        E  = (V.V_fun_t(AV));
        E  = (E + E')/2;

        % Cholesky on coarse matrix
        [R, flag] = chol(E, 'upper');
        if flag ~= 0
            error('Coarse matrix V''AV is not numerically SPD (chol flag=%d).', flag);
        end

        Z.V_fun  = @(x) V.V_fun(R\x);
        Z.V_fun_t  = @(x) (V.V_fun_t(x)'/R)';

        decE = struct();
        decE.E  = E;
        decE.R  = R;
        decE.Z  = Z;
        
        % Apply P: P = (I - VV') + \tau V(V'AV)^{-1}V'
        % Apply P: PX = X - V(V'X) + tau * Z(Z'X)
        Papply = @(X) X - V.V_fun(V.V_fun_t(X)) + tau * Z.V_fun(Z.V_fun_t(X));
    else
        % Build coarse matrix E = V' A V
        AV = apply_A(A, V);
        E  = (V' * AV);
        E  = (E + E')/2;

        % Cholesky on coarse matrix
        [R, flag] = chol(E, 'upper');
        if flag ~= 0
           error('Coarse matrix V''AV is not numerically SPD (chol flag=%d).', flag);
        end

        % "Move inverse inside" by absorbing R^{-1} into the basis:
        % Z := V / R  so that  V E^{-1} V' = Z Z'
        Z  = V / R;

        decE = struct();
        decE.E  = E;
        decE.R  = R;
        decE.Z  = Z;

        % Apply P: PX = X - V(V'X) + tau * Z(Z'X)
        if strcmp(output_type, 'handle')
            Papply = @(X) X - V*(V'*X) + tau * Z*(Z'*X);
        else
            n = size(V, 1);
            Papply = eye(n) - V*V' + tau * (Z*Z');
        end
    end
end

function AX = apply_A(A, X)
    if isnumeric(A)
        AX = A * X;
        return;
    end
    try
        AX = A(X);
    catch
        [n,m] = size(X);
        AX = zeros(n,m, class(X));
        for j = 1:m
            AX(:,j) = A(X(:,j));
        end
    end
end
