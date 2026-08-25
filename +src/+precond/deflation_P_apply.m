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

        Rt = R';
        Z.V_fun  = @(x) V.V_fun(R\x);
        Z.V_fun_t  = @(x) Rt\V.V_fun_t(x);

        decE = struct();
        decE.E  = E;
        decE.R  = R;
        decE.Z  = Z;
        
        % Apply P: P = (I - VV') + \tau V(V'AV)^{-1}V'
        % Apply P: PX = X - V(V'X) + tau * Z(Z'X)
        Papply = @(X) local_struct_apply( ...
            X,V.V_fun,V.V_fun_t,R,Rt,tau);
    else
        % Build coarse matrix E = V' A V
        AV = apply_A(A, V);
        Vt = V';
        E  = Vt*AV;
        E  = (E + E')/2;

        % Cholesky on coarse matrix
        [R, flag] = chol(E, 'upper');
        if flag ~= 0
           error('Coarse matrix V''AV is not numerically SPD (chol flag=%d).', flag);
        end

        % "Move inverse inside" by absorbing R^{-1} into the basis:
        % Z := V / R  so that  V E^{-1} V' = Z Z'
        Z  = V / R;
        Zt = Z';

        decE = struct();
        decE.E  = E;
        decE.R  = R;
        decE.Z  = Z;

        % Apply P: PX = X - V(V'X) + tau * Z(Z'X)
        if strcmp(output_type, 'handle')
            Papply = @(X) local_numeric_apply(X,V,Vt,Z,Zt,tau);
        else
            n = size(V, 1);
            Papply = eye(n) - V*Vt + tau*(Z*Zt);
        end
    end
end

function Y = local_struct_apply(X,Vfun,Vtfun,R,Rt,tau)
    coefficients = Vtfun(X);
    coarseSolve = R\(Rt\coefficients);
    Y = X - Vfun(coefficients) + tau*Vfun(coarseSolve);
end

function Y = local_numeric_apply(X,V,Vt,Z,Zt,tau)
    Y = X - V*(Vt*X) + tau*Z*(Zt*X);
end
