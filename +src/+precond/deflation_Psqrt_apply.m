function [Psqrt_apply, E, decE] = deflation_Psqrt_apply(V, A, tau, output_type)
    % Construct an operator handle for P^{1/2} where
    %   P = (I - V V') + tau * V * (V' A V)^{-1} * V'
    % hence
    %   P^{1/2} = (I - V V') + sqrt(tau) * V * (V' A V)^{-1/2} * V'
    %
    % Supported forms for V:
    %   - Numeric matrix V with orthonormal columns (V'V = I, tau > 0).
    %   - Vstruct with fields V_fun, V_fun_t, l, n: a function-handle form
    %     where V_fun(X) returns V*X and V_fun_t(X) returns V'*X.
    %     Intended for sparse-embedding / randomized-Nystrom bases that
    %     never materialize V explicitly.  V'V ~= I is assumed up to
    %     sketching error, matching the contract used by deflation_P_apply
    %     for the same input form.

    if nargin < 3 || isempty(tau), tau = 1; end
    if nargin < 4 || isempty(output_type), output_type = 'handle'; end
    if tau <= 0, error('tau must be positive.'); end
    if ~ismember(output_type, {'handle', 'matrix'})
        error('output_type must be ''handle'' or ''matrix''.');
    end
    if ~(isnumeric(A) || isa(A,'function_handle'))
        error('A must be a numeric matrix or a function handle Afun(X)=A*X.');
    end

    % ---- Detect V form, build V_apply / Vt_apply, and materialize Vdense
    %      ONLY for the n x l AV product that feeds E.
    if isstruct(V)
        required = {'V_fun','V_fun_t','l','n'};
        for f = required
            if ~isfield(V, f{1})
                error('Vstruct must contain field ''%s''.', f{1});
            end
        end
        Vdense    = V.V_fun(eye(V.l));   % one-time materialize (n x l)
        V_apply   = V.V_fun;
        Vt_apply  = V.V_fun_t;
        n_rows    = V.n;
        is_struct = true;
    elseif isnumeric(V)
        Vdense    = V;
        V_apply   = @(X) V * X;
        Vt_apply  = @(X) V' * X;
        n_rows    = size(V, 1);
        is_struct = false;
    else
        error('V must be a numeric matrix or a Vstruct.');
    end

    % Build coarse matrix E = V' A V (k-by-k SPD)
    AV = apply_A(A, Vdense);
    E  = Vt_apply(AV);
    E  = (E + E')/2;
    E  = full(E);    % eig() with [U,D] outputs rejects sparse input

    % Eigendecomposition of E (small k-by-k SPD):
    % E = U * diag(d) * U'
    [U, D] = eig(E);
    d = real(diag(D));

    if any(d <= 0)
        error('Coarse matrix V''AV is not numerically SPD (nonpositive eig).');
    end

    inv_sqrt_d = 1 ./ sqrt(d);

    decE            = struct();
    decE.E          = E;
    decE.U          = U;
    decE.Ut         = U';
    decE.d          = d;
    decE.inv_sqrt_d = inv_sqrt_d;
    decE.V_apply    = V_apply;
    decE.Vt_apply   = Vt_apply;

    % Helper: apply E^{-1/2} to a k-by-m matrix Y
    apply_E_inv_half = @(Y) U * (inv_sqrt_d .* (decE.Ut * Y));

    % Apply P^{1/2}:
    % Psqrt X = X - V(V'X) + sqrt(tau) * V * E^{-1/2} * (V'X)
    sqtau = sqrt(tau);
    if strcmp(output_type, 'handle')
        Psqrt_apply = @(X) X ...
            - V_apply(Vt_apply(X)) ...
            + sqtau * V_apply(apply_E_inv_half(Vt_apply(X)));
    else
        if is_struct
            error(['output_type=''matrix'' is not supported for Vstruct input; ', ...
                   'pass a numeric V, or request the handle form.']);
        end
        E_inv_half  = U * diag(inv_sqrt_d) * U';
        Psqrt_apply = eye(n_rows) - V*V' + sqtau * V * E_inv_half * V';
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
