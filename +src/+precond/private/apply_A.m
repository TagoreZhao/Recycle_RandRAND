function AX = apply_A(A, X)
%APPLY_A  Apply A to X whether A is a matrix or a function handle.
%   Shared package-private helper for +src/+precond.  If A is numeric it
%   returns A*X; if A is a function handle Afun(X)=A*X it tries the block
%   apply A(X) and falls back to a column-by-column loop.
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
