function Omega = sjlt(num_rows, num_cols, nnz_per_col)
%SJLT  Sparse Johnson-Lindenstrauss Transform.
%   Omega = SJLT(num_rows, num_cols, nnz_per_col) generates a sparse
%   random +-1/sqrt(s) matrix with nnz_per_col nonzeros per column.
%   When num_cols >= num_rows, builds directly; otherwise transposes.

    if num_cols >= num_rows
        rows = double.empty;
        nnz_per_col = min(num_cols, nnz_per_col);
        bad_size = num_rows < nnz_per_col;

        for i = 1:num_cols
            if bad_size
                row = randi(num_rows, nnz_per_col, 1);
            else
                row = randperm(num_rows, nnz_per_col);
            end
            rows = cat(2, rows, row);
        end
        cols = repelem(1:num_cols, nnz_per_col);
        vals = ones(1, num_cols * nnz_per_col);
        vals(rand(1, num_cols * nnz_per_col) <= 0.5) = -1;
        vals = vals / sqrt(cast(nnz_per_col, 'double'));
        Omega = sparse(rows, cols, vals);
    else
        Omega = src.precond.sjlt(num_cols, num_rows, nnz_per_col);
        Omega = Omega';
    end
end
