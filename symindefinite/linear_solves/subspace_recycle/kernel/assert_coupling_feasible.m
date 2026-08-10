function touched = assert_coupling_feasible(C, veldofs, n, case_name)
%ASSERT_COUPLING_FEASIBLE  Refuse a coupling block with more rows than DOFs to constrain.
%   TOUCHED = ASSERT_COUPLING_FEASIBLE(C, VELDOFS, N, CASE_NAME)
%
%   C is the nC-by-2N distributed-Lagrange-multiplier block at step N, VELDOFS the
%   velocity DOFs eliminated by apply_dirichlet_sym.  Returns the number of FREE
%   velocity DOFs the constraint rows actually touch, and errors when
%
%       nC > touched
%
%   because then rank(C) <= touched < nC, so C'y = 0 has a nonzero solution y and
%   [0; 0; y] lies in the null space of the KKT matrix
%
%       K = [Avel B' C'; B -eps*L 0; C 0 0].
%
%   K is therefore EXACTLY singular -- not merely ill conditioned -- at every step,
%   and K\b is meaningless.  This is a geometry error: the Lagrange points are
%   sampled more finely than the velocity nodes they interpolate.
%
%   WHY THE DIRICHLET COLUMNS ARE EXCLUDED.  apply_dirichlet_sym zeroes both the
%   rows and the COLUMNS of VELDOFS, so a constraint supported only on prescribed
%   velocity nodes contributes nothing to the rank of the eliminated system.
%   Counting those columns would overstate the budget and let a singular geometry
%   through -- which is precisely the case of an immersed body touching a wall.
%
%   WHAT THIS DOES NOT CATCH.  touched >= nC does NOT imply full row rank: two
%   coincident points, or points collinear inside a single element, still drop rank
%   while passing this count.  The test is a SUFFICIENT condition for singularity,
%   chosen because it is O(nnz(C)) ~ 3*nC and can therefore run on every step
%   unconditionally rather than under opts.verify.  build_stokes_sequence's
%   non-finite check on K\b is the backstop for the cases this misses.
%
%   IT DELIBERATELY DOES NOT FIRE ON MERE ILL CONDITIONING.  A near-degenerate
%   coupling is a legitimate object of study here (it is what drives kappa(Cap)),
%   so this gate triggers only on the exactly-singular case that no solver can
%   return an answer for.
%
%   See also: build_stokes_sequence, assemble_coupling, apply_dirichlet_sym.

    nC = size(C, 1);

    free = true(1, size(C, 2));
    free(veldofs) = false;
    touched = nnz(any(C(:, free) ~= 0, 1));

    if nC > touched
        error('assert_coupling_feasible:overConstrained', ...
              ['step %d (%s): %d Lagrange constraint rows are supported on only ' ...
               '%d free velocity DOFs, so rank(C) <= %d < %d and C''y = 0 has a ' ...
               'nonzero solution y.  The KKT matrix is then EXACTLY singular -- ' ...
               '[0;0;y] is in its null space -- and K\\b is meaningless.  This is ' ...
               'a geometry error, not an ill-conditioning to be regularized and ' ...
               'not a tolerance to be raised: the Lagrange points must be sparser ' ...
               'than the velocity nodes they interpolate.  Widen the point spacing ' ...
               'in define_motion_list.m (disk_sample / make_bar_rotating) or refine ' ...
               'the mesh.'], n, case_name, nC, touched, touched, nC);
    end
end
