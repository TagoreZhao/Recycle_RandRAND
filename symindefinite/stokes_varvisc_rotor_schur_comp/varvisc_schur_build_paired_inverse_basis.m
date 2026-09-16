function [V,info] = varvisc_schur_build_paired_inverse_basis(R,Omega,q)
%VARVISC_SCHUR_BUILD_PAIRED_INVERSE_BASIS orth(S^-q Omega), S = R*R'.
% Same power count and final orth convention as the native Schur sketch.
    validateattributes(q,{'numeric'},{'scalar','integer','nonnegative'});
    assert(size(R,1)==size(R,2) && size(Omega,1)==size(R,1));
    assert(size(Omega,2)<=size(R,1),'Sketch width exceeds reduced dimension.');
    Rt=R'; calls=0; columns=0;
    Y=src.precond.subspace_iter_plain(@inverse_apply,Omega,q);
    V=orth(real(Y));
    assert(~isempty(V),'Inverse sketch has zero numerical rank.');
    info=struct('inverse_applications',calls,'inverse_rhs_columns',columns, ...
        'basis_rank',size(V,2),'rank_drop',size(Omega,2)-size(V,2));
    function Z=inverse_apply(X)
        calls=calls+1; columns=columns+size(X,2);
        Z=Rt\(R\X);
    end
end
