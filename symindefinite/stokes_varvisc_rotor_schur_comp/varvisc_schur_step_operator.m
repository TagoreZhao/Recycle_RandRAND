function st = varvisc_schur_step_operator(ctx, tcur, u_prev)
%VARVISC_SCHUR_STEP_OPERATOR  Build the reduced SPD Schur operator.
%   A_n and D_n both change with viscosity, so every step rebuilds chol(A_n).
%   The Schur complement is exposed as a function handle and is materialized
%   only when ST.to_dense() is explicitly requested.  The full sparse KKT
%   matrix is likewise available only through ST.materialize_kkt().

    blk = varvisc_schur_assemble_blocks(ctx,tcur,u_prev);
    nSfull = ctx.nP + blk.nC;

    dA = decomposition(blk.A,'chol');
    keep = true(nSfull,1);
    keep(ctx.pin_node) = false;
    GtReduced = blk.Gt(:,keep);
    GReduced = blk.G(keep,:);
    DReduced = blk.D(keep,keep);
    b2Reduced = blk.b2(keep);

    st.apply = @(X) local_apply(X,DReduced,GReduced,GtReduced,dA);
    st.to_dense = @() local_to_dense(DReduced,GReduced,GtReduced,dA);
    st.materialize_kkt = @() local_materialize_kkt( ...
        blk.A,blk.Gt,blk.G,blk.D,blk.b1,blk.b2);
    st.rhs_S = GReduced*(dA\blk.b1) - b2Reduced;
    st.keep = keep;
    st.recover = @(ykeep) local_recover(ykeep,keep,ctx.pin_node,ctx.pin_val, ...
        nSfull,blk.b1,blk.Gt,dA);
    st.C = blk.C; st.gvec = blk.gvec;
    st.nC = blk.nC; st.nS = nSfull - 1; st.pin_node = ctx.pin_node;
    st.nu_e = blk.nu_e; st.A_bc = blk.A; st.D = blk.D;
    st.Gt = blk.Gt; st.G = blk.G;
    st.D_reduced = DReduced;
    st.Gt_reduced = GtReduced; st.G_reduced = GReduced;
end

function Y = local_apply(X,D,G,Gt,dA)
    if size(X,1) ~= size(Gt,2)
        error('varvisc_schur_step_operator:dimensionMismatch', ...
              'Schur input has %d rows; expected %d.',size(X,1),size(Gt,2));
    end
    Y = D*X + G*(dA\(Gt*X));
end

function S = local_to_dense(D,G,Gt,dA)
    S = full(D) + G*(dA\full(Gt));
    S = (S+S')/2;
end

function [K,b] = local_materialize_kkt(A,Gt,G,D,b1,b2)
    K = [A, Gt; G, -D];
    b = [b1; b2];
end

function x = local_recover(ykeep,keep,pin,pinval,nS,b1,Gt,dA)
    y = zeros(nS,1);
    y(keep) = ykeep;
    y(pin) = pinval;
    u = dA \ (b1 - Gt*y);
    x = [u; y];
end
