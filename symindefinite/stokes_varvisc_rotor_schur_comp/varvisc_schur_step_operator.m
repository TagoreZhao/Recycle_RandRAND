function st = varvisc_schur_step_operator(ctx, tcur, u_prev)
%VARVISC_SCHUR_STEP_OPERATOR  Build the reduced SPD Schur operator.
%   A_n and D_n both change with viscosity, so every step rebuilds chol(A_n).
%   The Schur complement is exposed as a function handle and is materialized
%   only when ST.to_dense() is explicitly requested.

    [K,b,C,gvec,nC,nu_e] = varvisc_schur_assemble_kkt(ctx,tcur,u_prev);
    nU = ctx.nU; nP = ctx.nP;
    nSfull = nP + nC;
    A = (K(1:nU,1:nU) + K(1:nU,1:nU)') / 2;
    Gt = K(1:nU,nU+1:end);
    D = -K(nU+1:end,nU+1:end);
    b1 = b(1:nU);
    b2 = b(nU+1:end);

    dA = decomposition(A, 'chol');
    keep = true(nSfull,1);
    keep(ctx.pin_node) = false;
    GtReduced = Gt(:,keep);
    DReduced = D(keep,keep);
    b2Reduced = b2(keep);

    st.apply = @(X) local_apply(X,DReduced,GtReduced,dA);
    st.to_dense = @() local_to_dense(DReduced,GtReduced,dA);
    st.rhs_S = GtReduced' * (dA \ b1) - b2Reduced;
    st.keep = keep;
    st.recover = @(ykeep) local_recover(ykeep,keep,ctx.pin_node,ctx.pin_val, ...
        nSfull,b1,Gt,dA);
    st.K = K; st.b = b; st.C = C; st.gvec = gvec;
    st.nC = nC; st.nS = nSfull - 1; st.pin_node = ctx.pin_node;
    st.nu_e = nu_e; st.A_bc = A; st.D = D; st.Gt = Gt;
    st.D_reduced = DReduced; st.Gt_reduced = GtReduced;
end

function Y = local_apply(X,D,Gt,dA)
    if size(X,1) ~= size(Gt,2)
        error('varvisc_schur_step_operator:dimensionMismatch', ...
              'Schur input has %d rows; expected %d.',size(X,1),size(Gt,2));
    end
    Y = D*X + Gt'*(dA\(Gt*X));
end

function S = local_to_dense(D,Gt,dA)
    S = full(D) + Gt'*(dA\full(Gt));
    S = (S+S')/2;
end

function x = local_recover(ykeep,keep,pin,pinval,nS,b1,Gt,dA)
    y = zeros(nS,1);
    y(keep) = ykeep;
    y(pin) = pinval;
    u = dA \ (b1 - Gt*y);
    x = [u; y];
end
