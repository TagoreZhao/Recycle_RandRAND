function st = varvisc_schur_step_operator(ctx, tcur, u_prev)
%VARVISC_SCHUR_STEP_OPERATOR  Form the complete dense SPD Schur complement.
%   A_n and D_n both change with viscosity, so every step rebuilds chol(A_n)
%   and forms S_n = D_n + G_n A_n^{-1} G_n'.

    [K,b,C,gvec,nC,nu_e] = varvisc_schur_assemble_kkt(ctx,tcur,u_prev);
    nU = ctx.nU; nP = ctx.nP;
    nSfull = nP + nC;
    A = (K(1:nU,1:nU) + K(1:nU,1:nU)') / 2;
    Gt = K(1:nU,nU+1:end);
    D = -K(nU+1:end,nU+1:end);
    b1 = b(1:nU);
    b2 = b(nU+1:end);

    dA = decomposition(A, 'chol');
    Y = dA \ full(Gt);
    Sfull = full(D) + Gt' * Y;
    Sfull = (Sfull + Sfull') / 2;
    rhsFull = Gt' * (dA \ b1) - b2;

    keep = true(nSfull,1);
    keep(ctx.pin_node) = false;
    st.S = Sfull(keep,keep);
    st.rhs_S = rhsFull(keep);
    st.keep = keep;
    st.recover = @(ykeep) local_recover(ykeep,keep,ctx.pin_node,ctx.pin_val, ...
        nSfull,b1,Gt,dA);
    st.K = K; st.b = b; st.C = C; st.gvec = gvec;
    st.nC = nC; st.nS = nSfull - 1; st.pin_node = ctx.pin_node;
    st.nu_e = nu_e; st.A_bc = A; st.D = D; st.Gt = Gt;
end

function x = local_recover(ykeep,keep,pin,pinval,nS,b1,Gt,dA)
    y = zeros(nS,1);
    y(keep) = ykeep;
    y(pin) = pinval;
    u = dA \ (b1 - Gt*y);
    x = [u; y];
end
