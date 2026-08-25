function blk = varvisc_schur_assemble_blocks(ctx, tcur, u_prev)
%VARVISC_SCHUR_ASSEMBLE_BLOCKS  Assemble post-BC sparse Schur blocks.
%   BLK contains the sparse blocks of
%
%       [ A   Gt ] [u] = [b1]
%       [ G  -D  ] [y]   [b2]
%
%   without materializing the full KKT matrix.  G and Gt are both stored so
%   repeated operator applications never transpose a large sparse matrix.

    import src.stokes.*

    N = ctx.N;
    nP = ctx.nP;

    nu_e = ctx.nu_fun(ctx.msh.cent(:,1), ctx.msh.cent(:,2), tcur);
    K1nu = assemble_visc_stiffness(ctx.msh, nu_e);
    ZN = sparse(N,N);
    A = ctx.Mdt + [K1nu, ZN; ZN, K1nu];
    if strcmp(ctx.bp_mode,'scalar')
        Lp = (ctx.h0^2/(12*min(nu_e)))*ctx.blk.L;
    else
        Lp = assemble_visc_stiffness(ctx.msh,ctx.h0^2./(12*nu_e));
    end

    mot = ctx.motion_fun(tcur);
    [C,gvec,nC] = assemble_coupling(ctx.TR,N,mot.X,mot.V);
    Ct = C';
    Gt = [ctx.BdivT, Ct];
    G = [ctx.Bdiv; C];
    D = blkdiag(Lp,sparse(nC,nC));

    b1 = ctx.Mdt*u_prev;
    if ctx.has_force
        b1 = b1 + ctx.blk.M2*ctx.fnod_fun(tcur);
    end
    b2 = [zeros(nP,1); gvec];

    bc = ctx.velbc_fun(tcur);
    dofs = bc.dofs(:);
    vals = bc.vals(:);
    if ~isempty(dofs)
        b1 = b1 - A(:,dofs)*vals;
        b2 = b2 - G(:,dofs)*vals;
        b1(dofs) = vals;

        A(dofs,:) = 0;
        A(:,dofs) = 0;
        A(dofs,dofs) = speye(numel(dofs));
        Gt(dofs,:) = 0;
        G(:,dofs) = 0;
    end

    pin = ctx.pin_node;
    pinval = ctx.pin_val;
    b1 = b1 - Gt(:,pin)*pinval;
    b2 = b2 + D(:,pin)*pinval;
    b2(pin) = pinval;
    Gt(:,pin) = 0;
    G(pin,:) = 0;
    D(pin,:) = 0;
    D(:,pin) = 0;
    D(pin,pin) = -1;

    blk = struct('A',A,'Gt',Gt,'G',G,'D',D,'b1',b1,'b2',b2, ...
        'C',C,'gvec',gvec,'nC',nC,'nu_e',nu_e);
end
