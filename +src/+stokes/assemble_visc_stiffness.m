function Kc = assemble_visc_stiffness(msh, coef_e)
%ASSEMBLE_VISC_STIFFNESS  P1 stiffness with an elementwise coefficient.
%   KC = ASSEMBLE_VISC_STIFFNESS(MSH, COEF_E)
%
%   Assembles the N x N P1 stiffness matrix with a per-element scalar
%   coefficient by rescaling the mesh struct's unit-stiffness triplets
%   (msh.Itrip / msh.Jtrip / msh.Vunit from assemble_fem_struct).
%
%   Coefficient-agnostic: serves both the viscosity-scaled velocity
%   stiffness A2(nu_e) (pass nu_e) and the Brezzi-Pitkaranta stabilization
%   block Lp_eps (pass eps_e = h0^2 ./ (12*nu_e)).
%
%   Inputs:
%     msh    - mesh struct from src.discretization.assemble_fem_struct
%     coef_e - M x 1 elementwise coefficient (evaluated at centroids)
%
%   Output:
%     Kc     - N x N sparse SPD stiffness, symmetrized to machine precision
%              (required by MINRES / ichol downstream).

    Kc = sparse(msh.Itrip, msh.Jtrip, msh.Vunit .* repelem(coef_e, 9), msh.N, msh.N);
    Kc = (Kc + Kc') / 2;
end
