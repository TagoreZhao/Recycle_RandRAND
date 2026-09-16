function [Papply,spectralApply,info,details] = varvisc_schur_two_stage_augmentation( ...
        Sapply,V,residual,m,tau1,tau2,needSpectrum)
%VARVISC_SCHUR_TWO_STAGE_AUGMENTATION Share [V,W] in both deflation stages.
% RESIDUAL is b-S*x0 in the same reduced coordinates as S and V. W is fresh
% projected Arnoldi on S, never on the stage-one transformed operator.
    if nargin<7, needSpectrum=false; end
    timer=tic;
    [W,arn]=varvisc_build_projected_arnoldi(Sapply,V,residual,m);
    arnoldi_s=toc(timer); if m==0, arnoldi_s=0; end
    Z=[V,W];
    [P1half,E1,dec1]=src.precond.deflation_Psqrt_apply(Z,Sapply,tau1,'handle');
    S1apply=@(X) P1half(Sapply(P1half(X)));
    [P2apply,E2,dec2]=src.precond.deflation_P_apply(Z,S1apply,tau2,'handle',0);
    Papply=@(X) P1half(P2apply(P1half(X)));
    spectralApply=[];
    if needSpectrum
        P2half=src.precond.deflation_Psqrt_apply(Z,S1apply,tau2,'handle');
        spectralApply=@(X) P2half(S1apply(P2half(X)));
    end
    health=varvisc_arnoldi_metrics(V,W,arn);
    info=struct('base_rank',size(V,2),'requested_m',m,'aug_rank',size(W,2), ...
        'total_rank',size(Z,2),'arnoldi_status',arn.status,'start_norm',arn.start_norm, ...
        'arnoldi_columns',arn.operator_columns,'residual_columns',0,'arnoldi_s',arnoldi_s, ...
        'vw_orth',health.vw_orth,'ww_orth',health.ww_orth,'arnoldi_relerr',health.arnoldi_relation, ...
        'coarse1_min_eig',min(dec1.d),'coarse2_min_diag',min(diag(dec2.R)));
    if nargout>3
        details=struct('V',V,'W',W,'Z',Z,'E1',E1,'E2',E2,'arnoldi',arn, ...
            'P1half',P1half,'S1apply',S1apply,'P2apply',P2apply);
    end
end
