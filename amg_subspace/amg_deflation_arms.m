function arms = amg_deflation_arms()
%AMG_DEFLATION_ARMS  The preconditioner arms compared in the deflation study.
%
%   arms = amg_deflation_arms() returns a struct array of recipes.  Each entry
%   describes one way of turning an AMG hierarchy (and, for some arms, the
%   basis sketched from it) into a preconditioner, plus the metadata the driver
%   needs to decide whether the arm is applicable and how to plot it.
%
%   The comparison the study exists for is defl_amg vs amg_direct at the SAME
%   AMG config: one and the same V-cycle, used once as the preconditioner and
%   once as the generator of a deflation basis.  The remaining arms bracket
%   that pair -- pcg_plain and pcg_ichol below it, defl_exact above it as the
%   unreachable floor, and ctau_amg / defl_amg_ichol to separate "AMG supplies
%   the subspace" from "AMG supplies the smoothing".
%
%   Fields
%     id           : short identifier, used as the CSV `arm` value
%     label        : legend text
%     needs_V      : true if build() requires a deflation basis
%     needs_sym    : true if the arm applies M itself and therefore needs a
%                    SYMMETRIC V-cycle (preSmooth == postSmooth).  pcg and the
%                    symmetric Lanczos in precond_spectrum are both invalid
%                    otherwise, so the driver marks these arms skipped rather
%                    than reporting numbers from an invalid method.
%     sketch_only  : true if the arm uses ONLY sketch-derived information.
%                    False marks reference curves (defl_exact) that consume
%                    exact eigenvectors and are never a fair competitor.
%     is_reference : true if the arm is drawn as a reference line rather than
%                    compared per AMG config
%     build        : @(ctx, V, tau) -> struct('prec',., 'valid',., 'skip_reason',.)
%                    ctx carries .A .Afun .L .Lt .Mamg .n .M_is_sym
%
%   The returned `prec` is always a plain function handle (or [] for the
%   unpreconditioned arm), which is simultaneously pcg's 5th positional
%   argument and precond_spectrum's Mfun -- one convention, so the measured
%   spectrum is the spectrum of the operator actually solved with.
%
%   See also AMG_SKETCH_BASIS, AMG_SKETCH_TAU, PRECOND_SPECTRUM,
%   RUN_AMG_DEFLATION_VS_PRECOND.

    arms = [ ...
        arm('pcg_plain',  'unpreconditioned',        false, false, true,  true,  @build_plain); ...
        arm('pcg_ichol',  'ichol(A,''nofill'')',     false, false, true,  true,  @build_ichol); ...
        arm('amg_direct', 'AMG as preconditioner',   false, true,  true,  false, @build_amg); ...
        arm('defl_amg',   'deflation, AMG sketch',   true,  false, true,  false, @build_defl); ...
        arm('defl_amg_ichol', 'ichol + AMG sketch coarse', true, false, true, false, @build_defl_ichol); ...
        arm('ctau_amg',   'AMG + deflation',         true,  true,  true,  false, @build_ctau_amg); ...
        arm('defl_exact', 'deflation, exact eigvecs', true, false, false, true,  @build_defl) ...
    ];
end

%% =========================================================================
%% Arm builders
%% =========================================================================
function s = build_plain(~, ~, ~)
%BUILD_PLAIN  No preconditioner: the spectrum of A itself, the reference for
%   both ends of every ratio column.
    s = ok_arm([]);
end

function s = build_ichol(ctx, ~, ~)
%BUILD_ICHOL  The standard cheap baseline, M = (L L')^{-1}.
    L = ctx.L;  Lt = ctx.Lt;
    s = ok_arm(@(r) Lt \ (L \ r));
end

function s = build_amg(ctx, ~, ~)
%BUILD_AMG  One V-cycle as the preconditioner -- the arm to beat.
    s = ok_arm(ctx.Mamg);
end

function s = build_defl(ctx, V, tau)
%BUILD_DEFL  Deflation only: P = (I - VV') + tau V(V'AV)^{-1}V'.
%   No AMG inside the solve at all -- the V-cycle's entire contribution is the
%   subspace it sketched.  Moves span(V) to tau and leaves the rest of the
%   spectrum, lam_max included, exactly where it was.
    s = ok_arm(src.precond.deflation_P_apply(V, ctx.Afun, tau, 'handle'));
end

function s = build_defl_ichol(ctx, V, tau)
%BUILD_DEFL_ICHOL  ichol smoothing with the AMG-sketched coarse space, via the
%   C_tau form.  Isolates "AMG contributed the subspace" from "AMG contributed
%   the smoothing": if this matches ctau_amg, the V-cycle's smoothing is not
%   what was buying the conditioning.
    L = ctx.L;  Lt = ctx.Lt;
    Minv = @(r) Lt \ (L \ r);
    s = ok_arm(src.precond.make_Ctau_prec(V, Minv, ctx.A, tau));
end

function s = build_ctau_amg(ctx, V, tau)
%BUILD_CTAU_AMG  Both mechanisms at once:
%       C_tau^{-1} = tau*Q + (I - QA) M^{-1} (I - AQ),  Q = V(V'AV)^{-1}V'.
%   Answers whether deflation and the V-cycle are complementary at the two
%   ends of the spectrum or merely redundant.
    s = ok_arm(src.precond.make_Ctau_prec(V, ctx.Mamg, ctx.A, tau));
end

%% =========================================================================
%% Local helpers
%% =========================================================================
function a = arm(id, label, needs_V, needs_sym, sketch_only, is_reference, buildFn)
    a = struct('id', id, 'label', label, 'needs_V', needs_V, ...
               'needs_sym', needs_sym, 'sketch_only', sketch_only, ...
               'is_reference', is_reference, 'build', buildFn);
end

function s = ok_arm(prec)
    s = struct('prec', prec, 'valid', true, 'skip_reason', '');
end
