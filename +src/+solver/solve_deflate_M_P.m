function Astat = solve_deflate_M_P(cfg, params, save_dir)
%SOLVE_DEFLATE_M_P  Run M*P deflation-preconditioned PCG on a given geometry.
%   ASTAT = SOLVE_DEFLATE_M_P(CFG, PARAMS, SAVE_DIR)
%
%   Inputs:
%     cfg      - struct from setup_geometry (mesh, cn, kappaFun, name)
%     params   - struct with solver hyperparameters:
%                .sm_eig, .lg_eig, .tau, .q, .SOLVER_TOL, .SOLVER_MAXIT,
%                .ICHOL_PREC_REFRESH, .DEFLAT_PREC_REFRESH, .dt, .Tstep,
%                .RAND_EIGS (vector), .sigma
%     save_dir - directory to save per-geometry results
%
%   Output:
%     Astat - struct with fields: solver_its, relres, flags, diffF

    import src.precond.*

    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % --- Unpack config ---
    msh      = cfg.mesh;
    cn       = cfg.cn;
    kappaFun = cfg.kappaFun;

    % --- Unpack params ---
    sm_eig             = params.sm_eig;
    lg_eig             = params.lg_eig;
    tau                = params.tau;
    q                  = params.q;
    SOLVER_TOL         = params.SOLVER_TOL;
    SOLVER_MAXIT       = params.SOLVER_MAXIT;
    ICHOL_PREC_REFRESH = params.ICHOL_PREC_REFRESH;
    DEFLAT_PREC_REFRESH = params.DEFLAT_PREC_REFRESH;
    DINVERSE_PREC_REFRESH = params.DINVERSE_PREC_REFRESH;
    dt                 = params.dt;
    Tstep              = params.Tstep;
    RAND_EIGS          = params.RAND_EIGS;
    sigma              = params.sigma;
    Kmodes             = params.Kmodes;
    snapshot_steps     = params.snapshot_steps;
    num_eigs           = params.num_eigs;

    % --- Unpack tentative_approx + min_subspace_iter params (RAND_EIGS=3) ---
    tent_theta            = params.tent_theta;
    tent_target_nc_factor = params.tent_target_nc_factor;
    tent_q                = params.tent_q;
    tent_omega            = params.tent_omega;
    tent_D_kind           = params.tent_D_kind;

    if isfield(params, 'visualize_kappa')
        visualize_kappa = params.visualize_kappa;
    else
        visualize_kappa = false;
    end

    if isfield(params, 'visualize_eigs')
        visualize_eigs = params.visualize_eigs;
    else
        visualize_eigs = false;
    end

    if isfield(params, 'AMG_PREC_REFRESH')
        AMG_PREC_REFRESH = params.AMG_PREC_REFRESH;
    else
        AMG_PREC_REFRESH = 1;
    end

    % --- AMG hyperparameters (forwarded to make_amg_preconditioner) ---
    if isfield(params, 'amg')
        amg_opts = params.amg;
    else
        amg_opts = struct();
    end
    if ~isfield(amg_opts, 'maxLevels'),     amg_opts.maxLevels     = 3;     end
    if ~isfield(amg_opts, 'minCoarseSize'), amg_opts.minCoarseSize = 800;   end
    if ~isfield(amg_opts, 'theta'),         amg_opts.theta         = 0.05;  end
    if ~isfield(amg_opts, 'omegaInterp'),   amg_opts.omegaInterp   = 4/3;   end
    if ~isfield(amg_opts, 'omegaSmooth'),   amg_opts.omegaSmooth   = 2/3;   end
    if ~isfield(amg_opts, 'preSmooth'),     amg_opts.preSmooth     = 1;     end
    if ~isfield(amg_opts, 'postSmooth'),    amg_opts.postSmooth    = 1;     end
    if ~isfield(amg_opts, 'maxAggSize'),    amg_opts.maxAggSize    = 16;    end
    if ~isfield(amg_opts, 'coarseSolve'),   amg_opts.coarseSolve   = 'chol';end

    % For K(t)
    eigK_large  = cell(numel(snapshot_steps),1);
    eigK_small  = cell(numel(snapshot_steps),1);
    
    % For A(t) = D_II + dt*K_II(t)
    eigA_large  = cell(numel(snapshot_steps),1);
    eigA_small  = cell(numel(snapshot_steps),1);

    snap_ptr       = 0;
    snapshot_times = [];

    % --- Variant name mapping ---
    rand_eigs_map = containers.Map([0, 1, 2, 3], {'rand_exact', 'rand_gaussian', 'rand_sparse', 'tent_approx'});
    num_variants = numel(RAND_EIGS);
    variant_names = cell(1, num_variants);
    for vi = 1:num_variants
        variant_names{vi} = rand_eigs_map(RAND_EIGS(vi));
    end

    % --- Mesh quantities ---
    N      = msh.N;
    IN     = msh.IN;
    Bdry   = msh.Bdry;
    numIN  = msh.numIN;
    numBdry = msh.numB;

    % --- Initial condition ---
    c = zeros(N, Tstep);
    c(Bdry, 1) = cn(Bdry);

    % Precompute Dirichlet coupling
    Dc0    = msh.D * cn;
    Dc0_IN = Dc0(IN);

    % Precompute Neumann load on interior DOFs (constant across timesteps)
    fN_I = cfg.fN(IN);

    % --- KL noise basis ---
    import src.forcing.*
    kvec     = generate_kvec(Kmodes);
    bbox     = [cfg.cx - cfg.sx, cfg.cx + cfg.sx, cfg.cy - cfg.sy, cfg.cy + cfg.sy];
    Phi      = eval_cosine_modes(msh.p, kvec, bbox);   % N x Kmodes
    Phi_IN   = Phi(IN, :);                              % numIN x Kmodes
    Phi_Bdry = Phi(Bdry, :);                            % numB x Kmodes
    dBeta    = sqrt(dt) * randn(Kmodes, Tstep);         % pre-generate all increments

    % --- Statistics ---
    Astat.unprecond_solver_its = zeros(Tstep-1, 1);
    Astat.chol_solver_its = zeros(Tstep-1, 1);
    Astat.ichol_solver_its = zeros(Tstep-1, 1);
    Astat.amg_precond_solver_its = zeros(Tstep-1, 1);
    Astat.amg_jac_precond_solver_its = zeros(Tstep-1, 1);
    for vi = 1:num_variants
        Astat.(['two_level_' variant_names{vi} '_solver_its']) = zeros(Tstep-1, 1);
    end
    for vi = 1:num_variants
        Astat.(['ichol_amg_two_level_' variant_names{vi} '_solver_its']) = zeros(Tstep-1, 1);
    end
    Astat.relres     = zeros(Tstep-1, 1);
    Astat.flags      = zeros(Tstep-1, 1);
    Astat.ReldiffF      = NaN(Tstep-1, 1);
    Astat.InvRelDiff = NaN(Tstep-1, 1);
    Astat.LowRankInvRelDiff = NaN(Tstep-1, 1);
    Astat.RelInitdiffF = NaN(Tstep-1, 1);

    % --- Per-variant subspace storage ---
    V_all     = cell(1, num_variants);
    V_all_amg = cell(1, num_variants);   % deflation V built with AMG-approximated A^{-1}

    % --- Locate exact-eigenvector variant for low-rank metric ---
    exact_vi = find(RAND_EIGS == 0, 1);
    if isempty(exact_vi)
        exact_vi = 1;  % fallback to first variant
    end

    %% ---------------------------- Time stepping -------------------------------
    for n = 1:Tstep-1
        tcur = n * dt;
        fprintf('[%s] t = %.3f\n', cfg.name, tcur);

        % --- ADDITIVE KL noise (mass-weighted Galerkin projection) ---
        dw_nodes_IN   = Phi_IN * dBeta(:, n);           % numIN x 1
        dw_nodes_Bdry = Phi_Bdry * dBeta(:, n);         % numB x 1
        noiseI = sigma * (msh.D_II * dw_nodes_IN + msh.D_IB * dw_nodes_Bdry);

        % --- external forcing (zero) ---
        fI = zeros(numIN, 1);

        % --- assemble K_II(t_n) and K_IB(t_n) ---
        kappa_e = kappaFun(msh.cent(:,1), msh.cent(:,2), tcur);
        Vscale  = msh.Vunit .* repelem(kappa_e, 9);
        Vii     = Vscale(msh.idxII);
        Vib     = Vscale(msh.idxIB);

        K_II = sparse(msh.I_II, msh.J_II, Vii, numIN, numIN);
        K_IB = sparse(msh.I_IB, msh.J_IB, Vib, numIN, numBdry);

        % --- system and RHS on interior DOFs ---
        A_II = msh.D_II + dt * K_II;
        rhsI = fI + msh.D_II * c(IN,n) + noiseI - (Dc0_IN + dt*(K_IB * cn(Bdry))) + dt * fN_I;

        %% -------- Ichol Preconditioner -----
        if (mod(n-1, ICHOL_PREC_REFRESH) == 0)
            fprintf('build new ichol preconditioner\n');
            L = ichol(A_II, struct('type','nofill'));
            Lt = L';
        end

        if mod(n-1, AMG_PREC_REFRESH) == 0
            fprintf('build new AMG preconditioner (ICHOL smoother)\n');
            Mapply = make_amg_preconditioner(A_II, ...
                'maxLevels',     amg_opts.maxLevels, ...
                'minCoarseSize', amg_opts.minCoarseSize, ...
                'theta',         amg_opts.theta, ...
                'omegaInterp',   amg_opts.omegaInterp, ...
                'preSmooth',     amg_opts.preSmooth, ...
                'postSmooth',    amg_opts.postSmooth, ...
                'maxAggSize',    amg_opts.maxAggSize, ...
                'coarseSolve',   amg_opts.coarseSolve, ...
                'fineSmootherL',  L, ...
                'fineSmootherLt', Lt);
            fprintf('build new AMG preconditioner (damped Jacobi smoother)\n');
            Mapply_jac = make_amg_preconditioner(A_II, ...
                'maxLevels',     amg_opts.maxLevels, ...
                'minCoarseSize', amg_opts.minCoarseSize, ...
                'theta',         amg_opts.theta, ...
                'omegaInterp',   amg_opts.omegaInterp, ...
                'omegaSmooth',   amg_opts.omegaSmooth, ...
                'preSmooth',     amg_opts.preSmooth, ...
                'postSmooth',    amg_opts.postSmooth, ...
                'maxAggSize',    amg_opts.maxAggSize, ...
                'coarseSolve',   amg_opts.coarseSolve);
        end

        if (mod(n-1, DINVERSE_PREC_REFRESH) == 0)
            fprintf('build exact Chol preconditioner\n');
            [R,flag_chol,P] = chol(A_II,'lower');
            invApply =  @(x) ( ( ( P * ( ( ( R \ ( ((x)' * P)' ) )' / R )' ) ) )' )';
            if flag_chol ~= 0
                warning('Cholesky failed at time step %d (flag = %d).', n, flag_chol);
            end
        end

        %% ---------- Deflation Preconditioner (per-variant) -------------
        if (mod(n-1, DEFLAT_PREC_REFRESH) == 0)
            fprintf('build new deflation preconditioner\n');
            IcholinvApply = @(x) ( ( ( P * ( ( ( R \ ( ((L*x)' * P)' ) )' / R )' ) ) )' * L )';
            IcholApply_build = @(x) L \ (A_II * (x' / L)');
            for vi = 1:num_variants
                re = RAND_EIGS(vi);
                if re == 1
                    s = size(A_II, 1);
                    omega = randn(s, 2 * lg_eig);
                    Vlarge = subspace_iter(IcholApply_build, omega, q);
                    omega = randn(s, 2 * sm_eig);
                    Vsmall = subspace_iter(IcholinvApply, omega, q);
                    Vtmp = [Vlarge Vsmall];
                    [Vtmp,~]  = qr(Vtmp,0);
                    V_all{vi} = Vtmp;
                elseif re == 0
                    [Vlarge, ~] = eigs(IcholApply_build,numIN, lg_eig, 'largestabs', struct('Tolerance',1e-8,'MaxIterations',5000));
                    [Vlarge,~]  = qr(Vlarge,0);
                    [Vsmall, ~] = eigs(IcholinvApply,numIN, sm_eig, 'smallestabs', struct('Tolerance',1e-8,'MaxIterations',5000));
                    [Vsmall,~]  = qr(Vsmall,0);
                    Vtmp = [Vlarge Vsmall];
                    [Vtmp,~]  = qr(Vtmp,0);
                    V_all{vi} = Vtmp;
                elseif re == 2
                    s = size(A_II, 1);
                    omega = sjlt(s, 2 * sm_eig, 8);
                    theta =  sjlt(s, 4 * sm_eig, 8)';
                    thetaVlarge = theta*subspace_iter_plain(IcholinvApply, omega, q);
                    [~,R_sk] = qr(thetaVlarge,0);
                    invR_sk = inv(R_sk);
                    Romega = chol(invR_sk'*(omega'*subspace_iter_plain(IcholinvApply,subspace_iter_plain(IcholinvApply, omega*invR_sk, q),q)));
                    Romega = Romega*R_sk;
                    V_fun = @(x) subspace_iter_plain(IcholinvApply, omega*(Romega\x), q);
                    V_fun_t = @(x) ((subspace_iter_plain(IcholinvApply, x, q)'*omega)/Romega)';
                    Vstruct.V_fun = V_fun;
                    Vstruct.V_fun_t = V_fun_t;
                    Vstruct.l = size(Romega,1);
                    Vstruct.n = s;
                    V_all{vi} = Vstruct;
                elseif re == 3
                    s = size(A_II, 1);
                    target_nc = round(tent_target_nc_factor * sm_eig);
                    P_tent = tentative_prolongator(A_II, tent_theta, [], target_nc);
                    P_tent = full(P_tent);   % min_subspace_iter expects a dense block
                    switch tent_D_kind
                        case 'all_ones'
                            Dinv_tent = ones(s, 1);
                        case 'diag_M'
                            Dinv_tent = 1 ./ local_compute_diag_M(A_II, L, Lt);
                        otherwise
                            error('Unknown tent_D_kind: %s', tent_D_kind);
                    end
                    V_all{vi} = min_subspace_iter(IcholApply_build, P_tent, ...
                                                  tent_q, Dinv_tent, tent_omega, false);
                end
            end

            % Build V_all_amg using AMG-approximated A^{-1}: T_amg^{-1} = L^T M^{-1} L
            IcholinvApply_amg = @(x) Lt * Mapply(L * x);
            for vi = 1:num_variants
                re = RAND_EIGS(vi);
                if re == 1
                    s = size(A_II, 1);
                    omega = randn(s, 2 * lg_eig);
                    Vlarge = subspace_iter(IcholApply_build, omega, q);
                    omega = randn(s, 2 * sm_eig);
                    Vsmall = subspace_iter(IcholinvApply_amg, omega, q);
                    Vtmp = [Vlarge Vsmall];
                    [Vtmp,~]  = qr(Vtmp,0);
                    V_all_amg{vi} = Vtmp;
                elseif re == 0
                    [Vlarge, ~] = eigs(IcholApply_build,numIN, lg_eig, 'largestabs', struct('Tolerance',1e-8,'MaxIterations',5000));
                    [Vlarge,~]  = qr(Vlarge,0);
                    [Vsmall, ~] = eigs(IcholinvApply_amg,numIN, sm_eig, 'smallestabs', struct('Tolerance',1e-8,'MaxIterations',5000));
                    [Vsmall,~]  = qr(Vsmall,0);
                    Vtmp = [Vlarge Vsmall];
                    [Vtmp,~]  = qr(Vtmp,0);
                    V_all_amg{vi} = Vtmp;
                elseif re == 2
                    s = size(A_II, 1);
                    omega = sjlt(s, 2 * sm_eig, 8);
                    theta =  sjlt(s, 4 * sm_eig, 8)';
                    thetaVlarge = theta*subspace_iter_plain(IcholinvApply_amg, omega, q);
                    [~,R_sk] = qr(thetaVlarge,0);
                    invR_sk = inv(R_sk);
                    Romega = chol(invR_sk'*(omega'*subspace_iter_plain(IcholinvApply_amg,subspace_iter_plain(IcholinvApply_amg, omega*invR_sk, q),q)));
                    Romega = Romega*R_sk;
                    V_fun_amg   = @(x) subspace_iter_plain(IcholinvApply_amg, omega*(Romega\x), q);
                    V_fun_amg_t = @(x) ((subspace_iter_plain(IcholinvApply_amg, x, q)'*omega)/Romega)';
                    Vstruct_amg.V_fun   = V_fun_amg;
                    Vstruct_amg.V_fun_t = V_fun_amg_t;
                    Vstruct_amg.l = size(Romega,1);
                    Vstruct_amg.n = s;
                    V_all_amg{vi} = Vstruct_amg;
                elseif re == 3
                    % min_subspace_iter only uses forward IcholApply_build (independent of AMG); reuse V_all{vi}.
                    V_all_amg{vi} = V_all{vi};
                end
            end
        end

        if visualize_eigs
            if snap_ptr < numel(snapshot_steps) && n == snapshot_steps(snap_ptr+1)
                snap_ptr = snap_ptr + 1;
                snapshot_times(end+1) = tcur;

                fprintf('Computing eigenvalues at t = %.3f (snapshot %d)\n', tcur, snap_ptr);
                k = min(size(A_II,1), num_eigs);
                opts = struct('tol',1e-6,'maxit',1000);

                % K(t): largest/smallest
                eigK_large{snap_ptr} = dt*eigs(K_II, k, 'largestabs',  opts);
                eigK_small{snap_ptr} = dt*eigs(K_II, k, 'smallestabs', opts);

                % A(t): largest/smallest
                eigA_large{snap_ptr} = eigs(A_II, k, 'largestabs',  opts);
                eigA_small{snap_ptr} = eigs(A_II, k, 'smallestabs', opts);
            end
        end

        %% -------------------- SOLVE & ENFORCE DIRICHLET -----------------------
        x0 = c(IN, n);   % warm start
        rhs_norm   = norm(rhsI);
        rhs_scaled = rhsI / rhs_norm;
        x0_scaled  = x0 / rhs_norm;

        % --------- Unpreconditioned -----------------
        [~, ~, ~, unprecond_solver_iter] = pcg(A_II, rhs_scaled, SOLVER_TOL, SOLVER_MAXIT, [], [], x0_scaled);
        fprintf('Unprecond: finish solving linear system in %.3f iteration\n', unprecond_solver_iter);

        % --------- Ichol Preconditioned Solver -----------------
        [~, ~, ~, ichol_solver_iter] = pcg( ...
            A_II, rhs_scaled, SOLVER_TOL, SOLVER_MAXIT, L, Lt, x0_scaled);
        fprintf('Ichol: finish solving linear system in %.3f iteration\n', ichol_solver_iter);

        % --------- AMG Preconditioned Solver (ICHOL smoother) -----------------
        [~, ~, ~, amg_iter] = pcg(A_II, rhs_scaled, SOLVER_TOL, SOLVER_MAXIT, Mapply, [], x0_scaled);
        Astat.amg_precond_solver_its(n) = amg_iter;
        fprintf('AMG (ICHOL smoother): finish solving linear system in %.3f iteration\n', amg_iter);

        % --------- AMG Preconditioned Solver (damped Jacobi smoother) ---------
        [~, ~, ~, amg_jac_iter] = pcg(A_II, rhs_scaled, SOLVER_TOL, SOLVER_MAXIT, Mapply_jac, [], x0_scaled);
        Astat.amg_jac_precond_solver_its(n) = amg_jac_iter;
        fprintf('AMG (Jacobi smoother): finish solving linear system in %.3f iteration\n', amg_jac_iter);

        % --------- Chol Preconditioned Solver -----------------
        [c_in, flag, relres, chol_solver_iter] = pcg( ...
            A_II, rhs_scaled, SOLVER_TOL, SOLVER_MAXIT, invApply, [], x0_scaled);
        fprintf('Direct Inverse: finish solving linear system in %.3f iteration\n', chol_solver_iter);

        % --------- Two Level Preconditioned Solver (per-variant) ---------
        IcholApply_def = @(x) L \ ( A_II * ( Lt \ x ) );
        for vi = 1:num_variants
            re = RAND_EIGS(vi);
            vn = variant_names{vi};
            [Papply_vi, ~, ~] = deflation_P_apply(V_all{vi}, IcholApply_def, tau, [], re);
            Bapply_vi = @(r) Lt \ ( Papply_vi( L \ r ) );
            [~, ~, ~, iter_vi] = pcg(A_II, rhs_scaled, SOLVER_TOL, SOLVER_MAXIT, @(r) Bapply_vi(r), [], x0_scaled);
            fprintf('two level precond [%s]: finish solving linear system in %.3f iteration\n', vn, iter_vi);
            Astat.(['two_level_' vn '_solver_its'])(n) = iter_vi;
        end

        % --------- ICHOL + AMG-deflation Two Level Preconditioned Solver (per-variant) ---------
        for vi = 1:num_variants
            re = RAND_EIGS(vi);
            vn = variant_names{vi};
            if re == 3
                % V_all_amg{vi} == V_all{vi} for re==3; iter count is identical.
                Astat.(['ichol_amg_two_level_' vn '_solver_its'])(n) = ...
                    Astat.(['two_level_' vn '_solver_its'])(n);
                continue;
            end
            [Papply_vi, ~, ~] = deflation_P_apply(V_all_amg{vi}, IcholApply_def, tau, [], re);
            Bapply_vi = @(r) Lt \ ( Papply_vi( L \ r ) );
            [~, ~, ~, iter_vi] = pcg(A_II, rhs_scaled, SOLVER_TOL, SOLVER_MAXIT, @(r) Bapply_vi(r), [], x0_scaled);
            fprintf('ichol+amg two level [%s]: finish solving linear system in %.3f iteration\n', vn, iter_vi);
            Astat.(['ichol_amg_two_level_' vn '_solver_its'])(n) = iter_vi;
        end

        if flag ~= 0
            warning('linear solve failed at time step %d (flag = %d).', n, flag);
        end

        c(IN, n+1) = c_in * rhs_norm;

        % --- Statistics ---
        Astat.unprecond_solver_its(n) = unprecond_solver_iter;
        Astat.ichol_solver_its(n) = ichol_solver_iter;
        Astat.chol_solver_its(n) = chol_solver_iter;
        Astat.relres(n)     = relres;
        Astat.flags(n)      = flag;
        if n == 1
            A_prev = A_II;
            A_1 = A_II;
        else
            Astat.ReldiffF(n) = norm(A_II / norm(A_II,'fro') - A_prev / norm(A_prev, 'fro'), 'fro');
            Astat.RelInitdiffF(n) = norm(A_II / norm(A_II,'fro') - A_1 / norm(A_1, 'fro'), 'fro');
            Omega = randn(size(A_II,1),10)/sqrt(10);
            AinvOmega = A_II \ Omega;
            Astat.InvRelDiff(n) = norm(invApply(Omega) - AinvOmega,'fro') / norm(AinvOmega,'fro');
            V = V_all{exact_vi};
            Astat.LowRankInvRelDiff(n) = norm(A_II \ (V*(V'*Omega) - Omega),'fro') / norm(AinvOmega,'fro');
            A_prev = A_II;
        end

        % Memory cleanup
        clear K_II K_IB Vscale Vii Vib kappa_e;
    end

    %% ---------------------------- Save results --------------------------------
    tgrid = (1:Tstep-1) * dt;

    % Unpreconditioned iterations plot
    f = figure('Visible','off');
    plot(tgrid, Astat.unprecond_solver_its, '-o'); grid on;
    xlabel('t_n'); ylabel('iterations');
    title(sprintf('%s | %s  (ICHOL=%d, Deflat=%d)', ...
        strrep(cfg.name,'_',' '), cfg.kappa_label, ICHOL_PREC_REFRESH, DEFLAT_PREC_REFRESH));
    exportgraphics(f, fullfile(save_dir, 'unprecond_solver_iterations.png'), 'Resolution', 180);
    close(f);

    % Ichol iterations plot
    f = figure('Visible','off');
    plot(tgrid, Astat.ichol_solver_its, '-o'); grid on;
    xlabel('t_n'); ylabel('iterations');
    title(sprintf('%s | %s  (ICHOL=%d, Deflat=%d)', ...
        strrep(cfg.name,'_',' '), cfg.kappa_label, ICHOL_PREC_REFRESH, DEFLAT_PREC_REFRESH));
    exportgraphics(f, fullfile(save_dir, 'ichol_solver_iterations.png'), 'Resolution', 180);
    close(f);

    % Chol iterations plot
    f = figure('Visible','off');
    plot(tgrid, Astat.chol_solver_its, '-o'); grid on;
    xlabel('t_n'); ylabel('iterations');
    title(sprintf('%s | %s  (ICHOL=%d, Deflat=%d)', ...
        strrep(cfg.name,'_',' '), cfg.kappa_label, ICHOL_PREC_REFRESH, DEFLAT_PREC_REFRESH));
    exportgraphics(f, fullfile(save_dir, 'chol_solver_iterations.png'), 'Resolution', 180);
    close(f);

    % AMG (ICHOL smoother) iterations plot
    f = figure('Visible','off');
    plot(tgrid, Astat.amg_precond_solver_its, '-o'); grid on;
    xlabel('t_n'); ylabel('iterations');
    title(sprintf('%s | %s  AMG-ICHOL  (AMG=%d, Deflat=%d)', ...
        strrep(cfg.name,'_',' '), cfg.kappa_label, AMG_PREC_REFRESH, DEFLAT_PREC_REFRESH));
    exportgraphics(f, fullfile(save_dir, 'amg_solver_iterations.png'), 'Resolution', 180);
    close(f);

    % AMG (damped Jacobi smoother) iterations plot
    f = figure('Visible','off');
    plot(tgrid, Astat.amg_jac_precond_solver_its, '-o'); grid on;
    xlabel('t_n'); ylabel('iterations');
    title(sprintf('%s | %s  AMG-Jacobi  (AMG=%d, Deflat=%d)', ...
        strrep(cfg.name,'_',' '), cfg.kappa_label, AMG_PREC_REFRESH, DEFLAT_PREC_REFRESH));
    exportgraphics(f, fullfile(save_dir, 'amg_jac_solver_iterations.png'), 'Resolution', 180);
    close(f);

    % Per-variant two-level iterations plots
    for vi = 1:num_variants
        vn = variant_names{vi};
        field = ['two_level_' vn '_solver_its'];
        f = figure('Visible','off');
        plot(tgrid, Astat.(field), '-o'); grid on;
        xlabel('t_n'); ylabel('iterations');
        title(sprintf('%s | %s [%s]  (ICHOL=%d, Deflat=%d)', ...
            strrep(cfg.name,'_',' '), cfg.kappa_label, strrep(vn,'_',' '), ...
            ICHOL_PREC_REFRESH, DEFLAT_PREC_REFRESH));
        exportgraphics(f, fullfile(save_dir, ['two_level_' vn '_solver_iterations.png']), 'Resolution', 180);
        close(f);
    end

    % Per-variant ICHOL+AMG two-level iterations plots
    for vi = 1:num_variants
        vn = variant_names{vi};
        field = ['ichol_amg_two_level_' vn '_solver_its'];
        f = figure('Visible','off');
        plot(tgrid, Astat.(field), '-o'); grid on;
        xlabel('t_n'); ylabel('iterations');
        title(sprintf('%s | %s [ichol+amg two-level %s]  (ICHOL=%d, AMG=%d, Deflat=%d)', ...
            strrep(cfg.name,'_',' '), cfg.kappa_label, strrep(vn,'_',' '), ...
            ICHOL_PREC_REFRESH, AMG_PREC_REFRESH, DEFLAT_PREC_REFRESH));
        exportgraphics(f, fullfile(save_dir, ['ichol_amg_two_level_' vn '_solver_iterations.png']), 'Resolution', 180);
        close(f);
    end

    % All-solvers comparison plot
    f = figure('Visible','off', 'Position', [100 100 900 500]);
    hold on;
    % plot(tgrid, Astat.unprecond_solver_its, ':', 'DisplayName', 'unprecond');
    plot(tgrid, Astat.ichol_solver_its, '--', 'DisplayName', 'ichol');
    plot(tgrid, Astat.amg_precond_solver_its, ':', 'DisplayName', 'amg-ichol');
    plot(tgrid, Astat.amg_jac_precond_solver_its, ':', 'DisplayName', 'amg-jac');
    plot(tgrid, Astat.chol_solver_its, '-.', 'DisplayName', 'chol');
    var_colors = lines(num_variants);
    for vi = 1:num_variants
        vn = variant_names{vi};
        field = ['two_level_' vn '_solver_its'];
        plot(tgrid, Astat.(field), '-', 'Color', var_colors(vi,:), ...
            'DisplayName', strrep(vn, '_', ' '));
    end
    amg_var_colors = lines(num_variants + 3);
    for vi = 1:num_variants
        vn = variant_names{vi};
        field = ['ichol_amg_two_level_' vn '_solver_its'];
        plot(tgrid, Astat.(field), '--', 'Color', amg_var_colors(vi+3,:), ...
            'DisplayName', ['ichol+amg ' strrep(vn, '_', ' ')]);
    end
    hold off; grid on;
    xlabel('t_n'); ylabel('PCG iterations');
    title(sprintf('%s | %s  (ICHOL=%d, Deflat=%d)', ...
        strrep(cfg.name,'_',' '), cfg.kappa_label, ICHOL_PREC_REFRESH, DEFLAT_PREC_REFRESH));
    legend('Location', 'best', 'FontSize', 7);
    exportgraphics(f, fullfile(save_dir, 'all_solvers_comparison.png'), 'Resolution', 180);
    close(f);

    % Step-to-step change plot
    f = figure('Visible','off');
    plot(tgrid, Astat.ReldiffF, '-o'); grid on;
    xlabel('t_n'); ylabel('||A_i - A_{i-1}||_F / ||A_prev||_F');
    title(sprintf('%s | %s: step-to-step change', strrep(cfg.name,'_',' '), cfg.kappa_label));
    exportgraphics(f, fullfile(save_dir, 'relative_step_to_step_change.png'), 'Resolution', 180);
    close(f);

    % Diff from Initial plot
    f = figure('Visible','off');
    plot(tgrid, Astat.RelInitdiffF, '-o'); grid on;
    xlabel('t_n'); ylabel('||A_i - A_1||_F / ||A_1||_F');
    title(sprintf('%s | %s: diff from initial', strrep(cfg.name,'_',' '), cfg.kappa_label));
    exportgraphics(f, fullfile(save_dir, 'diff_from_initial.png'), 'Resolution', 180);
    close(f);

    % Direct Inv Change Plot
    f = figure('Visible','off');
    plot(tgrid, Astat.InvRelDiff, '-o'); grid on;
    xlabel('t_n'); ylabel('||A_1^{-1} \Omega - A_i^{-1} \Omega||_F / ||A_i^{-1} \Omega||_F');
    title(sprintf('%s | %s: relative inverse difference', strrep(cfg.name,'_',' '), cfg.kappa_label));
    exportgraphics(f, fullfile(save_dir, 'relative_inverse_difference.png'), 'Resolution', 180);
    close(f);

    % Low Rank Inv Difference Plot
    f = figure('Visible','off');
    plot(tgrid, Astat.LowRankInvRelDiff, '-o'); grid on;
    xlabel('t_n'); ylabel('||A_1^{-1} (P_v - I) \Omega||_F / ||A_i^{-1} \Omega||_F');
    title(sprintf('%s | %s: low rank inverse difference', strrep(cfg.name,'_',' '), cfg.kappa_label));
    exportgraphics(f, fullfile(save_dir, 'low_rank_inverse_difference.png'), 'Resolution', 180);
    close(f);

    % CSVs
    writematrix(Astat.unprecond_solver_its, fullfile(save_dir, 'unprecond_solver_iterations.csv'));
    writematrix(Astat.ichol_solver_its, fullfile(save_dir, 'ichol_solver_iterations.csv'));
    writematrix(Astat.amg_precond_solver_its, fullfile(save_dir, 'amg_solver_iterations.csv'));
    writematrix(Astat.amg_jac_precond_solver_its, fullfile(save_dir, 'amg_jac_solver_iterations.csv'));
    writematrix(Astat.chol_solver_its, fullfile(save_dir, 'chol_solver_iterations.csv'));
    for vi = 1:num_variants
        vn = variant_names{vi};
        field = ['two_level_' vn '_solver_its'];
        writematrix(Astat.(field), fullfile(save_dir, ['two_level_' vn '_solver_iterations.csv']));
    end
    for vi = 1:num_variants
        vn = variant_names{vi};
        field = ['ichol_amg_two_level_' vn '_solver_its'];
        writematrix(Astat.(field), fullfile(save_dir, ['ichol_amg_two_level_' vn '_solver_iterations.csv']));
    end

    if visualize_eigs
        %% ---- A(t), largest eigs ----
        for k = 1:numel(snapshot_steps)
            lam = eigA_large{k};
            if isempty(lam), continue; end

            lam = sort(real(lam), 'descend');
            lam = max(lam, 1e-12);

            fig = figure;
            semilogy(1:numel(lam), lam, '.-');
            xlabel('Index i');
            ylabel('Eigenvalue \lambda_i (log scale)');
            title(sprintf('ESD of A(t) (largest) at t = %.3f', snapshot_times(k)));
            grid on;

            fname = sprintf('%s/A_largest_t%.3f', save_dir, snapshot_times(k));
            saveas(fig, [fname '.png']);
            saveas(fig, [fname '.fig']);
            close(fig);
        end

        %% ---- A(t), smallest eigs ----
        for k = 1:numel(snapshot_steps)
            lam = eigA_small{k};
            if isempty(lam), continue; end

            lam = sort(real(lam), 'descend');
            lam = max(lam, 1e-12);

            fig = figure;
            semilogy(1:numel(lam), lam, '.-');
            xlabel('Index i');
            ylabel('Eigenvalue \lambda_i (log scale)');
            title(sprintf('ESD of A(t) (smallest) at t = %.3f', snapshot_times(k)));
            grid on;

            fname = sprintf('%s/A_smallest_t%.3f', save_dir, snapshot_times(k));
            saveas(fig, [fname '.png']);
            saveas(fig, [fname '.fig']);
            close(fig);
        end

        %% ---- K(t), smallest eigs ----
        for k = 1:numel(snapshot_steps)
            lam = eigK_small{k};
            if isempty(lam), continue; end

            lam = sort(real(lam), 'descend');
            lam = max(lam, 1e-12);

            fig = figure;
            semilogy(1:numel(lam), lam, '.-');
            xlabel('Index i');
            ylabel('Eigenvalue \lambda_i (log scale)');
            title(sprintf('ESD of K(t) (smallest) at t = %.3f', snapshot_times(k)));
            grid on;

            fname = sprintf('%s/K_smallest_t%.3f', save_dir, snapshot_times(k));
            saveas(fig, [fname '.png']);
            saveas(fig, [fname '.fig']);
            close(fig);
        end

        if ~isempty(snapshot_steps)
            largest_eigs  = eigs(msh.D_II, min(size(msh.D_II,1), num_eigs), 'largestabs',  struct('tol',1e-6,'maxit',1000));
            smallest_eigs = eigs(msh.D_II, min(size(msh.D_II,1), num_eigs), 'smallestabs', struct('tol',1e-6,'maxit',1000));

            lam_largest  = sort(real(largest_eigs),  'descend');
            lam_smallest = sort(real(smallest_eigs), 'descend');

            lam_largest  = max(lam_largest,  1e-12);
            lam_smallest = max(lam_smallest, 1e-12);

            % Largest
            fig = figure;
            semilogy(1:numel(lam_largest), lam_largest, '.-');
            xlabel('Index i');
            ylabel('Largest Eigenvalue \lambda_i (log scale)');
            title('ESD of M (= D_{II}) (largest)');
            grid on;

            saveas(fig, fullfile(save_dir, 'M_largest.png'));
            saveas(fig, fullfile(save_dir, 'M_largest.fig'));
            close(fig);

            % Smallest
            fig = figure;
            semilogy(1:numel(lam_smallest), lam_smallest, '.-');
            xlabel('Index i');
            ylabel('Smallest Eigenvalue \lambda_i (log scale)');
            title('ESD of M (= D_{II}) (smallest)');
            grid on;

            saveas(fig, fullfile(save_dir, 'M_smallest.png'));
            saveas(fig, fullfile(save_dir, 'M_smallest.fig'));
            close(fig);
        end
    end

    %% ---------------------------- Kappa animation ----------------------------
    if visualize_kappa
        gifpath = fullfile(save_dir, 'kappa_animation.gif');
        if exist(gifpath, 'file'), delete(gifpath); end

        % Pass 1: compute consistent z/color limits across all timesteps
        zlo = Inf;  zhi = -Inf;
        for n = 1:Tstep-1
            kn  = kappaFun(msh.p(:,1), msh.p(:,2), n * dt);
            zlo = min(zlo, min(kn));
            zhi = max(zhi, max(kn));
        end

        % Pass 2: render one frame per timestep and append to GIF
        fig = figure('Visible','off', 'Color','w');
        ax  = axes('Parent', fig);
        colormap(ax, parula);
        colorbar(ax);

        for n = 1:Tstep-1
            kn = kappaFun(msh.p(:,1), msh.p(:,2), n * dt);
            h  = trisurf(msh.t, msh.p(:,1), msh.p(:,2), kn, ...
                     'Parent', ax, 'FaceColor', 'interp', 'EdgeColor', 'none');
            if zlo < zhi
                zlim(ax, [zlo, zhi]);
                clim(ax, [zlo, zhi]);
            end
            xlabel(ax, 'x');  ylabel(ax, 'y');  zlabel(ax, '\kappa');
            title(ax, sprintf('%s | %s   t = %.3f', ...
                strrep(cfg.name,'_',' '), cfg.kappa_label, n * dt));
            view(ax, [-30, 30]);
            drawnow;
            exportgraphics(fig, gifpath, 'Append', true);
            delete(h);
        end
        close(fig);
        fprintf('Saved kappa animation to %s\n', gifpath);
    end
end

function kvec = generate_kvec(K)
%GENERATE_KVEC  K wavenumber pairs (kx,ky) sorted by ascending frequency.
%   kvec = generate_kvec(K) returns a K x 2 matrix of non-negative integer
%   wavenumber pairs with (kx,ky) ~= (0,0), sorted by kx^2+ky^2 then kx.
    candidates = [];
    maxk = ceil(sqrt(2*K)) + 1;
    for kx = 0:maxk
        for ky = 0:maxk
            if kx == 0 && ky == 0, continue; end
            candidates = [candidates; kx ky kx^2+ky^2]; %#ok<AGROW>
        end
    end
    [~, idx] = sortrows(candidates, [3 1]);
    kvec = candidates(idx(1:K), 1:2);
end

function d = local_compute_diag_M(A, L, Lt)
%LOCAL_COMPUTE_DIAG_M  diag(M) = diag(L^{-1} A L^{-T}) via 256-column batched probing.
    n = size(A, 1); chunk = 256; d = zeros(n, 1);
    for c0 = 1:chunk:n
        c1 = min(c0 + chunk - 1, n);
        E = sparse(c0:c1, 1:(c1 - c0 + 1), 1, n, c1 - c0 + 1);
        Z = Lt \ E; AZ = A * Z;
        d(c0:c1) = full(sum(Z .* AZ, 1)).';
    end
end
