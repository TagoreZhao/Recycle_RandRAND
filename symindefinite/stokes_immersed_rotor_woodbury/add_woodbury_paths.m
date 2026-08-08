function paths = add_woodbury_paths()
%ADD_WOODBURY_PATHS  Path bootstrap for the Woodbury-update study.
%   PATHS = ADD_WOODBURY_PATHS()
%
%   Directories added:
%     repoRoot     makes src.* importable (src.stokes.*, src.discretization.*)
%     kernelDir    linear_solves/subspace_recycle/kernel -- build_stokes_sequence,
%                  seq_K, seq_dCblk: the verified rank-2nC form this study solves
%                  with, plus its on-disk sequence cache
%     siblingDir   symindefinite/stokes_immersed_rotor -- define_motion_list ONLY
%     thisFileDir  this study
%
%   SHADOWING HAZARD, AND WHY EVERY LOCAL HELPER IS RENAMED.  The Schur twin
%   copies the figure-style helpers under their ORIGINAL names
%   (save_benchmark_figure, benchmark_fig_defaults, solver_style_table) and relies
%   on addpath('-begin') to win over the sibling folder's same-named copies.  That
%   ordering cannot be relied on here: build_stokes_sequence calls
%   add_recycle_paths() internally, which PREPENDS the sibling rotor dir -- mid-run,
%   after our '-begin'.  Rather than fight for path position, every helper in this
%   folder carries a woodbury_ / _woodbury name, so there is no collision to lose.
%   assert_local_helpers() is still called by every driver: it costs nothing and it
%   pins define_motion_list to the sibling.
%
%   Returns a struct of resolved directories and creates the output roots.
%
%   See also: assert_local_helpers, make_woodbury_params, add_recycle_paths.

    thisFileDir = fileparts(mfilename('fullpath'));
    symDir      = fileparts(thisFileDir);            % .../symindefinite
    repoRoot    = fileparts(symDir);                 % .../Recycle_RandRAND

    paths.thisFileDir = thisFileDir;
    paths.symDir      = symDir;
    paths.repoRoot    = repoRoot;
    paths.siblingDir  = fullfile(symDir, 'stokes_immersed_rotor');
    paths.kernelDir   = fullfile(symDir, 'linear_solves', 'subspace_recycle', ...
                                'kernel');
    paths.outDir      = fullfile(thisFileDir, 'woodbury_direct');
    paths.smokeDir    = fullfile(thisFileDir, 'woodbury_direct_smoke');
    paths.testDir     = fullfile(thisFileDir, 'tests');

    addpath(repoRoot);                               % makes src.* importable

    if ~exist(paths.kernelDir, 'dir')
        error('add_woodbury_paths:noKernel', ...
              ['Sequence kernel not found at %s (needed for ' ...
               'build_stokes_sequence, seq_K, seq_dCblk).'], paths.kernelDir);
    end
    addpath(paths.kernelDir);

    if ~exist(paths.siblingDir, 'dir')
        error('add_woodbury_paths:noSibling', ...
              'Sibling benchmark not found at %s (needed for define_motion_list).', ...
              paths.siblingDir);
    end
    addpath(paths.siblingDir, '-end');               % define_motion_list ONLY

    addpath(thisFileDir, '-begin');

    for d = {paths.outDir, paths.smokeDir}
        if ~exist(d{1}, 'dir')
            mkdir(d{1});
        end
    end
end
