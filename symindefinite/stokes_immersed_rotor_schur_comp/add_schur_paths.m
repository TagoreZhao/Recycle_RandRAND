function paths = add_schur_paths()
%ADD_SCHUR_PATHS  Path bootstrap for the Schur-complement study.
%   PATHS = ADD_SCHUR_PATHS()
%
%   SHADOWING HAZARD.  define_motion_list lives ONLY in the sibling folder
%   symindefinite/stokes_immersed_rotor, so that folder must be on the path --
%   but it also contains files whose names collide with this study's, and
%   addpath PREPENDS by default.  The sibling is therefore added with '-end'
%   and this folder with '-begin', so local copies always win.  Call
%   assert_local_helpers() after this to make the guarantee enforceable.
%
%   Returns a struct of resolved directories and creates the output root.

    thisFileDir = fileparts(mfilename('fullpath'));
    repoRoot    = fileparts(fileparts(thisFileDir));

    paths.thisFileDir  = thisFileDir;
    paths.repoRoot     = repoRoot;
    paths.siblingDir   = fullfile(repoRoot, 'symindefinite', 'stokes_immersed_rotor');
    paths.outDir       = fullfile(thisFileDir, 'schur_recycle');
    paths.testDir      = fullfile(thisFileDir, 'tests');

    addpath(repoRoot);                                   % makes src.* importable
    if exist(paths.siblingDir, 'dir')
        addpath(paths.siblingDir, '-end');               % define_motion_list ONLY
    else
        error('add_schur_paths:noSibling', ...
              'Sibling benchmark not found at %s (needed for define_motion_list).', ...
              paths.siblingDir);
    end
    addpath(thisFileDir, '-begin');                      % local copies win

    if ~exist(paths.outDir, 'dir')
        mkdir(paths.outDir);
    end
end
