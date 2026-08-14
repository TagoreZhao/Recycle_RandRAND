function paths = add_varvisc_schur_paths()
%ADD_VARVISC_SCHUR_PATHS  Bootstrap paths for the variable-viscosity study.

    thisDir = fileparts(mfilename('fullpath'));
    repoRoot = fileparts(fileparts(thisDir));

    paths.thisDir = thisDir;
    paths.repoRoot = repoRoot;
    paths.varviscDir = fullfile(repoRoot, 'symindefinite', 'stokes_varvisc_rotor');
    paths.outDir = fullfile(thisDir, 'varvisc_schur_recycle');
    paths.testDir = fullfile(thisDir, 'tests');

    addpath(repoRoot);
    if ~exist(paths.varviscDir, 'dir')
        error('add_varvisc_schur_paths:noSibling', ...
              'Variable-viscosity sibling not found at %s.', paths.varviscDir);
    end
    addpath(paths.varviscDir, '-end');
    addpath(thisDir, '-begin');
end
