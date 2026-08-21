function paths = add_varvisc_schur_paths()
%ADD_VARVISC_SCHUR_PATHS  Bootstrap paths for the variable-viscosity study.

    thisDir = fileparts(mfilename('fullpath'));
    repoRoot = fileparts(fileparts(thisDir));

    paths.thisDir = thisDir;
    paths.repoRoot = repoRoot;
    paths.varviscDir = fullfile(repoRoot, 'symindefinite', 'stokes_varvisc_rotor');
    paths.outDir = fullfile(thisDir, 'varvisc_schur_recycle');
    paths.testDir = fullfile(thisDir, 'tests');
    paths.schurLocalDir = fullfile(repoRoot, 'symindefinite', ...
        'linear_solves', 'schur_complement', 'local');

    addpath(repoRoot);
    if ~exist(paths.varviscDir, 'dir')
        error('add_varvisc_schur_paths:noSibling', ...
              'Variable-viscosity sibling not found at %s.', paths.varviscDir);
    end
    addpath(paths.varviscDir, '-end');
    if ~exist(paths.schurLocalDir, 'dir')
        error('add_varvisc_schur_paths:noSchurLocal', ...
              'Shared Schur helpers not found at %s.', paths.schurLocalDir);
    end
    addpath(paths.schurLocalDir, '-end');
    addpath(thisDir, '-begin');
end
