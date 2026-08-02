function paths = add_recycle_paths()
%ADD_RECYCLE_PATHS  Put every directory this study depends on on the MATLAB path.
%
%   paths = ADD_RECYCLE_PATHS()
%
%   Call once, right after the standard linear_solves preamble.  Returns a struct
%   of the resolved directories so scripts can locate the repo root, the shared
%   kernel and their own output folder without repeating fileparts chains.
%
%   Directories added:
%     repoRoot                      +src package (src.precond.*, src.stokes.*)
%     subspace_recycle/kernel       this folder's helpers
%     subspace_capture              subspace_capture_directed (the capture metric)
%     symindefinite/stokes_immersed_rotor
%                                   define_motion_list, make_recording_pdef,
%                                   augment_recycle_V
%
%   See also: build_stokes_sequence, seq_K.

    kernelDir  = fileparts(mfilename('fullpath'));
    studyDir   = fileparts(kernelDir);                 % .../subspace_recycle
    solvesDir  = fileparts(studyDir);                  % .../linear_solves
    symDir     = fileparts(solvesDir);                 % .../symindefinite
    repoRoot   = fileparts(symDir);                    % .../Recycle_RandRAND

    paths = struct();
    paths.repoRoot   = repoRoot;
    paths.kernelDir  = kernelDir;
    paths.studyDir   = studyDir;
    paths.solvesDir  = solvesDir;
    paths.cacheDir   = fullfile(kernelDir, 'cache');
    paths.captureDir = fullfile(repoRoot, 'subspace_capture');
    paths.rotorDir   = fullfile(symDir, 'stokes_immersed_rotor');

    addpath(repoRoot);
    addpath(kernelDir);
    addpath(paths.captureDir);
    addpath(paths.rotorDir);

    if ~exist(paths.cacheDir, 'dir'), mkdir(paths.cacheDir); end
end
