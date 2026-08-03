function paths = add_paths()
%ADD_PATHS  Put every directory this study depends on on the MATLAB path.
%
%   paths = ADD_PATHS()
%
%   Call once at the top of every experiment.  Returns the resolved directories
%   so scripts never repeat fileparts chains.
%
%   Directories added:
%     repoRoot                                +src (src.precond.*, src.stokes.*)
%     coordinate_drift/kernel                 this folder
%     linear_solves/subspace_recycle/kernel   build_stokes_sequence, seq_K,
%                                             ildl_coordinate_map, transport_V,
%                                             orth_trunc
%     subspace_capture                        subspace_capture_directed
%     GP_train/pumadyn32nm                    the SPD driver's data + helpers
%
%   Nothing outside coordinate_drift/ is written to.
%
%   See also: make_case, run_all.

    kernelDir = fileparts(mfilename('fullpath'));
    studyDir  = fileparts(kernelDir);                  % .../coordinate_drift
    symDir    = fileparts(studyDir);                   % .../symindefinite
    repoRoot  = fileparts(symDir);                     % .../Recycle_RandRAND

    paths = struct();
    paths.repoRoot    = repoRoot;
    paths.kernelDir   = kernelDir;
    paths.studyDir    = studyDir;
    paths.expDir      = fullfile(studyDir, 'experiments');
    paths.outDir      = fullfile(studyDir, 'output');
    paths.figDir      = fullfile(studyDir, 'figures');
    paths.recycleKern = fullfile(symDir, 'linear_solves', 'subspace_recycle', 'kernel');
    paths.captureDir  = fullfile(repoRoot, 'subspace_capture');
    paths.gpDir       = fullfile(repoRoot, 'GP_train', 'pumadyn32nm');

    addpath(repoRoot);
    addpath(kernelDir);
    addpath(paths.expDir);
    addpath(paths.recycleKern);
    addpath(paths.captureDir);
    addpath(paths.gpDir);

    if ~exist(paths.outDir, 'dir'), mkdir(paths.outDir); end
    if ~exist(paths.figDir, 'dir'), mkdir(paths.figDir); end
end
