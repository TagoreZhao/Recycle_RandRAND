% TEST_SEQUENCE_CACHE_GUARD  The cache cannot silently return a different problem.
%
% The cache filename records only (case_name, h0, dt, nsteps).  Everything else
% that defines the problem -- Tstep/Tmax, nu, the channel box, and above all the
% Lagrange-point layout, which lives inside define_motion_list.m as literals -- is
% invisible to it.  Editing the point spacing and re-running therefore used to
% return the OLD sequence under the same name, with no warning, which is how an
% entire benchmark run was once spent measuring a problem that no longer existed.
%
% T2 is the assertion that pins the class of bug directly: two builds that differ
% only in Tstep produce the SAME cache tag and DIFFERENT fingerprints.
%
% T6/T7 cover the companion gate: a coupling block with more rows than free
% velocity DOFs makes K exactly singular, and must be refused at assembly time
% rather than surfacing as NaN sixty steps later.
%
% WHY H_TINY = 0.15.  make_bar_rotating's max(8, ...) floor binds at this mesh, so
% nb = 8 whether define_motion_list currently says 1.5*h0 or 0.6*h0.  These guards
% must not depend on the literal they exist to police -- at h0 = 0.1 the two give
% 8 and 12 points respectively and the tests would read differently depending on
% the working tree.
%
% Run:  test_sequence_cache_guard

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
p = add_recycle_paths();
rng(1);

H_TINY = 0.15;
NS     = 2;
DT     = 0.02;
CASE   = 'bar_rotating';
npass  = 0;

TAG   = sprintf('seq_%s_h%s_dt%s_n%d', CASE, num2str(H_TINY), num2str(DT), NS);
TAG   = regexprep(TAG, '[^\w]', '_');
CFILE = fullfile(p.cacheDir, [TAG '.mat']);

% Start from a known-absent entry.  Deleted again at the end; a failing assert
% leaves the file behind, which the next run clears here.
if exist(CFILE, 'file') == 2, delete(CFILE); end

opts_nc = struct('case_name', CASE, 'h0', H_TINY, 'dt', DT, 'nsteps', NS, ...
                 'use_cache', false, 'verify', false, 'quiet', true);
opts_c  = opts_nc;  opts_c.use_cache = true;

fprintf('=== test_sequence_cache_guard ===\n');

%% ---- T1: the fingerprint exists and describes what was asked for --------
S0 = build_stokes_sequence(opts_nc);
assert(isfield(S0, 'fingerprint'), 'T1: no fingerprint on the returned sequence');
fp = S0.fingerprint;
assert(strcmp(fp.case_name, CASE) && fp.h0 == H_TINY && fp.dt == DT && ...
       fp.nsteps == NS, 'T1: fingerprint disagrees with the request');
assert(abs(fp.Tmax - DT * fp.Tstep) < 1e-14, ...
       'T1: Tmax %.6g is not dt*Tstep %.6g', fp.Tmax, DT * fp.Tstep);
assert(~isempty(fp.X_first) && isequal(size(fp.X_first), size(fp.V_first)), ...
       'T1: point/velocity arrays are missing or mismatched');
npass = npass + 1;
fprintf('  PASS T1: fingerprint present and self-consistent (%d points)\n', ...
        size(fp.X_first, 1));

%% ---- T2: the tag's blind spot -- same filename, different problem -------
% Tstep sets Tmax and hence the rotor's angular velocity, but never reaches the
% tag.  This is exactly the aliasing the fingerprint exists to catch.
oA = opts_nc;  oA.Tstep = 61;
oB = opts_nc;  oB.Tstep = 41;
SA = build_stokes_sequence(oA);
SB = build_stokes_sequence(oB);

tagA = regexprep(sprintf('seq_%s_h%s_dt%s_n%d', CASE, num2str(H_TINY), ...
                         num2str(DT), NS), '[^\w]', '_');
assert(strcmp(tagA, TAG), 'T2: tag construction drifted from this test');
dX = max(abs(SA.fingerprint.X_first(:) - SB.fingerprint.X_first(:)));
assert(dX > 1e-8, ...
       'T2: Tstep 61 vs 41 left X_first identical (%.3e) -- nothing to detect', dX);
npass = npass + 1;
fprintf('  PASS T2: same tag, fingerprints differ by %.3e in X_first\n', dX);

%% ---- T3: a stale entry warns and rebuilds, and the file is corrected ----
S = build_stokes_sequence(opts_nc);
S.fingerprint.X_first(end, :) = [];      % as if the point spacing had changed
save(CFILE, 'S', '-v7.3');
n_before = size(S.fingerprint.X_first, 1);

ws = warning('off', 'build_stokes_sequence:staleCache');  lastwarn('');
S3 = build_stokes_sequence(opts_c);
[~, wid] = lastwarn;  warning(ws);
assert(strcmp(wid, 'build_stokes_sequence:staleCache'), ...
       'T3: expected staleCache warning, got id "%s"', wid);
assert(size(S3.fingerprint.X_first, 1) == n_before + 1, ...
       'T3: returned a sequence with %d points, expected the rebuilt %d', ...
       size(S3.fingerprint.X_first, 1), n_before + 1);
L = load(CFILE, 'S');
assert(size(L.S.fingerprint.X_first, 1) == n_before + 1, ...
       'T3: the stale file on disk was not overwritten');
npass = npass + 1;
fprintf('  PASS T3: stale entry warned, rebuilt, and overwrote the file\n');

%% ---- T4: a legacy entry with no fingerprint warns and rebuilds ----------
S = build_stokes_sequence(opts_nc);
S = rmfield(S, 'fingerprint');
save(CFILE, 'S', '-v7.3');

ws = warning('off', 'build_stokes_sequence:cacheNoFingerprint');  lastwarn('');
S4 = build_stokes_sequence(opts_c);
[~, wid] = lastwarn;  warning(ws);
assert(strcmp(wid, 'build_stokes_sequence:cacheNoFingerprint'), ...
       'T4: expected cacheNoFingerprint warning, got id "%s"', wid);
assert(isfield(S4, 'fingerprint'), 'T4: rebuild did not attach a fingerprint');
npass = npass + 1;
fprintf('  PASS T4: legacy entry warned and rebuilt with a fingerprint\n');

%% ---- T5: a clean hit is silent, and returns the same sequence -----------
lastwarn('');
S5 = build_stokes_sequence(opts_c);            % now a matching entry is on disk
[~, wid] = lastwarn;
assert(isempty(wid), 'T5: a matching cache hit warned with id "%s"', wid);
assert(isequal(S5.b{NS}, S0.b{NS}) && isequal(S5.xref{NS}, S0.xref{NS}), ...
       'T5: cached sequence differs from the uncached build');
npass = npass + 1;
fprintf('  PASS T5: matching hit is silent and bit-identical\n');

%% ---- T6: the over-constrained gate fires, with no mesh involved ---------
% 4 constraint rows over nU = 6 velocity DOFs, but supported on columns 1 and 2
% only, and column 2 is Dirichlet -- so exactly 1 free DOF is touched.
Cbad = sparse([1 2 3 4], [1 1 2 2], [1 1 1 1], 4, 6);
ok = false;
try
    assert_coupling_feasible(Cbad, 2, 7, 'synthetic');
catch ME
    ok = strcmp(ME.identifier, 'assert_coupling_feasible:overConstrained');
end
assert(ok, 'T6: over-constrained C was not refused');
npass = npass + 1;
fprintf('  PASS T6: over-constrained coupling refused at assembly time\n');

%% ---- T7: the shipped geometry passes the gate, with margin --------------
touched = assert_coupling_feasible(S0.Ccpl{1}, S0.veldofs, 1, CASE);
assert(touched >= S0.nC, 'T7: touched=%d is below nC=%d', touched, S0.nC);
npass = npass + 1;
fprintf('  PASS T7: real step-1 coupling feasible (nC=%d <= touched=%d)\n', ...
        S0.nC, touched);

if exist(CFILE, 'file') == 2, delete(CFILE); end

fprintf('=== test_sequence_cache_guard: %d checks passed ===\n', npass);
