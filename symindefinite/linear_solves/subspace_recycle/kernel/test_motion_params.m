% TEST_MOTION_PARAMS  The bar's geometry knobs are parameters, and the defaults
% reproduce the previous hardcoded literals BIT-FOR-BIT.
%
% 37 call sites across 5 studies, and 26 cache files, depend on the default
% geometry not moving.  A one-point change in nb would change nC everywhere and
% invalidate every committed result, so T1/T2 compare against hand-written copies
% of the ORIGINAL expressions rather than against the new code path -- comparing
% the new code to itself would prove nothing.
%
% T1 uses typecast to uint64, not isequal: isequal cannot distinguish 0 from -0,
% and "bit-identical" is exactly the claim being made.
%
% T6 pins the one subtlety that is easy to get wrong: build_stokes_sequence
% evaluates step n at t = n*dt, so step 1 is t = dt and NOT t = 0.  Placing a
% chosen angle on step 1 therefore needs theta0 = target - omega*dt, and omega*dt
% collapses to 2*pi*nrev/Tstep.
%
% Run:  test_motion_params

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
p = add_recycle_paths();
rng(1);

H_TINY = 0.15;      % mesh-touching tests only; T1-T6 build no mesh
NS     = 2;
DT     = 0.02;
TSTEP  = 61;
NREV   = 2;         % must match make_bar_rotating's literal; T6 fails loudly if not
npass  = 0;

% A geo struct built by hand: T1-T6 need no mesh at all.
geo = struct('x1', 0, 'x2', 4, 'y1', 0, 'y2', 1, 'xc', 2, 'yc', 0.5, ...
             'h0', 0.03, 'Tmax', DT * TSTEP);

fprintf('=== test_motion_params ===\n');

%% ---- T1: defaults are bit-identical to the original literals ------------
% The original expressions, transcribed verbatim from the pre-change file.
Lb_old  = 0.35 * (geo.y2 - geo.y1);
om_old  = 2 * pi * NREV / geo.Tmax;
nb_old  = max(8, ceil(2 * Lb_old / (0.6 * geo.h0)));
s_old   = linspace(-Lb_old, Lb_old, nb_old)';

cases0 = define_motion_list(DT);
bar0   = cases0{1}.factory(geo);

worst = 0;
for t = [DT, 30*DT, 60*DT]
    th_old = om_old * t;
    X_old  = [geo.xc + s_old * cos(th_old), geo.yc + s_old * sin(th_old)];
    V_old  = om_old * [-s_old * sin(th_old), s_old * cos(th_old)];
    m      = bar0.motion_fun(t);
    okX = isequal(typecast(m.X(:), 'uint64'), typecast(X_old(:), 'uint64'));
    okV = isequal(typecast(m.V(:), 'uint64'), typecast(V_old(:), 'uint64'));
    assert(okX && okV, ...
           'T1 bar at t=%.4f: not bit-identical to the original literals', t);
    worst = max(worst, max(abs(m.X(:) - X_old(:))));
end

% The disks consume no knobs; prove the params path leaves them untouched too.
cases1 = define_motion_list(DT, struct());
cases2 = define_motion_list(DT, struct('theta0', 0, 'Lb_frac', 0.35, ...
                                       'bar_pt_frac', 0.6));
for ci = 1:3
    for t = [DT, 60*DT]
        a = cases0{ci}.factory(geo).motion_fun(t);
        b = cases1{ci}.factory(geo).motion_fun(t);
        c = cases2{ci}.factory(geo).motion_fun(t);
        assert(isequal(typecast(a.X(:), 'uint64'), typecast(b.X(:), 'uint64')) && ...
               isequal(typecast(a.X(:), 'uint64'), typecast(c.X(:), 'uint64')) && ...
               isequal(typecast(a.V(:), 'uint64'), typecast(b.V(:), 'uint64')) && ...
               isequal(typecast(a.V(:), 'uint64'), typecast(c.V(:), 'uint64')), ...
               'T1 %s: no-arg / struct() / explicit-defaults disagree at t=%.4f', ...
               cases0{ci}.name, t);
    end
end
npass = npass + 1;
fprintf('  PASS T1: defaults bit-identical, all 3 cases (max |dX| = %.1e)\n', worst);

%% ---- T2: nb never tips, including at the exact ceil boundaries ----------
% nb = max(8, ceil(2*Lb/(0.6*h0))) is an INTEGER that sizes nC.  A floating-point
% reassociation that moves it by one would be silent and catastrophic, so probe
% the exact h0 where ceil changes value, and one ulp either side.
hs = [];
for k = 8:60
    ht = 0.7 / (0.6 * k);
    hs = [hs, ht, ht - eps(ht), ht + eps(ht)];      %#ok<AGROW>
end
hs = [hs, linspace(0.005, 0.3, 200)];
bad  = 0;
csT2 = define_motion_list(DT);          % registry is geo-independent; hoist it
for hh = hs
    g2 = geo;  g2.h0 = hh;
    Lb_o  = 0.35 * (g2.y2 - g2.y1);
    nb_o  = max(8, ceil(2 * Lb_o / (0.6 * g2.h0)));
    barh  = csT2{1}.factory(g2);
    mh    = barh.motion_fun(DT);
    if size(mh.X, 1) ~= nb_o, bad = bad + 1; end
end
assert(bad == 0, 'T2: nb differs from the original at %d of %d h0 values', ...
       bad, numel(hs));
npass = npass + 1;
fprintf('  PASS T2: nb identical at %d h0 values incl. every ceil boundary\n', ...
        numel(hs));

%% ---- T3: params_tag is empty for defaults, distinct otherwise -----------
t_def  = define_motion_list(DT);
t_emp  = define_motion_list(DT, struct());
t_exp  = define_motion_list(DT, struct('theta0', 0, 'Lb_frac', 0.35, 'bar_pt_frac', 0.6));
assert(isempty(t_def{1}.params_tag) && isempty(t_emp{1}.params_tag) && ...
       isempty(t_exp{1}.params_tag), 'T3: default params_tag is not empty');

tA = define_motion_list(DT, struct('Lb_frac', 0.499));
tB = define_motion_list(DT, struct('theta0', 1.3647));
tC = define_motion_list(DT, struct('Lb_frac', 0.499, 'theta0', 1.3647));
tD = define_motion_list(DT, struct('theta0', 1.3647, 'Lb_frac', 0.499));
tags = {tA{1}.params_tag, tB{1}.params_tag, tC{1}.params_tag};
assert(all(~cellfun(@isempty, tags)), 'T3: non-default params_tag is empty');
assert(numel(unique(tags)) == 3, 'T3: non-default tags collide');
assert(strcmp(tC{1}.params_tag, tD{1}.params_tag), ...
       'T3: params_tag depends on field order');
for ci = 2:3
    assert(isempty(tC{ci}.params_tag), ...
           'T3: %s got a tag but consumes no knobs', tC{ci}.name);
end
% 0.4990 and 0.4991 must not alias onto one filename (num2str would).
tE = define_motion_list(DT, struct('Lb_frac', 0.4990));
tF = define_motion_list(DT, struct('Lb_frac', 0.4991));
assert(~strcmp(tE{1}.params_tag, tF{1}.params_tag), ...
       'T3: nearby sweep points alias onto one tag');
npass = npass + 1;
fprintf('  PASS T3: tag empty at defaults, distinct and order-free otherwise\n');

%% ---- T4/T5: bad input is rejected by identifier -------------------------
id = '';
try
    define_motion_list(DT, struct('Lb_fract', 0.4));   % typo
catch ME
    id = ME.identifier;
end
assert(strcmp(id, 'define_motion_list:unknownMotionParam'), ...
       'T4: typo not rejected (got "%s")', id);

id = '';
try
    define_motion_list(DT, struct('Lb_frac', 0.5));
catch ME
    id = ME.identifier;
end
assert(strcmp(id, 'define_motion_list:barLeavesChannel'), ...
       'T5: Lb_frac = 0.5 not rejected (got "%s")', id);
npass = npass + 2;
fprintf('  PASS T4/T5: unknown field and Lb_frac >= 0.5 rejected by identifier\n');

%% ---- T6: the offset that actually lands the bar vertical on STEP 1 ------
% Step n is evaluated at t = n*dt, so step 1 is t = dt.  omega*dt cancels dt:
%     omega*dt = 2*pi*nrev/Tmax * dt = 2*pi*nrev/Tstep.
theta0 = pi/2 - 2*pi*NREV/TSTEP;
barT   = define_motion_list(DT, struct('Lb_frac', 0.499, 'theta0', theta0));
mT     = barT{1}.factory(geo).motion_fun(DT);          % <- t = dt, i.e. step 1
offx   = max(abs(mT.X(:,1) - geo.xc));
assert(offx < 1e-12, 'T6: step 1 is not vertical, max |x - xc| = %.3e', offx);

% and the naive theta0 = pi/2 is NOT vertical at step 1 -- the trap this pins.
barN = define_motion_list(DT, struct('Lb_frac', 0.499, 'theta0', pi/2));
offn = max(abs(barN{1}.factory(geo).motion_fun(DT).X(:,1) - geo.xc));
assert(offn > 1e-3, ...
       'T6: theta0 = pi/2 came out vertical (%.3e); the step/time mapping moved', offn);

gap = min(min(mT.X(:,2) - geo.y1), min(geo.y2 - mT.X(:,2)));
assert(abs(gap - (0.5 - 0.499)*(geo.y2 - geo.y1)) < 1e-12, ...
       'T6: wall gap %.3e disagrees with (0.5 - Lb_frac)*H', gap);
npass = npass + 1;
fprintf(['  PASS T6: theta0 = pi/2 - 2*pi*nrev/Tstep puts step 1 vertical ' ...
         '(gap %.4f = %.2f*h0); naive pi/2 misses by %.3e\n'], ...
        gap, gap/geo.h0, offn);

%% ---- T7/T8: recorded on S, and a sweep does not thrash one cache file ---
TAG_D = regexprep(sprintf('seq_bar_rotating_h%s_dt%s_n%d', num2str(H_TINY), ...
                          num2str(DT), NS), '[^\w]', '_');
th2   = pi/2 - 2*pi*NREV/TSTEP;
csSfx = define_motion_list(DT, struct('theta0', th2));
sfx   = csSfx{1}.params_tag;
TAG_T = regexprep(sprintf('seq_bar_rotating_h%s_dt%s_n%d%s', num2str(H_TINY), ...
                          num2str(DT), NS, sfx), '[^\w]', '_');
F_D = fullfile(p.cacheDir, [TAG_D '.mat']);
F_T = fullfile(p.cacheDir, [TAG_T '.mat']);
if exist(F_D, 'file') == 2, delete(F_D); end
if exist(F_T, 'file') == 2, delete(F_T); end

oD = struct('case_name', 'bar_rotating', 'h0', H_TINY, 'dt', DT, 'nsteps', NS, ...
            'use_cache', true, 'verify', false, 'quiet', true);
oT = oD;  oT.motion_params = struct('theta0', th2);

S_D = build_stokes_sequence(oD);
S_T = build_stokes_sequence(oT);
assert(isfield(S_D, 'motion_params') && isfield(S_D, 'motion_meta'), ...
       'T7: motion_params/motion_meta missing from a fresh build');
assert(S_D.motion_params.theta0 == 0 && abs(S_T.motion_params.theta0 - th2) < 1e-15, ...
       'T7: recorded theta0 is wrong');
assert(exist(F_D, 'file') == 2 && exist(F_T, 'file') == 2 && ~strcmp(F_D, F_T), ...
       'T8: the two sweep points did not get separate cache files');

% Both must now hit CLEANLY -- no staleCache, no rebuild.  That is the whole
% point of the tag suffix: without it these two share a filename and each build
% would evict the other.
lastwarn('');
S_D2 = build_stokes_sequence(oD);
[~, w1] = lastwarn;
lastwarn('');
S_T2 = build_stokes_sequence(oT);
[~, w2] = lastwarn;
assert(isempty(w1) && isempty(w2), ...
       'T8: a matching hit warned ("%s" / "%s") -- the sweep is thrashing', w1, w2);
assert(isfield(S_D2, 'motion_params') && isfield(S_T2, 'motion_meta'), ...
       'T7: motion_params/motion_meta missing after a CACHE HIT');
assert(abs(S_T2.motion_params.theta0 - th2) < 1e-15, ...
       'T7: cache hit stamped the wrong motion_params');
npass = npass + 2;
fprintf('  PASS T7/T8: recorded on build and on hit; sweep points get separate files\n');

if exist(F_D, 'file') == 2, delete(F_D); end
if exist(F_T, 'file') == 2, delete(F_T); end

fprintf('=== test_motion_params: %d checks passed ===\n', npass);
