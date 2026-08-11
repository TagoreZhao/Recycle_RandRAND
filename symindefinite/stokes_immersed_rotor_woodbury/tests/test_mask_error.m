%TEST_MASK_ERROR  The rule that decides which forward errors this study reports.
%
%   Run:  cd tests; test_mask_error
%
%   woodbury_mask_error encodes one claim: a forward error measured against an
%   imperfect reference is evidence only when it stands clear of that reference's
%   own error.  Everything the bad-reference study concludes from a forward error
%   passes through it, so the boundary cases are pinned here rather than left to
%   whatever a format string happened to do.
%
%   See also: woodbury_mask_error, run_woodbury_bad_reference.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
studyDir    = fileparts(thisFileDir);
addpath(studyDir);
add_woodbury_paths();

fprintf('=== test_mask_error ===\n');
np = 0;  nf = 0;

% --- M1  clear of the yardstick: reported plain ---------------------------
[t, r] = woodbury_mask_error(1e-6, 1e-12);
[np, nf] = chk(np, nf, sprintf('M1  err >> unc is reportable ("%s")', t), ...
               r && strcmp(t, '1.000e-06'));

% --- M2  swamped by the yardstick: printed, flagged, not reportable -------
[t, r] = woodbury_mask_error(2e-12, 1e-12);
[np, nf] = chk(np, nf, ...
    sprintf('M2  err within 10x of unc is flagged, not suppressed ("%s")', t), ...
    ~r && strcmp(t, '2.000e-12~'));

% --- M3  the boundary is strict and sits exactly at factor*unc ------------
[~, rlo] = woodbury_mask_error(10e-12, 1e-12);      % exactly 10x: NOT reportable
[~, rhi] = woodbury_mask_error(10.1e-12, 1e-12);
[np, nf] = chk(np, nf, 'M3  boundary: err == 10*unc excluded, just above included', ...
               ~rlo && rhi);

% --- M4  the factor is honoured -------------------------------------------
[~, r3] = woodbury_mask_error(5e-12, 1e-12, 3);
[~, r9] = woodbury_mask_error(5e-12, 1e-12, 9);
[np, nf] = chk(np, nf, 'M4  FACTOR moves the boundary', r3 && ~r9);

% --- M5  no yardstick means nothing to be swamped by ----------------------
% unc = 0 arises when a reference is exact by construction (the manufactured-RHS
% arm), and NaN when the uncertainty probe itself failed.  Both must leave a
% finite error reportable rather than silently masking the arm that has the best
% ground truth in the study.
[~, r0] = woodbury_mask_error(1e-16, 0);
[~, rn] = woodbury_mask_error(1e-16, NaN);
[~, ri] = woodbury_mask_error(1e-16, Inf);
[np, nf] = chk(np, nf, 'M5  unc in {0, NaN, Inf} leaves a finite err reportable', ...
               r0 && rn && ri);

% --- M6  a non-finite error is never evidence -----------------------------
[t1, ra] = woodbury_mask_error(NaN, 1e-12);
[~,  rb] = woodbury_mask_error(Inf, 0);
[np, nf] = chk(np, nf, sprintf('M6  non-finite err prints "%s" and is never reportable', t1), ...
               strcmp(t1, 'n/a') && ~ra && ~rb);

% --- M7  bad input is refused rather than formatted -----------------------
bad  = {{[1 2], 1e-12}, {1e-6, [1 2]}, {1e-6, 1e-12, 0}, {1e-6, 1e-12, -1}};
okM7 = true;
for k = 1:numel(bad)
    try
        woodbury_mask_error(bad{k}{:});
        okM7 = false;
    catch
        % expected
    end
end
[np, nf] = chk(np, nf, 'M7  non-scalar err/unc and non-positive factor each throw', okM7);

fprintf('\n  %d passed, %d failed\n', np, nf);
if nf > 0
    error('test_mask_error:fail', '%d assertion(s) failed.', nf);
end

%==========================================================================
function [np, nf] = chk(np, nf, name, cond)
    if cond
        np = np + 1;
        fprintf('  PASS  %s\n', name);
    else
        nf = nf + 1;
        fprintf(2, '  FAIL  %s\n', name);
    end
end
