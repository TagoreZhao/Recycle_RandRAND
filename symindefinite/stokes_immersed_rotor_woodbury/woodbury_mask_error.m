function [txt, reportable] = woodbury_mask_error(err, unc, factor)
%WOODBURY_MASK_ERROR  Format a forward error, flagged when its reference is not
%   good enough to have measured it.
%
%   [TXT, REPORTABLE] = WOODBURY_MASK_ERROR(ERR, UNC)
%   [TXT, REPORTABLE] = WOODBURY_MASK_ERROR(ERR, UNC, FACTOR)   default FACTOR = 10
%
%   ERR is a relative forward error measured against a reference solution; UNC is
%   an estimate of that reference's OWN error.  A forward error is only evidence
%   about the method under test when it stands clear of the yardstick:
%
%       reportable  <=>  ERR > FACTOR * UNC
%
%   WHY THIS EXISTS AS A FUNCTION.  The whole difficulty in the bad-reference
%   study is that stressing the operator degrades the reference too, so "Woodbury
%   is wrong by 1e-12" can mean "and K_n\b is wrong by 3e-12", which is no result
%   at all.  Printing a masked number is a claim about what the run measured, so
%   it is a rule that deserves a test rather than a format string inlined at four
%   call sites.  A flagged value is still PRINTED -- suppressing it would hide the
%   fact that a point was taken -- but it carries a trailing '~' and REPORTABLE is
%   false, so a caller can exclude it from a fit or a headline series.
%
%   Non-finite ERR prints 'n/a' and is never reportable; a zero or non-finite UNC
%   is treated as no information about the reference, which makes any finite ERR
%   reportable (there is nothing to be swamped by).
%
%   See also: run_woodbury_bad_reference.

    if nargin < 3 || isempty(factor), factor = 10; end
    validateattributes(err, {'numeric'}, {'scalar', 'real'}, mfilename, 'err', 1);
    validateattributes(unc, {'numeric'}, {'scalar', 'real'}, mfilename, 'unc', 2);
    validateattributes(factor, {'numeric'}, {'scalar', 'real', 'positive'}, ...
                       mfilename, 'factor', 3);

    if ~isfinite(err)
        txt        = 'n/a';
        reportable = false;
        return;
    end

    if ~isfinite(unc) || unc <= 0
        reportable = true;                 % no yardstick to be swamped by
    else
        reportable = err > factor * unc;
    end

    if reportable
        txt = sprintf('%.3e', err);
    else
        txt = sprintf('%.3e~', err);       % '~' == reference-limited
    end
end
