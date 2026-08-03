function v = vrec(exp_name, claim, quantity, measured, predicted, ok)
%VREC  One row of the verdict table.
%
%   v = VREC(EXP, CLAIM, QUANTITY, MEASURED, PREDICTED, OK)
%
%   Every experiment returns an array of these and RUN_ALL concatenates them
%   into output/verdicts.csv, which is what section 6 of the document is built
%   from.  Recording PREDICTED next to MEASURED is the point: a claim that was
%   never given a falsifiable number is not evidence.
%
%   OK may be logical (PASS/FAIL) or NaN, which records the row as 'REPORT' --
%   a measured quantity with no pass criterion, used where the theory gives a
%   direction rather than a threshold.
%
%   See also: run_all.

    if islogical(ok)
        status = 'FAIL';
        if ok, status = 'PASS'; end
    else
        status = 'REPORT';
    end

    v = struct('experiment', string(exp_name), ...
               'claim',      string(claim), ...
               'quantity',   string(quantity), ...
               'measured',   string(fmt(measured)), ...
               'predicted',  string(predicted), ...
               'verdict',    string(status));
end

function s = fmt(x)
    if ischar(x) || isstring(x)
        s = char(x);
    elseif isscalar(x)
        s = sprintf('%.4g', x);
    else
        s = mat2str(x(:)', 4);
    end
end
