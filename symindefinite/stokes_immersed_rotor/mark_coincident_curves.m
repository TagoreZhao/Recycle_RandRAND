function tags = mark_coincident_curves(stats)
%MARK_COINCIDENT_CURVES  Flag solvers whose iteration curve duplicates another.
%
%   TAGS = MARK_COINCIDENT_CURVES(STATS)  returns an n x 1 cellstr.  TAGS{s} is
%   '' unless solver s produced exactly the same iteration counts as an EARLIER
%   solver, in which case it is ' (= <short label of that solver>)'.
%
%   Appended to the legend entry, this turns "one of the curves is invisible"
%   into a stated fact on the figure.  It fires for real here: at
%   DEFLAT_RECYCLE_K = 0 the recycling solver has nothing to recycle, so
%   two_level_krylov reproduces two_level_gaussian exactly and its curve is
%   hidden underneath no matter how the styles are chosen.
%
%   Only exact equality counts -- a one-iteration difference is a real
%   difference and the reader must not be told the curves are the same.
%
%   See also: solver_short_label, plot_solver_curves.

    keys = stats.solver_keys;
    n    = numel(keys);
    tags = repmat({''}, n, 1);
    for s = 2:n
        vs = stats.solver_its.(keys{s})(:);
        for r = 1:s-1
            vr = stats.solver_its.(keys{r})(:);
            if isequal(size(vs), size(vr)) && isequal(vs, vr)
                tags{s} = sprintf(' (= %s)', solver_short_label(keys{r}));
                break
            end
        end
    end
end
