function files = replot_woodbury(outDir, opts)
%REPLOT_WOODBURY  Regenerate every figure from the committed CSVs, solving nothing.
%   FILES = REPLOT_WOODBURY()            % the default output dir
%   FILES = REPLOT_WOODBURY(OUTDIR)
%   FILES = REPLOT_WOODBURY(OUTDIR, OPTS) % woodbury_fig_defaults overrides
%
%   Figure tweaks -- a colour, an axis label, a legend position -- must never
%   require re-running the benchmark.  write_woodbury_outputs writes every
%   quantity any figure needs into <case>_results.csv, and
%   write_woodbury_figures reads only those files, so this is the whole redraw
%   path.
%
%   Example:  replot_woodbury([], struct('resolution', 300))
%
%   See also: write_woodbury_figures, write_woodbury_outputs.

    paths = add_woodbury_paths();
    assert_woodbury_helpers();

    if nargin < 1 || isempty(outDir), outDir = paths.outDir; end
    if nargin < 2, opts = struct(); end

    if ~exist(outDir, 'dir')
        error('replot_woodbury:noDir', ...
              'Output directory does not exist: %s', outDir);
    end

    fprintf('Replotting from %s\n', outDir);
    files = write_woodbury_figures(outDir, opts);
    fprintf('Regenerated %d figure(s).\n', numel(files));
end
