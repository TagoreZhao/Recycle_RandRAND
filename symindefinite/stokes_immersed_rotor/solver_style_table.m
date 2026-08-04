function sty = solver_style_table(n)
%SOLVER_STYLE_TABLE  Distinct line styles for N overlaid solver curves.
%
%   STY = SOLVER_STYLE_TABLE(N)  returns an N x 1 struct array with fields
%       .color      1x3 RGB
%       .marker     char
%       .linestyle  char
%       .linewidth  scalar
%       .markersize scalar
%
%   The eight base rows are the Okabe-Ito colourblind-safe palette (yellow
%   dropped -- illegible on white) and differ in colour AND marker AND
%   linestyle, because in this benchmark curves genuinely coincide: with
%   DEFLAT_RECYCLE_K = 0 the two_level_krylov iteration counts are bit-identical
%   to two_level_gaussian, and two_level_sjlt is within one iteration of both.
%
%   LINEWIDTH decreases monotonically down the table.  Curves are drawn in
%   registry order, so a later thin line lands on top of an earlier thick one
%   and coincident curves render as concentric bands instead of one erasing the
%   others.  Staggered markers (see plot_solver_curves) separate them further.
%
%   Beyond eight the table cycles: colours advance one step out of phase with
%   markers, so the repeat period is 8*7 rather than 8.
%
%   See also: plot_solver_curves, mark_coincident_curves.

    validateattributes(n, {'numeric'}, {'scalar', 'integer', 'nonnegative'}, ...
                       mfilename, 'n');

    colors = [0.00 0.00 0.00;    % black
              0.84 0.37 0.00;    % vermillion
              0.00 0.45 0.70;    % blue
              0.00 0.62 0.45;    % bluish green
              0.80 0.47 0.65;    % reddish purple
              0.90 0.62 0.00;    % orange
              0.34 0.71 0.91;    % sky blue
              0.60 0.60 0.60];   % grey
    markers    = {'o', 's', '^', 'd', 'v', 'p', 'h', '>'};
    linestyles = {'-', '--', '-.', '-', '--', '-.', ':', ':'};
    linewidths = [2.2 2.0 1.9 1.8 1.6 1.5 1.4 1.1];
    markersizes = [5 5 5 6 6 6 6 6];

    nb = size(colors, 1);
    sty = repmat(struct('color', [0 0 0], 'marker', 'o', 'linestyle', '-', ...
                        'linewidth', 1.5, 'markersize', 5), n, 1);
    for s = 1:n
        ic = mod(s - 1, nb) + 1;
        im = mod(s - 1 + floor((s - 1) / nb), numel(markers)) + 1;
        sty(s).color      = colors(ic, :);
        sty(s).marker     = markers{im};
        sty(s).linestyle  = linestyles{ic};
        sty(s).linewidth  = linewidths(ic);
        sty(s).markersize = markersizes(ic);
    end
end
