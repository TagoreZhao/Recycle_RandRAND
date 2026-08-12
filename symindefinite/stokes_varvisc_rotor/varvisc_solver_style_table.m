function sty = varvisc_solver_style_table(n)
%SOLVER_STYLE_TABLE  Distinct line styles for N overlaid solver curves.
%
%   STY = SOLVER_STYLE_TABLE(N)  returns an N x 1 struct array with fields
%       .color      1x3 RGB
%       .marker     char
%       .linestyle  char
%       .linewidth  scalar
%       .markersize scalar
%
%   Rows 1-8 are the Okabe-Ito colourblind-safe palette (yellow dropped --
%   illegible on white); rows 9-11 extend it because the registry outgrew eight
%   solvers.  They differ in colour AND marker AND linestyle, because curves can
%   genuinely coincide (e.g. two_level_sjlt vs two_level_gaussian on quiet cases).
%
%   EXTENDING is safe by construction.  The palette size enters only as
%   mod(s-1, nb) and floor((s-1)/nb), both of which are unchanged for s <= 8 when
%   nb grows, so appending rows leaves every existing curve's style bit-identical
%   and only affects solvers past the eighth.
%
%   LINEWIDTH decreases monotonically down the table.  Curves are drawn in
%   registry order, so a later thin line lands on top of an earlier thick one
%   and coincident curves render as concentric bands instead of one erasing the
%   others.  Staggered markers (see varvisc_plot_solver_curves) separate them further.
%
%   Beyond eleven the table cycles: colours advance one step out of phase with
%   markers, so the repeat period is 11*10 rather than 11.
%
%   See also: varvisc_plot_solver_curves, varvisc_mark_coincident_curves.

    validateattributes(n, {'numeric'}, {'scalar', 'integer', 'nonnegative'}, ...
                       mfilename, 'n');

    colors = [0.00 0.00 0.00;    % black
              0.84 0.37 0.00;    % vermillion
              0.00 0.45 0.70;    % blue
              0.00 0.62 0.45;    % bluish green
              0.80 0.47 0.65;    % reddish purple
              0.90 0.62 0.00;    % orange
              0.34 0.71 0.91;    % sky blue
              0.60 0.60 0.60;    % grey
              0.45 0.16 0.51;    % violet     (extension, see note above)
              0.40 0.50 0.15;    % olive      (extension)
              0.55 0.05 0.15];   % maroon     (extension)
    markers    = {'o', 's', '^', 'd', 'v', 'p', 'h', '>', '<', '*'};
    % Only four line styles exist, so rows must repeat one; adjacent rows are
    % kept different in colour, marker AND line style so coincident curves stay
    % readable as concentric bands.
    linestyles = {'-', '--', '-.', '-', '--', '-.', ':', ':', '-', ':', '-'};
    linewidths = [2.2 2.0 1.9 1.8 1.6 1.5 1.4 1.1 1.0 0.9 0.85];
    markersizes = [5 5 5 6 6 6 6 6 6 6 6];

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
