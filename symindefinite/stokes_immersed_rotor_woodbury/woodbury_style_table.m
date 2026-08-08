function sty = woodbury_style_table(n)
%WOODBURY_STYLE_TABLE  Distinct line styles for N overlaid arm curves.
%
%   STY = WOODBURY_STYLE_TABLE(N)  returns an N x 1 struct array with fields
%       .color      1x3 RGB
%       .marker     char
%       .linestyle  char
%       .linewidth  scalar
%       .markersize scalar
%
%   The base rows are the Okabe-Ito colourblind-safe palette (yellow dropped --
%   illegible on white) and differ in colour AND marker AND linestyle, because in
%   this study curves genuinely coincide: on disk_static the woodbury and frozen
%   arms are the same solve to machine precision and sit exactly on top of each
%   other.
%
%   LINEWIDTH decreases monotonically down the table.  Curves are drawn in
%   registry order, so a later thin line lands on top of an earlier thick one and
%   coincident curves render as concentric bands instead of one erasing the other.
%
%   LOCAL COPY of the sibling benchmarks' solver_style_table, RENAMED for the
%   shadowing reason documented in add_woodbury_paths.  Twelve colours are kept
%   even though this study registers three arms, so that adding arms later cannot
%   silently wrap the palette and give two of them the same colour.
%
%   See also: write_woodbury_figures, woodbury_fig_defaults.

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
              0.55 0.00 0.00;    % dark red
              0.00 0.35 0.35;    % dark teal
              0.45 0.30 0.65;    % violet
              0.40 0.40 0.00];   % olive
    markers    = {'o', 's', '^', 'd', 'v', 'p', 'h', '>', '<', '*', 'x', '+'};
    linestyles = {'-', '--', '-.', '-', '--', '-.', ':', ':', ...
                  '-', '--', '-.', ':'};
    linewidths  = [2.2 2.0 1.9 1.8 1.6 1.5 1.4 1.1 2.0 1.8 1.6 1.4];
    markersizes = [5 5 5 6 6 6 6 6 6 6 6 6];

    nb  = size(colors, 1);
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
