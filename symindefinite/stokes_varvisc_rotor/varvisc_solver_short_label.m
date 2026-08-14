function s = varvisc_solver_short_label(key)
%SOLVER_SHORT_LABEL  Registry key -> compact legend text.
%
%   S = SOLVER_SHORT_LABEL(KEY)   KEY char -> S char
%   S = SOLVER_SHORT_LABEL(KEYS)  KEYS cellstr -> S cellstr of the same size
%
%   The registry labels in varvisc_define_solver_list are up to 57 characters
%   ('MINRES (ILDL + deflation L^{-T}PL^{-1}, gaussian V)').  Ten of those in
%   one legend is wider than the axes, which is why the legend used to sit on
%   top of the data.  These names are <= 30 characters; the full label still
%   appears as the subtitle of the per-solver figure, and the shared
%   'MINRES / ILDL + deflation' part is carried by the figure title.
%
%   Keyed off the registry KEY, not the label, so the live path (labels in
%   memory) and the replot path (keys read back from run_config) agree.
%
%   See also: varvisc_plot_solver_curves, varvisc_mark_coincident_curves.

    if iscell(key)
        s = cellfun(@varvisc_solver_short_label, key, 'UniformOutput', false);
        return
    end
    if isstring(key), key = char(key); end
    validateattributes(key, {'char'}, {'row'}, mfilename, 'key');

    switch key
        case 'minres_unprec',        s = 'unpreconditioned';
        case 'block_jacobi',         s = 'block Jacobi (refreshed)';
        case 'block_jacobi_frozen',  s = 'block Jacobi (frozen)';
        case 'ildl_nofill',          s = 'ILDL (no-fill)';
        case 'exact_ldl_frozen',     s = 'exact LDL (frozen)';
        case 'gmres_exact_inv_frozen', s = 'GMRES: exact K_1^{-1}';
        case 'two_level_sjlt',       s = '2-level: sjlt V';
        case 'two_level_gaussian',   s = '2-level: gaussian V';
        case 'two_level_polynomial', s = '2-level: polynomial V';
        case 'two_level_exact',      s = '2-level: exact V';
        case 'two_level_esketch',    s = '2-level: C^{-1}BC^{-T} sketch V';
        otherwise
            % Unknown key: readable fallback so a newly registered solver still
            % plots without touching this file.
            s = strrep(key, '_', ' ');
            if startsWith(s, 'two level ')
                s = ['2-level: ' extractAfter(s, 'two level ')];
            end
    end
end
