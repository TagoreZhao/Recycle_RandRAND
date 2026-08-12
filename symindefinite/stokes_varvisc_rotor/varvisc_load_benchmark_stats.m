function [all_stats, cfg] = varvisc_load_benchmark_stats(results_root)
%LOAD_BENCHMARK_STATS  Rebuild per-case stats from a finished benchmark run.
%
%   [ALL_STATS, CFG] = LOAD_BENCHMARK_STATS(RESULTS_ROOT)
%
%   Reads <RESULTS_ROOT>/all_results.csv plus run_config.mat / run_config.json
%   and returns the same struct shape solve_stokes_varvisc produces, so the
%   figure writers cannot tell which path they are being driven from.  This is
%   what makes replotting possible without re-solving.
%
%   ALL_STATS{k} has:
%       .case_name .geometry .dt
%       .solver_keys .solver_labels
%       .solver_its.(key) .solver_flag.(key)
%       .coupling_change .constraint_res .relres .backslash_relres .nC
%       .diffK .nu_contrast .dK_nnz_frac
%       .solver_err.(last key)   only when the run recorded solver_err_last
%
%   CFG has .params .geometry .case_names .solver_keys .solver_labels.
%
%   Config sources are tried in order of trustworthiness: run_config.mat (exact
%   cell shapes), then run_config.json, then the CSV columns alone -- older
%   results directories may have any subset of these.
%
%   See also: varvisc_write_all_results_csv, replot_varvisc_benchmark, varvisc_write_case_figures.

    in_csv = fullfile(results_root, 'all_results.csv');
    if ~exist(in_csv, 'file')
        error('varvisc_load_benchmark_stats:noCsv', ...
              'No all_results.csv in %s', results_root);
    end
    T  = readtable(in_csv);
    vn = T.Properties.VariableNames;

    cfg = read_run_config(results_root);

    % --- solver keys: config order when available, else CSV column order -----
    csv_keys = erase(vn(endsWith(vn, '_its')), '_its');
    if isfield(cfg, 'solver_keys') && ~isempty(cfg.solver_keys)
        keys = as_cellstr(cfg.solver_keys);
        keys = keys(ismember(keys, csv_keys));       % tolerate a stale config
        keys = [keys(:); setdiff(csv_keys(:), keys(:), 'stable')];
    else
        keys = csv_keys(:);
    end
    if isempty(keys)
        error('varvisc_load_benchmark_stats:noSolvers', ...
              'No <key>_its columns in %s', in_csv);
    end

    if isfield(cfg, 'solver_labels') && numel(as_cellstr(cfg.solver_labels)) == numel(keys)
        labels = as_cellstr(cfg.solver_labels);
    else
        labels = keys;
    end

    % --- case order: config order when available, else first appearance ------
    csv_cases = unique(string(T.case_name), 'stable');
    if isfield(cfg, 'case_names') && ~isempty(cfg.case_names)
        cases = string(as_cellstr(cfg.case_names));
        cases = cases(ismember(cases, csv_cases));
        cases = [cases(:); setdiff(csv_cases(:), cases(:), 'stable')];
    else
        cases = csv_cases(:);
    end

    dt = 1;
    if isfield(cfg, 'params') && isstruct(cfg.params) && isfield(cfg.params, 'dt')
        dt = cfg.params.dt;
    end
    geometry = 'unknown';
    if isfield(cfg, 'geometry') && ~isempty(cfg.geometry)
        geometry = char(string(cfg.geometry));
    elseif ismember('geometry', vn)
        geometry = char(string(T.geometry(1)));
    end

    all_stats = cell(numel(cases), 1);
    for k = 1:numel(cases)
        rows = string(T.case_name) == cases(k);
        Tk   = sortrows(T(rows, :), 'timestep');

        st = struct();
        st.case_name     = char(cases(k));
        st.geometry      = geometry;
        st.dt            = dt;
        st.solver_keys   = keys(:);
        st.solver_labels = labels(:);
        st.solver_its    = struct();
        st.solver_flag   = struct();
        for s = 1:numel(keys)
            st.solver_its.(keys{s})  = Tk.([keys{s} '_its']);
            flagCol = [keys{s} '_flag'];
            if ismember(flagCol, vn)
                st.solver_flag.(keys{s}) = Tk.(flagCol);
            else
                st.solver_flag.(keys{s}) = zeros(height(Tk), 1);
            end
        end
        st.coupling_change  = col(Tk, 'diffF',            vn, height(Tk));
        st.constraint_res   = col(Tk, 'constraint_res',   vn, height(Tk));
        st.relres           = col(Tk, 'relres',           vn, height(Tk));
        st.backslash_relres = col(Tk, 'backslash_relres', vn, height(Tk));
        st.nC               = col(Tk, 'nC',               vn, height(Tk));
        st.diffK            = col(Tk, 'diffK',            vn, height(Tk));
        st.nu_contrast      = col(Tk, 'nu_contrast',      vn, height(Tk));
        st.dK_nnz_frac      = col(Tk, 'dK_nnz_frac',      vn, height(Tk));
        if ismember('solver_err_last', vn) && ~all(isnan(Tk.solver_err_last))
            st.solver_err.(keys{end}) = Tk.solver_err_last;
        end
        all_stats{k} = st;
    end

    % Report back what was actually used, so callers do not re-derive it.
    cfg.solver_keys   = keys(:);
    cfg.solver_labels = labels(:);
    cfg.case_names    = cellstr(cases(:));
    cfg.geometry      = geometry;
end

%==========================================================================
function v = col(Tk, name, vn, n)
    if ismember(name, vn), v = Tk.(name); else, v = nan(n, 1); end
end

%==========================================================================
function cfg = read_run_config(results_root)
    cfg = struct();
    matFile = fullfile(results_root, 'run_config.mat');
    if exist(matFile, 'file')
        S = load(matFile);
        if isfield(S, 'cfg_out') && isstruct(S.cfg_out)
            cfg = S.cfg_out;
            return
        end
    end
    jsonFile = fullfile(results_root, 'run_config.json');
    if exist(jsonFile, 'file')
        try
            cfg = jsondecode(fileread(jsonFile));
        catch err
            warning('varvisc_load_benchmark_stats:badJson', ...
                    'Could not parse %s (%s); falling back to CSV columns.', ...
                    jsonFile, err.message);
        end
    end
end

%==========================================================================
function c = as_cellstr(v)
%AS_CELLSTR  Normalise the several shapes jsondecode/load can hand back
% (cellstr, string array, char row, nested 1x1 cells) to a plain cellstr column.
    if isempty(v), c = {}; return; end
    if ischar(v),   c = {v};        return; end
    if isstring(v), c = cellstr(v(:)); return; end
    if iscell(v)
        c = cell(numel(v), 1);
        for i = 1:numel(v)
            vi = v{i};
            while iscell(vi) && isscalar(vi), vi = vi{1}; end
            c{i} = char(string(vi));
        end
        return
    end
    c = {char(string(v))};
end
