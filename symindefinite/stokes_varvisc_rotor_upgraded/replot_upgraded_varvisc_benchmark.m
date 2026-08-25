function replot_upgraded_varvisc_benchmark(varargin)
%REPLOT_UPGRADED_VARVISC_BENCHMARK Redraw retained upgraded benchmark plots.
    here = fileparts(mfilename('fullpath'));
    parent = fullfile(fileparts(here), 'stokes_varvisc_rotor');
    addpath(parent);
    results_root = fullfile(here, 'benchmark_varvisc_upgraded');
    replot_varvisc_benchmark(results_root, varargin{:});
    if is_dry_run(varargin), return; end
    varvisc_write_upgraded_linear_iteration_figures(results_root, []);
end

function value = is_dry_run(args)
    value = false;
    for k = 1:2:numel(args)-1
        if (ischar(args{k}) || isstring(args{k})) && ...
                strcmpi(string(args{k}), "DryRun")
            value = logical(args{k + 1});
            return
        end
    end
end
