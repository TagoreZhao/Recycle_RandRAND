function run_varvisc_augmented_benchmark(mode, case_index)
%RUN_VARVISC_AUGMENTED_BENCHMARK Matched full run or one small integration test.
% run_varvisc_augmented_benchmark('full')   three cases, h=.05, 1000+m
% run_varvisc_augmented_benchmark('smoke')  one case, h=.16, 16+m
% run_varvisc_augmented_benchmark('full',2) run/checkpoint just case 2
% run_varvisc_augmented_benchmark('finalize') render all completed cases
    if nargin < 1, mode = 'full'; end
    validatestring(mode,{'full','smoke','finalize'});
    here = fileparts(mfilename('fullpath'));
    VARVISC_OVERRIDES = struct('h0',0.05,'AUGMENT_ENABLED',true, ...
        'DEFLAT_SM_EIG',500,'SKETCH_OVERSAMPLE',2,'DEFLAT_Q',2, ...
        'AUGMENT_M',20,'AUGMENT_SWEEP',[10 40], ...
        'AUGMENT_SNAPSHOTS',[1 15 30 45 60],'AUGMENT_PROBE_DIM',200, ...
        'RESULTS_NAME','benchmark_varvisc_augmented', ...
        'RESET_RNG_EACH_CASE',true,'RESUME_COMPLETED_CASES',true);
    if nargin > 1
        validateattributes(case_index,{'numeric'},{'scalar','integer','>=',1,'<=',3});
        VARVISC_OVERRIDES.RUN_CASE_INDEX = case_index;
    end
    if strcmp(mode,'finalize'), VARVISC_OVERRIDES.POSTPROCESS_ONLY = true; end
    if strcmp(mode,'smoke')
        VARVISC_OVERRIDES.RESUME_COMPLETED_CASES = false;
        VARVISC_OVERRIDES.h0 = 0.16;
        VARVISC_OVERRIDES.DEFLAT_SM_EIG = 8;
        VARVISC_OVERRIDES.Tstep = 4;
        VARVISC_OVERRIDES.PHYSICAL_TMAX = 1.2;
        VARVISC_OVERRIDES.CASE_NAMES = {'bar_rotating_nu_orbiting'};
        VARVISC_OVERRIDES.AUGMENT_SNAPSHOTS = [1 3];
        VARVISC_OVERRIDES.AUGMENT_PROBE_DIM = 40;
        VARVISC_OVERRIDES.RESULTS_NAME = 'benchmark_varvisc_augmented_smoke';
    end
    run(fullfile(here,'run_varvisc_benchmark.m'));
end
