function assert_varvisc_schur_helpers()
%ASSERT_VARVISC_SCHUR_HELPERS  Reject missing or shadowed local helpers.

    thisDir = fileparts(mfilename('fullpath'));
    localNames = { ...
        'varvisc_schur_assemble_kkt', 'varvisc_schur_context_init', ...
        'varvisc_schur_step_operator', 'varvisc_schur_make_cfg', ...
        'make_varvisc_schur_params', 'add_varvisc_schur_paths'};

    for i = 1:numel(localNames)
        resolved = which(localNames{i});
        if isempty(resolved) || ~strncmp(resolved, thisDir, numel(thisDir))
            error('assert_varvisc_schur_helpers:shadowed', ...
                  'Expected %s under %s, resolved to %s.', ...
                  localNames{i}, thisDir, resolved);
        end
    end

    resolved = which('varvisc_define_case_list');
    if isempty(resolved)
        error('assert_varvisc_schur_helpers:noCaseList', ...
              'varvisc_define_case_list is unavailable; run add_varvisc_schur_paths.');
    end
end
