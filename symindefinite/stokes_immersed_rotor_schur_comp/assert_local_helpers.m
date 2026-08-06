function assert_local_helpers()
%ASSERT_LOCAL_HELPERS  Fail loudly if a sibling file shadows a local one.
%   ASSERT_LOCAL_HELPERS()
%
%   The sibling benchmark folder must be on the path for define_motion_list,
%   and it contains same-named helpers.  If path order ever flips, figures and
%   labels would silently come out wrong instead of erroring.  This turns that
%   silent failure into a hard one.

    thisFileDir = fileparts(mfilename('fullpath'));

    local_names = { ...
        'schur_assemble_kkt', 'schur_context_init', 'schur_step_operator', ...
        'schur_make_cfg', 'make_schur_params', 'add_schur_paths'};

    for i = 1:numel(local_names)
        w = which(local_names{i});
        if isempty(w)
            error('assert_local_helpers:missing', ...
                  'Local helper "%s" is not on the path.', local_names{i});
        end
        if ~strncmp(w, thisFileDir, numel(thisFileDir))
            error('assert_local_helpers:shadowed', ...
                  ['Local helper "%s" is shadowed by:\n    %s\n' ...
                   'Expected it under:\n    %s\n' ...
                   'Re-run add_schur_paths() to restore path order.'], ...
                  local_names{i}, w, thisFileDir);
        end
    end

    % define_motion_list must resolve to the SIBLING (it is reused, not copied).
    w = which('define_motion_list');
    if isempty(w)
        error('assert_local_helpers:noMotionList', ...
              'define_motion_list not found; add_schur_paths() was not run.');
    end
end
