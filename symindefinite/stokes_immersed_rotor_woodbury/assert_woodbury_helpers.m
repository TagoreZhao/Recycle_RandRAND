function assert_woodbury_helpers()
%ASSERT_WOODBURY_HELPERS  Fail loudly if anything shadows this study's helpers.
%   ASSERT_WOODBURY_HELPERS()
%
%   Every helper in this folder is named woodbury_* / *_woodbury precisely so that
%   no sibling file can collide with it (see the hazard note in
%   add_woodbury_paths).  This check is therefore expected to be trivially
%   satisfied -- it is kept because the cost is microseconds and the failure it
%   guards against is silent: a shadowed helper would produce plausible numbers
%   from the wrong operator.
%
%   NOTE ON ITS OWN NAME.  The Schur twin already ships an assert_local_helpers.m;
%   this one is assert_WOODBURY_helpers so the guard cannot itself be the thing
%   that gets shadowed.  It checks its own name first, for the same reason.
%
%   It also pins define_motion_list to the SIBLING folder, which is the one file
%   this study reuses by path rather than copying.
%
%   See also: add_woodbury_paths.

    thisFileDir = fileparts(mfilename('fullpath'));
    siblingDir  = fullfile(fileparts(thisFileDir), 'stokes_immersed_rotor');

    % REQUIRED: the solver core.  Nothing in this study means anything without it,
    % and a missing piece here should not be discovered halfway through a run.
    required_names = { ...
        'assert_woodbury_helpers', 'add_woodbury_paths', 'make_woodbury_params', ...
        'woodbury_context_init', 'woodbury_solve'};

    % CHECKED IF PRESENT: the engine and output layer.  Absence is left to the
    % caller's own "Undefined function" error, which is already unambiguous;
    % what must never happen is one of these resolving to somebody else's file.
    optional_names = { ...
        'solve_woodbury_sequence', 'run_woodbury_benchmark', ...
        'write_woodbury_outputs', 'write_woodbury_figures', ...
        'replot_woodbury', 'woodbury_fig_defaults', ...
        'save_woodbury_figure', 'woodbury_style_table', ...
        'woodbury_naive', 'dd_woodbury_scalar', ...
        'woodbury_chain_build', 'woodbury_chain_apply', ...
        'run_woodbury_scalar_stress', 'run_woodbury_recursive', ...
        'run_woodbury_bad_reference', 'woodbury_mask_error'};

    for i = 1:numel(required_names)
        if isempty(which(required_names{i}))
            error('assert_woodbury_helpers:missing', ...
                  'Required helper "%s" is not on the path.', required_names{i});
        end
    end

    all_names = [required_names, optional_names];
    for i = 1:numel(all_names)
        w = which(all_names{i});
        if isempty(w)
            continue;                       % optional and absent: not our problem
        end
        if ~strncmp(w, thisFileDir, numel(thisFileDir))
            error('assert_woodbury_helpers:shadowed', ...
                  ['Local helper "%s" is shadowed by:\n    %s\n' ...
                   'Expected it under:\n    %s\n' ...
                   'Re-run add_woodbury_paths() to restore path order.'], ...
                  all_names{i}, w, thisFileDir);
        end
    end

    % define_motion_list must resolve to the SIBLING (reused, never copied).
    w = which('define_motion_list');
    if isempty(w)
        error('assert_woodbury_helpers:noMotionList', ...
              'define_motion_list not found; add_woodbury_paths() was not run.');
    end
    if ~strncmp(w, siblingDir, numel(siblingDir))
        error('assert_woodbury_helpers:motionListNotSibling', ...
              ['define_motion_list resolved to:\n    %s\n' ...
               'It must come from the sibling benchmark:\n    %s\n' ...
               'A local copy would let the two studies drift apart.'], ...
              w, siblingDir);
    end

    % The sequence kernel supplies the verified rank-2nC form this study rests on.
    for f = {'build_stokes_sequence', 'seq_K', 'seq_dCblk'}
        if isempty(which(f{1}))
            error('assert_woodbury_helpers:noKernelFn', ...
                  '%s not found; add_woodbury_paths() was not run.', f{1});
        end
    end
end
