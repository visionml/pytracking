% Set path to the python in the pytracking conda environment
python_path = 'PATH_TO_CONDA_INSTALLATION/envs/pytracking/bin/python';

% Set path to pytracking
pytracking_path = 'PATH_TO_VISIONML/pytracking';

% Set path to trax installation. Check
% https://trax.readthedocs.io/en/latest/tutorial_compiling.html for
% compilation information
trax_path = 'PATH_TO_VOT_TOOLKIT/native/trax';

tracker_name = 'dimp';          % Name of the tracker to evaluate
runfile_name = 'dimp18_vot';    % Name of the parameter file to use
debug = 0;

%%
tracker_label = [tracker_name, '_', runfile_name];

% Generate python command
tracker_command = sprintf(['%s -c "import sys; sys.path.append(''%s'');', ...
                           'sys.path.append(''%s/support/python'');', ...
                           'import run_vot;', ...
                           'run_vot.run_vot(''%s'', ''%s'', debug=%d)"'],...
                           python_path, pytracking_path, trax_path, ...
                           tracker_name, runfile_name, debug);


tracker_interpreter = python_path;

tracker_linkpath = {[trax_path, '/build'],...
		[trax_path, '/build/support/client'],...
		[trax_path, '/build/support/opencv']};
