import importlib
import os
import pickle
from pytracking.evaluation.environment import env_settings


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
    """

    def __init__(self, name: str, parameter_name: str, run_id: int = None):
        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))

        self.parameters = self.get_parameters()
        self.tracker_class = tracker_module.get_tracker_class()

        self.default_visualization = getattr(self.parameters, 'visualization', False)
        self.default_debug = getattr(self.parameters, 'debug', 0)

    def run(self, seq, visualization=None, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
        """
        visualization_ = visualization
        debug_ = debug
        if debug is None:
            debug_ = self.default_debug
        if visualization is None:
            if debug is None:
                visualization_ = self.default_visualization
            else:
                visualization_ = True if debug else False

        self.parameters.visualization = visualization_
        self.parameters.debug = debug_

        tracker = self.tracker_class(self.parameters)

        output_bb, execution_times = tracker.track_sequence(seq)

        self.parameters.free_memory()

        return output_bb, execution_times
    def run_video(self, videofilepath, optional_box=None, debug=None):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        debug_ = debug
        if debug is None:
            debug_ = self.default_debug
        self.parameters.debug = debug_

        self.parameters.tracker_name = self.name
        self.parameters.param_name = self.parameter_name
        tracker = self.tracker_class(self.parameters)
        tracker.track_videofile(videofilepath, optional_box)

    def run_webcam(self, debug=None):
        """Run the tracker with the webcam.
        args:
            debug: Debug level.
        """

        debug_ = debug
        if debug is None:
            debug_ = self.default_debug
        self.parameters.debug = debug_

        self.parameters.tracker_name = self.name
        self.parameters.param_name = self.parameter_name
        tracker = self.tracker_class(self.parameters)

        tracker.track_webcam()

    def get_parameters(self):
        """Get parameters."""

        parameter_file = '{}/parameters.pkl'.format(self.results_dir)
        if os.path.isfile(parameter_file):
            return pickle.load(open(parameter_file, 'rb'))

        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()

        if self.run_id is not None:
            pickle.dump(params, open(parameter_file, 'wb'))

        return params


