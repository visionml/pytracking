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
        self.tracker_class = tracker_module.get_tracker_class()


    def run(self, seq, visualization=None, debug=None, visdom_info=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
        """
        visdom_info = {} if visdom_info is None else visdom_info
        params = self.get_parameters()
        visualization_ = visualization

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        if visualization is None:
            if debug is None:
                visualization_ = getattr(params, 'visualization', False)
            else:
                visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_
        params.visdom_info = visdom_info

        tracker = self.tracker_class(params)

        output = tracker.track_sequence(seq)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        visdom_info = {} if visdom_info is None else visdom_info

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_
        params.visdom_info = visdom_info

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        tracker = self.tracker_class(params)
        tracker.track_videofile(videofilepath, optional_box)

    def run_webcam(self, debug=None, visdom_info=None):
        """Run the tracker with the webcam.
        args:
            debug: Debug level.
        """
        visdom_info = {} if visdom_info is None else visdom_info
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.visdom_info = visdom_info

        tracker = self.tracker_class(params)

        tracker.track_webcam()

    def run_vot(self, debug=None, visdom_info=None):
        """ Run on vot"""
        visdom_info = {} if visdom_info is None else visdom_info
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.run_id = self.run_id
        params.visdom_info = visdom_info

        tracker = self.tracker_class(params)
        tracker.initialize_features()
        tracker.track_vot()

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


