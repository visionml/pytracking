import importlib
import os
import numpy as np
from collections import OrderedDict
from pytracking.evaluation.environment import env_settings
import time
import cv2 as cv
from pytracking.utils.visdom import Visdom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytracking.utils.plotting import draw_figure, overlay_mask
from pytracking.utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
from ltr.data.image_loader import imread_indexed


def trackerlist(name: str, parameter_name: str, run_ids=None, display_name: str = None):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, run_id, display_name) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, run_id: int = None, display_name: str = None):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            self.segmentation_dir = '{}/{}/{}'.format(env.segmentation_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
            self.segmentation_dir = '{}/{}/{}_{:03d}'.format(env.segmentation_path, self.name, self.parameter_name,
                                                             self.run_id)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker', self.name))
        if os.path.isdir(tracker_module_abspath):
            tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.visdom = None

    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info)

                # Show help
                help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                            'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                            'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                            'block list.'
                self.visdom.register(help_text, 'text', 1, 'Help')
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

    def create_tracker(self, params):
        tracker = self.tracker_class(params)
        tracker.visdom = self.visdom
        return tracker

    def run_sequence(self, seq, visualization=None, debug=None, visdom_info=None, multiobj_mode=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            visdom_info: Visdom info.
            multiobj_mode: Which mode to use for multiple objects.
        """
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

        self._init_visdom(visdom_info, debug_)
        if visualization_ and self.visdom is None:
            self.init_visualization()

        # Get init information
        init_info = seq.init_info()
        is_single_object = not seq.multiobj_mode  # init_info.get('object_ids') is None and len(init_info.get('object_ids')) == 1

        if multiobj_mode is None:
            multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default' or is_single_object:
            tracker = self.create_tracker(params)
            output = self._track_sequence_single(tracker, seq, init_info)
            # if 'segmentation' in output:
            #     self._insert_init_segmentation(output, params)

        elif multiobj_mode == 'sequential':
            if init_info['object_ids'] != init_info['init_object_ids']:
                raise NotImplementedError('Currently assumes all objects are initialized in the first frame')
            init_info_split = self._split_info(init_info)
            outputs = OrderedDict()
            for obj_id in init_info['object_ids']:
                tracker = self.create_tracker(params)
                outputs[obj_id] = self._track_sequence_single(tracker, seq, init_info_split[obj_id], obj_id)
                # if 'segmentation' in outputs[obj_id]:
                #     self._insert_init_segmentation(outputs[obj_id], params)
            output = self._merge_outputs(outputs, params, seq)

        elif multiobj_mode == 'parallel':
            outputs = self._track_sequence_multi(seq, params, init_info)
            # if 'segmentation' in outputs[obj_id]:
            #     self._insert_init_segmentation(outputs[obj_id], params)
            output = self._merge_outputs(outputs, params, seq)

        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        return output

    def _split_info(self, info):
        info_split = OrderedDict()
        init_other = OrderedDict()
        for obj_id in info['init_object_ids']:
            info_split[obj_id] = dict()
            init_other[obj_id] = dict()
            info_split[obj_id]['object_ids'] = [obj_id]
            info_split[obj_id]['sequence_object_ids'] = info['sequence_object_ids']
            if 'init_bbox' in info:
                info_split[obj_id]['init_bbox'] = info['init_bbox'][obj_id]
                init_other[obj_id]['init_bbox'] = info['init_bbox'][obj_id]
            if 'init_mask' in info:
                info_split[obj_id]['init_mask'] = (info['init_mask'] == int(obj_id)).astype(np.uint8)
                init_other[obj_id]['init_mask'] = info_split[obj_id]['init_mask']
        for obj_info in info_split.values():
            obj_info['init_other'] = init_other
        return info_split

    def _merge_outputs(self, outputs, params, seq):
        out_first = list(outputs.values())[0]
        out_types = out_first.keys()
        output = dict()
        output['start_frame'] = {obj_id: out.get('start_frame', 0) for obj_id, out in outputs.items()}
        if 'target_bbox' in out_types:
            output['target_bbox'] = {obj_id: out['target_bbox'] for obj_id, out in outputs.items()}

        if 'segmentation' in out_types:
            nb_frames = len(seq.frames)
            eg_seg_output = out_first['segmentation'][0]

            # Stack all masks
            segmentation_maps = [
                np.stack([out['segmentation'][i - out.get('start_frame', 0)] if i >= out.get('start_frame', 0)
                          else np.zeros_like(eg_seg_output) for out in outputs.values()])
                for i in range(nb_frames)]

            if params.get('return_raw_scores', False):
                segmentation_raw = [
                    np.stack([out['segmentation_raw'][i - out.get('start_frame', 0)] if i >= out.get('start_frame', 0)
                              else np.ones_like(eg_seg_output) * (-100.0) for out in outputs.values()])
                    for i in range(nb_frames)]

            # if tracker implements a merge function, use it to refine the individual score maps
            if hasattr(self.tracker_class, 'merge_segmentation_results'):
                # TODO
                raw_results = params.get('return_raw_scores', False)
                if raw_results:
                    segmentation_maps = [self.tracker_class.merge_segmentation_results(s, raw_results)
                                         for s in segmentation_raw]
                else:
                    segmentation_maps = [self.tracker_class.merge_segmentation_results(s, raw_results)
                                         for s in segmentation_maps]
                obj_ids = np.array([0, *map(int, outputs.keys())], dtype=np.uint8)
                segm_threshold = getattr(params, 'segm_threshold', 0.5)
                merged_segmentation = [obj_ids[s.argmax(axis=0)]
                                       for s in segmentation_maps]
            else:
                obj_ids = np.array([0, *map(int, outputs.keys())], dtype=np.uint8)
                segm_threshold = getattr(params, 'segm_threshold', 0.5)
                merged_segmentation = [obj_ids[np.where(s.max(axis=0) > segm_threshold, s.argmax(axis=0) + 1, 0)]
                                       for s in segmentation_maps]

            output['segmentation'] = merged_segmentation

        if 'time' in out_types:
            output['time'] = {obj_id: out['time'] for obj_id, out in outputs.items()}
        return output

    def _binarize_segmentation(self, segmentations, params):
        segm_threshold = getattr(params, 'segm_threshold', 0.5)
        return [(s > segm_threshold).astype(np.uint8) for s in segmentations]

    def _insert_init_segmentation(selfs, output, params):
        if len(output['segmentation']) == len(output['target_bbox']) - 1:
            init_segmentation = np.zeros_like(output['segmentation'][-1])
            output['segmentation'].insert(0, init_segmentation)

    def _track_sequence_single(self, tracker, seq, init_info, object_id=None):

        # Define outputs
        output = {'target_bbox': [],
                  'time': [],
                  'segmentation': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        if tracker.params.visualization and self.visdom is None:
            self.visualize(image, init_info.get('init_bbox'))

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'segmentation': init_info.get('init_mask')}

        _store_outputs(out, init_default)

        # if self.visdom is not None:
        #     tracker.visdom_draw_tracking(image, init_info.get('init_bbox'), init_info.get('init_mask'))

        # Track
        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                else:
                    time.sleep(0.1)

            image = self._read_image(frame_path)

            if getattr(seq, 'ground_truth_rect', None) is not None and \
                    (tracker.params.debug > 1 or getattr(tracker.params, 'use_gt_box', False)):
                if isinstance(seq.ground_truth_rect, (dict, OrderedDict)):
                    if object_id is not None:
                        tracker.gt_state = seq.ground_truth_rect[object_id][frame_num, :]
                else:
                    tracker.gt_state = seq.ground_truth_rect[frame_num, :]

            start_time = time.time()

            out = tracker.track(image, seq.frame_info(frame_num))

            _store_outputs(out, {'time': time.time() - start_time})

            segmentation = out['segmentation'] if 'segmentation' in out else None
            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, out['target_bbox'], segmentation)
            elif tracker.params.visualization:
                self.visualize(image, out['target_bbox'], segmentation)

        for key in ['target_bbox', 'segmentation']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def _track_sequence_multi(self, seq, params, init_info):

        object_ids = init_info['object_ids']

        # Define outputs
        outputs = OrderedDict({obj_id: {'target_bbox': [],
                                        'time': [],
                                        'segmentation': [],
                                        'segmentation_raw': []} for obj_id in object_ids})

        def _store_outputs(tracker_out: dict, obj_id, defaults=None, start_frame=None):
            defaults = {} if defaults is None else defaults
            output = outputs[obj_id]
            if start_frame is not None:
                output['start_frame'] = start_frame
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        # if tracker.params.visualization and self.visdom is None:
        #     self.visualize(image, seq.get('init_bbox'))

        init_info_split = self._split_info(init_info)
        trackers = OrderedDict({obj_id: self.create_tracker(params) for obj_id in object_ids})

        prev_output = OrderedDict()

        for obj_id in init_info['init_object_ids']:
            start_time = time.time()
            out = trackers[obj_id].initialize(image, init_info_split[obj_id])
            if out is None:
                out = {}

            init_default = {'target_bbox': init_info_split[obj_id].get('init_bbox'),
                            'time': time.time() - start_time,
                            'segmentation': init_info_split[obj_id].get('init_mask'),
                            'segmentation_raw': (init_info_split[obj_id].get('init_mask') - 0.5) * 200.0}

            _store_outputs(out, obj_id, init_default, start_frame=0)

            # Store previous output
            prev_output[obj_id] = out
            for key in ['target_bbox', 'segmentation', 'segmentation_raw']:
                if key not in out and init_default[key] is not None:
                    prev_output[obj_id][key] = init_default[key]

        initialized_ids = init_info['init_object_ids'].copy()

        if self.visdom is not None:
            boxes = [info['init_bbox'] for info in init_info_split.values()]
            data = [image] + boxes
            if 'init_mask' in init_info_split[object_ids[0]]:
                masks = [info['init_mask'] for info in init_info_split.values()]
                data.extend(masks)
            self.visdom.register(data, 'Tracking', 1, 'Tracking')

        # Track
        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                else:
                    time.sleep(0.1)

            image = self._read_image(frame_path)

            info = seq.frame_info(frame_num)

            if info.get('init_object_ids', False):
                init_info_split = self._split_info(info)
                for obj_init_info in init_info_split.values():
                    obj_init_info['previous_output'] = prev_output
                info['init_other'] = obj_init_info['init_other']

            info['previous_output'] = prev_output
            prev_output = OrderedDict()

            for obj_id in initialized_ids:
                start_time = time.time()

                if getattr(params, 'use_gt_box', False):
                    trackers[obj_id].gt_state = seq.ground_truth_rect[obj_id][frame_num]
                if getattr(params, 'use_gt_mask', False):
                    if seq.ground_truth_seg[frame_num] is not None:
                        gt_mask = imread_indexed(seq.ground_truth_seg[frame_num])
                        trackers[obj_id].gt_mask = (gt_mask == int(obj_id)).astype(np.uint8)
                    else:
                        trackers[obj_id].gt_mask = None
                out = trackers[obj_id].track(image, info)

                _store_outputs(out, obj_id, {'time': time.time() - start_time})

                prev_output[obj_id] = out

            # Initialize new
            if info.get('init_object_ids'):
                for obj_id in info['init_object_ids']:
                    start_time = time.time()
                    out = trackers[obj_id].initialize(image, init_info_split[obj_id])
                    if out is None:
                        out = {}

                    init_default = {'target_bbox': init_info_split[obj_id].get('init_bbox'),
                                    'time': time.time() - start_time,
                                    'segmentation': init_info_split[obj_id].get('init_mask'),
                                    'segmentation_raw': (init_info_split[obj_id].get('init_mask') - 0.5) * 200.0}

                    _store_outputs(out, obj_id, init_default, start_frame=frame_num)

                    # Store previous output
                    prev_output[obj_id] = out

                    for key in ['target_bbox', 'segmentation', 'segmentation_raw']:
                        if key not in out and init_default[key] is not None:
                            prev_output[obj_id][key] = init_default[key]

                initialized_ids.extend(info['init_object_ids'])

            # segmentation = out['segmentation'] if 'segmentation' in out else None

            if self.visdom is not None:
                boxes = [out['target_bbox'] for out in prev_output.values()]
                data = [image] + boxes
                if 'segmentation' in prev_output[object_ids[0]]:
                    masks = [out['segmentation'] for out in prev_output.values()]
                    data.extend(masks)
                self.visdom.register(data, 'Tracking', 1, 'Tracking')
            # elif tracker.params.visualization:
            #     self.visualize(image, out['target_bbox'], segmentation)

        for key in ['target_bbox', 'segmentation', 'segmentation_raw']:
            for output in outputs.values():
                if key in output and len(output[key]) <= 1:
                    output.pop(key)

        return outputs

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        self._init_visdom(visdom_info, debug_)
        tracker = self.create_tracker(params)

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        if hasattr(tracker, 'initialize_features'):
            tracker.initialize_features()

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)
        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, list, tuple)
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, {'init_bbox': optional_box})
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, {'init_bbox': init_state})
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                return

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, {'init_bbox': init_state})

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def run_webcam(self, debug=None, visdom_info=None):
        """Run the tracker with the webcam.
        args:
            debug: Debug level.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        self._init_visdom(visdom_info, debug_)

        tracker = self.create_tracker(params)

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.mode_switch = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                    self.mode_switch = True
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'track'
                    self.mode_switch = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        if hasattr(tracker, 'initialize_features'):
            tracker.initialize_features()

        prev_output = OrderedDict()
        object_id = '1'
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            if ui_control.mode == 'track' and ui_control.mode_switch:
                ui_control.mode_switch = False
                init_state = ui_control.get_bb()
                out = tracker.initialize(frame, {'init_bbox': init_state, 'object_ids': object_id})
                if out is None:
                    out = {}
                prev_output[object_id] = out
            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)
            elif ui_control.mode == 'track':
                info = OrderedDict()
                info['previous_output'] = prev_output
                prev_output = OrderedDict()

                out = tracker.track(frame, info)
                prev_output[object_id] = out
                state = [int(s) for s in out['target_bbox']]
                if 'segmentation' in out:
                    frame_disp = overlay_mask(frame_disp, out['segmentation'])
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Put text
            font_color = (0, 0, 0)
            if ui_control.mode == 'init' or ui_control.mode == 'select':
                cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            elif ui_control.mode == 'track':
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def run_vot(self):
        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.run_id = self.run_id
        tracker = self.create_tracker(params)
        tracker.initialize_features()

        import pytracking.utils.vot as vot

        def _convert_anno_to_list(vot_anno):
            vot_anno = [vot_anno[0][0][0], vot_anno[0][0][1], vot_anno[0][1][0], vot_anno[0][1][1],
                        vot_anno[0][2][0], vot_anno[0][2][1], vot_anno[0][3][0], vot_anno[0][3][1]]
            return vot_anno

        def _convert_image_path(image_path):
            image_path_new = image_path[20:- 2]
            return "".join(image_path_new)

        """Run tracker on VOT."""

        handle = vot.VOT("polygon")

        vot_anno_polygon = handle.region()
        vot_anno_polygon = _convert_anno_to_list(vot_anno_polygon)

        init_state = convert_vot_anno_to_rect(vot_anno_polygon, tracker.params.vot_anno_conversion_type)

        image_path = handle.frame()
        if not image_path:
            return
        image_path = _convert_image_path(image_path)

        image = self._read_image(image_path)
        tracker.initialize(image, {'init_bbox': init_state})

        # Track
        while True:
            image_path = handle.frame()
            if not image_path:
                break
            image_path = _convert_image_path(image_path)

            image = self._read_image(image_path)
            out = tracker.track(image)
            state = out['target_bbox']

            handle.report(vot.Rectangle(state[0], state[1], state[2], state[3]))

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()
        return params

    def init_visualization(self):
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()

    def visualize(self, image, state, segmentation=None):
        self.ax.cla()
        self.ax.imshow(image)
        if segmentation is not None:
            self.ax.imshow(segmentation, alpha=0.5)
        rect = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(rect)

        if getattr(self, 'gt_state', None) is not None:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        draw_figure(self.fig)

        if self.pause_mode:
            keypress = False
            while not keypress:
                keypress = plt.waitforbuttonpress()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def _read_image(self, image_file: str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)


