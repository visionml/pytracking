'''
Edit the tracker.py file accordingly.

Set self.rtsp to your rtsp stream

Set self.localIP as it's the Server IP (your PC)

Set self.localPort as it's the Server communication port

'''

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
from ltr.data.bounding_box_utils import masks_to_bboxes
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
from pathlib import Path
import torch

import threading

import socket
import argparse
import imutils

_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}


def trackerlist(name: str, parameter_name: str, run_ids = None, display_name: str = None):
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

        # Initializing Parameters

        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id
        self.display_name = display_name

        # RTSP Stream
        self.rtsp = 'rtsp://...'
        
        # Number of frames to skip while processing
        self.skip_frame = 1
        self.camera_flag = False
        self.distance_flag = False                # Flag for distance calculation

        # Socket programming parameters
        self.localIP     = "127.0.0.1"            # Server IP
        self.localPort   = 8554                   # Server Port
        self.bufferSize  = 1024
        self.initBB = (0,0,0,0)
        self.initAA = (0,0,0,0)

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            self.segmentation_dir = '{}/{}/{}'.format(env.segmentation_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
            self.segmentation_dir = '{}/{}/{}_{:03d}'.format(env.segmentation_path, self.name, self.parameter_name, self.run_id)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker', self.name))
        if os.path.isdir(tracker_module_abspath):
            tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.visdom = None

    def connection(self, localIP, localPort):

        # Create a datagram socket and bind IP/Port Address
        UDPServerSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        UDPServerSocket.bind((localIP, localPort))

        return UDPServerSocket

    def receive_msg(self, conn, bufferSize):

        bytesAddressPair = conn.recvfrom(bufferSize)

        message = b''
        message = bytesAddressPair[0]
        message = message.decode('utf-8')
        address = bytesAddressPair[1]

        return message, address

    
    def send_msg(self, conn, address, box):

        payload = str(box)
        payload = payload.encode('utf-8')
        conn.sendto(payload, address)


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

    
    def camera_video(self, videofilepath, sf):

        self.cap = cv.VideoCapture(videofilepath)
        
        txt = videofilepath.split("/")
        txt = txt[2].split(".")
        txt = txt[0]
        

        self.camera_thread = False

        # VIDEO Writing
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        self.out = cv.VideoWriter(txt + '_Result.avi',cv.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))

        counter = 0
        skip_frame = sf

        while True:
            if self.camera_flag == False:
                self.success, self.frame = self.cap.read()

                counter += 1

                if counter == skip_frame + 1:
                    self.camera_flag = True

                    self.new_frame = self.frame

                    counter = 0

            if self.success == False:
                break

            if self.camera_thread == True:
                break

            time.sleep(0.005)

    def camera_rtsp(self, rtsp):

        while True:

            self.cap = cv.VideoCapture(rtsp)

            self.camera_thread = False

            while (self.cap.isOpened() == True):
                self.success, self.frame = self.cap.read()
                self.camera_flag = True

                if self.success == False:
                    break

                if self.camera_thread == True:
                    break

                time.sleep(0.005)

            if self.camera_thread == True:
                break


    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the video file.
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

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
            if hasattr(tracker, 'initialize_features'):
                tracker.initialize_features()

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        camera = threading.Thread(target=self.camera_video, args=(videofilepath, self.skip_frame,))
        camera.start()
        print('Camera initiated')
        
        
        while True:
            if self.camera_flag == True:
                break

        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.imshow(display_name, self.frame)


        def _build_init_info(box):
            return {'init_bbox': OrderedDict({1: box}), 'init_object_ids': [1, ], 'object_ids': [1, ],
                    'sequence_object_ids': [1, ]}

        if self.success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(self.frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                if self.camera_flag == True:
                    frame_disp = self.new_frame.copy()

                    cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                               1.5, (0, 0, 255), 1)

                    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                    init_state = [x, y, w, h]
                    tracker.initialize(self.new_frame, _build_init_info(init_state))
                    output_boxes.append(init_state)
                    self.camera_flag = False

                    break

                else:
                    time.sleep(0.01)


        while True:

            if self.camera_flag == True:
                frame_disp = self.new_frame.copy()
                t1 = time.time()
                out = tracker.track(self.new_frame)
                t2 = time.time() - t1
                #print('FPS: ', 1/t2)
                
                state = [int(s) for s in out['target_bbox'][1]]
                output_boxes.append(state)
                
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]), (0, 0, 255), 2)
                
                font_color = (0, 0, 255)
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, font_color, 1)
                
                # Display the resulting frame
                cv.imshow(display_name, frame_disp)
                self.out.write(frame_disp)
                key = cv.waitKey(1)
                if key == ord('q'):
                    self.camera_thread = True
                    camera.join()
                    
                    break
                
                elif key == ord('r'):
                    frame_disp = self.new_frame.copy()
                    
                    cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 2.5, (0, 0, 255), 1)
                    
                    cv.imshow(display_name, frame_disp)
                    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                    init_state = [x, y, w, h]
                    tracker.initialize(self.new_frame, _build_init_info(init_state))
                    output_boxes.append(init_state)
                    
                self.camera_flag = False
            
            else:
                time.sleep(0.01)
        
        # When everything done, release the capture
        self.cap.release()
        cv.destroyAllWindows()
        
        
        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))
            
            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_video_comm(self, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the communication with the Ground Station (socket programming) using rtsp.
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

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
            if hasattr(tracker, 'initialize_features'):
                tracker.initialize_features()

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        while True:
            
            camera = threading.Thread(target=self.camera_rtsp, args=(self.rtsp,))
            camera.start()
            print('Camera thread Initiated')

            output_boxes = []

            while True:
                if self.camera_flag == True:
                    break

            self.conn = self.connection(self.localIP,self.localPort)
            
            while True:
                self.message, self.address = self.receive_msg(self.conn, self.bufferSize)
                if self.message == "Connection":
                    print('Connection Established.')
                    Mess_conn = "Connected"
                    self.send_msg(self.conn, self.address, Mess_conn)
                
                elif self.message == "Send Tracker Coordinates":
                    print('message', self.message)
                    cv.destroyAllWindows()
                    self.conn.close()

                elif self.message == "Close":
                    print('Ground Station is Closed, waiting for reconnection ...')


                elif self.message == "Terminate":
                    print('Program terminated from Ground Station.')

                    self.camera_thread = True
                    camera.join()

                    self.cap.release()
                    cv.destroyAllWindows()
                    self.conn.close()

                    return
                
                else:
                    self.initBB = (int(self.message.split(",")[0][1:]), int(self.message.split(",")[1][1:]), int(self.message.split(",")[2][1:]), 
                    int(self.message.split(",")[3][1:(len(self.message.split(",")[3])-1)]))
                    print('Bounding Box from GCS:', self.initBB)

                    break


            while True:

                if self.camera_flag == True:

                    def _build_init_info(box):
                        return {'init_bbox': OrderedDict({1: box}), 'init_object_ids': [1, ], 'object_ids': [1, ],
                                'sequence_object_ids': [1, ]}

                    if self.initAA[0] != self.initBB[0] or self.initAA[1] != self.initBB[1] or self.initAA[2] != self.initBB[2] or self.initAA[3] != self.initBB[3]:
                        init_state = self.initBB
                        tracker.initialize(self.frame, _build_init_info(init_state))
                        output_boxes.append(init_state)

                    self.initAA = self.initBB
                    frame_disp = self.frame.copy()

                    if self.initBB is not None:
                        # Draw box
                    
                        out = tracker.track(self.frame)
                        state = [int(s) for s in out['target_bbox'][1]]
                        output_boxes.append(state)

                        cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                    (0, 255, 0), 5)

                    
                        self.message2, self.address2 = self.receive_msg(self.conn, self.bufferSize)

                        if self.message2 == "Send Tracker Coordinates":
                            self.send_msg(self.conn, self.address2, state)
                    
                        elif self.message2 == "Close":

                            print('Ground Station is Closed, waiting for reconnection ...')
                            cv.destroyAllWindows()

                            self.camera_thread = True
                            camera.join()

                            self.initBB = (0,0,0,0)
                            self.initAA = (0,0,0,0)

                            break

                        elif self.message2 == "Terminate":
                            print('Program terminated from Ground Station')

                            self.camera_thread = True
                            camera.join()

                            self.cap.release()
                            cv.destroyAllWindows()
                            self.conn.close()

                            return               
                    
                        self.camera_flag = False
            
                else:
                    time.sleep(0.01)
        
            # When everything done, release the capture
            self.cap.release()
            cv.destroyAllWindows()
            self.conn.close()


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

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.new_init = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'init'
                    self.new_init = True

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

        camera = threading.Thread(target=self.camera_rtsp, args=(self.rtsp,))
        camera.start()
        print('Camera initiated')

        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)

        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        next_object_id = 1
        sequence_object_ids = []
        prev_output = OrderedDict()

        while True:
            if self.camera_flag == True:
                break

        while True:
            if self.camera_flag == True:
                frame_disp = self.frame.copy()

                info = OrderedDict()
                info['previous_output'] = prev_output

                if ui_control.new_init:
                    ui_control.new_init = False
                    init_state = ui_control.get_bb()

                    info['init_object_ids'] = [next_object_id, ]
                    info['init_bbox'] = OrderedDict({next_object_id: init_state})
                    sequence_object_ids.append(next_object_id)

                    next_object_id += 1

                # Draw box
                if ui_control.mode == 'select':
                    cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)

                if len(sequence_object_ids) > 0:
                    info['sequence_object_ids'] = sequence_object_ids
                    out = tracker.track(self.frame, info)
                    prev_output = OrderedDict(out)

                    if 'segmentation' in out:
                        frame_disp = overlay_mask(frame_disp, out['segmentation'])

                    if 'target_bbox' in out:
                        for obj_id, state in out['target_bbox'].items():
                            state = [int(s) for s in state]
                            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                        _tracker_disp_colors[obj_id], 5)


                # Put text
                font_color = (0, 255, 0)
                cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 85), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)

                # Display the resulting frame
                cv.imshow(display_name, frame_disp)
                self.out.write(frame_disp)
                key = cv.waitKey(1)
                if key == ord('q'):
                    self.camera_thread = True
                    camera.join()

                    break

                elif key == ord('r'):
                    next_object_id = 1
                    sequence_object_ids = []
                    prev_output = OrderedDict()

                    info = OrderedDict()

                    info['object_ids'] = []
                    info['init_object_ids'] = []
                    info['init_bbox'] = OrderedDict()
                    tracker.initialize(self.frame, info)
                    ui_control.mode = 'init'

                self.camera_flag = False
            
            else:
                time.sleep(0.005)
        

        # When everything done, release the capture
        self.cap.release()
        cv.destroyAllWindows()


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

        if isinstance(state, (OrderedDict, dict)):
            boxes = [v for k, v in state.items()]
        else:
            boxes = (state,)

        for i, box in enumerate(boxes, start=1):
            col = _tracker_disp_colors[i]
            col = [float(c) / 255.0 for c in col]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=col, facecolor='none')
            self.ax.add_patch(rect)

        if getattr(self, 'gt_state', None) is not None:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g', facecolor='none')
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
