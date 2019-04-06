import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time


class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params

    def initialize(self, image, state, class_info=None):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def track_sequence(self, sequence):
        """Run tracker on a sequence."""

        # Initialize
        image = self._read_image(sequence.frames[0])

        times = []
        start_time = time.time()
        self.initialize(image, sequence.init_state)
        init_time = getattr(self, 'time', time.time() - start_time)
        times.append(init_time)

        if self.params.visualization:
            self.init_visualization()
            self.visualize(image, sequence.init_state)

        # Track
        tracked_bb = [sequence.init_state]
        for frame in sequence.frames[1:]:
            image = self._read_image(frame)

            start_time = time.time()
            state = self.track(image)
            times.append(time.time() - start_time)

            tracked_bb.append(state)

            if self.params.visualization:
                self.visualize(image, state)

        return tracked_bb, times


    def track_webcam(self):
        """Run tracker with webcam."""

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

                if self.target_tl[0] > self.target_br[0]:
                    temp_var = self.target_tl
                    self.target_tl = self.target_br
                    self.target_br = temp_var

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        cv.namedWindow("Display")
        cv.setMouseCallback("Display", ui_control.mouse_callback)

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            if ui_control.mode == 'track' and ui_control.mode_switch:
                ui_control.mode_switch = False
                init_state = [ui_control.target_tl[0], ui_control.target_tl[1],
                              ui_control.target_br[0] - ui_control.target_tl[0],
                              ui_control.target_br[1] - ui_control.target_tl[1]]
                self.initialize(frame, init_state)

            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.target_tl, ui_control.target_br, (255, 0, 0), 2)
            elif ui_control.mode == 'track':
                state = self.track(frame)
                state = [int(s) for s in state]
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Display the resulting frame
            cv.imshow('Display', frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def init_visualization(self):
        # plt.ion()
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()

    def visualize(self, image, state):
        self.ax.cla()
        self.ax.imshow(image)
        rect = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(rect)

        if hasattr(self, 'gt_state') and False:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.001)

        if self.pause_mode:
            plt.waitforbuttonpress()

    def _read_image(self, image_file: str):
        return cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2RGB)

