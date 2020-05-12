import os
import sys
import importlib
import numpy as np
import torch
import time
import matplotlib.patches as patches
import cv2 as cv
import matplotlib.pyplot as plt
from pytracking.analysis.plot_results import get_plot_draw_styles
from pytracking.utils.plotting import draw_figure
from pytracking.evaluation import get_dataset, trackerlist

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)


class Display:
    def __init__(self, sequence_length, plot_draw_styles, sequence_name):
        self.active = True
        self.frame_number = 0
        self.pause_mode = True
        self.step_size = 0
        self.step_direction = 'forward'
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.key_callback_fn)
        plt.tight_layout()

        self.sequence_length = sequence_length
        self.sequence_name = sequence_name
        self.plot_draw_styles = plot_draw_styles

    def key_callback_fn(self, event):
        if event.key == ' ':
            self.pause_mode = not self.pause_mode
            self.step_size = 0
            self.step_direction = 'forward'
        elif event.key == 'right':
            if self.pause_mode:
                self.frame_number += 1

                if self.frame_number >= self.sequence_length:
                    self.frame_number = self.sequence_length - 1
            elif self.step_direction == 'stop':
                self.step_direction = 'forward'
                self.step_size = 0
            elif self.step_direction == 'backward' and self.step_size == 0:
                self.step_direction = 'stop'
            else:
                self.step_size += 1
        elif event.key == 'left':
            if self.pause_mode:
                self.frame_number -= 1

                if self.frame_number < 0:
                    self.frame_number = 0
            elif self.step_direction == 'stop':
                self.step_direction = 'backward'
                self.step_size = 0
            elif self.step_direction == 'forward' and self.step_size == 0:
                self.step_direction = 'stop'
            else:
                self.step_size -= 1
        elif event.key == 'escape' or event.key == 'q':
            self.active = False

    def _get_speed(self):
        delta = 0
        if self.step_direction == 'forward':
            delta = 2 ** abs(self.step_size)
        elif self.step_direction == 'backward':
            delta = -1 * 2 ** abs(self.step_size)

        return delta

    def step(self):
        delta = self._get_speed()

        self.frame_number += delta
        if self.frame_number < 0:
            self.frame_number = 0
        elif self.frame_number >= self.sequence_length:
            self.frame_number = self.sequence_length - 1

    def show(self, image, bb_list, trackers, gt=None):
        self.ax.cla()
        self.ax.imshow(image)

        # Draw rects
        rect_handles = []
        for i, bb in enumerate(bb_list):
            rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1,
                                     edgecolor=self.plot_draw_styles[i]['color'], facecolor='none')
            self.ax.add_patch(rect)

            rect_handles.append(patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1,
                                     edgecolor=self.plot_draw_styles[i]['color'],
                                                  facecolor=self.plot_draw_styles[i]['color'],
                                                  label=trackers[i]))

        if gt is not None:
            rect = patches.Rectangle((gt[0], gt[1]), gt[2], gt[3], linewidth=2, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
            rect_handles.append(rect)

        self.ax.set_axis_off()
        self.ax.axis('equal')
        plt.legend(handles=rect_handles, loc=4, borderaxespad=0.)
        mode = 'manual' if self.pause_mode else 'auto     '
        speed = self._get_speed()
        self.fig.suptitle('Sequence: {}    Mode: {}    Speed: {:d}x'.format(self.sequence_name, mode, speed),
                          fontsize=14)
        draw_figure(self.fig)


def read_image(image_file: str):
    im = cv.imread(image_file)
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)


def _get_display_name(tracker):
    if tracker.display_name is None:
        if tracker.run_id is not None:
            return '{}_{}_{:03d}'.format(tracker.name, tracker.parameter_name, tracker.run_id)
        else:
            return '{}_{}'.format(tracker.name, tracker.parameter_name)
    else:
        return tracker.display_name


def playback_results(trackers, sequence):
    """
    Playback saved results of input trackers for a particular sequence. You can navigate the sequence using left/right
    arrow keys. You can also change to 'auto' mode by pressing space bar, in which case the sequence will be replayed
    at a particular speed. The speed for playback in 'auto' mode can be controlled using the left/right arrow keys.
    You can exit the application using escape or q keys.
    """
    plot_draw_styles = get_plot_draw_styles()

    tracker_results = []
    # Load results
    for trk_id, trk in enumerate(trackers):
        # Load results
        base_results_path = '{}/{}'.format(trk.results_dir, sequence.name)
        results_path = '{}.txt'.format(base_results_path)

        if os.path.isfile(results_path):
            try:
                pred_bb = torch.tensor(np.loadtxt(str(results_path), dtype=np.float64))
            except:
                pred_bb = torch.tensor(np.loadtxt(str(results_path), delimiter=',', dtype=np.float64))
        else:
            raise Exception('Result not found. {}'.format(results_path))

        tracker_results.append(pred_bb)

    # Convert to list of shape seq_length * num_trackers * 4
    tracker_results = torch.stack(tracker_results, dim=1).tolist()
    tracker_names = [_get_display_name(t) for t in trackers]

    display = Display(len(tracker_results), plot_draw_styles, sequence.name)

    while display.active:
        frame_number = display.frame_number
        image = read_image(sequence.frames[frame_number])

        display.show(image, tracker_results[frame_number], tracker_names)

        time.sleep(0.01)
        if display.pause_mode and display.frame_number == frame_number:
            time.sleep(0.1)
        elif not display.pause_mode:
            display.step()

