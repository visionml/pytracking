import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os
from pytracking.utils.load_text import load_text
from pathlib import Path
from PIL import Image


class TrackingNetDataset(BaseDataset):
    """ TrackingNet test set.

    Publication:
        TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.
        Matthias Mueller,Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2018
        https://ivul.kaust.edu.sa/Documents/Publications/2018/TrackingNet%20A%20Large%20Scale%20Dataset%20and%20Benchmark%20for%20Object%20Tracking%20in%20the%20Wild.pdf

    Download the dataset using the toolkit https://github.com/SilvioGiancola/TrackingNet-devkit.
    """
    def __init__(self, load_frames=True, vos_mode=False):
        super().__init__()
        self.base_path = self.env_settings.trackingnet_path
        self.load_frames = load_frames

        sets = 'TEST'
        if not isinstance(sets, (list, tuple)):
            if sets == 'TEST':
                sets = ['TEST']
            elif sets == 'TRAIN':
                sets = ['TRAIN_{}'.format(i) for i in range(5)]

        self.sequence_list = self._list_sequences(self.base_path, sets)

        self.vos_mode = vos_mode

        self.mask_path = None
        if self.vos_mode:
            self.mask_path = self.env_settings.trackingnet_mask_path

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(set, seq_name) for set, seq_name in self.sequence_list])

    def _construct_sequence(self, set, sequence_name):
        anno_path = '{}/{}/anno/{}.txt'.format(self.base_path, set, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        if self.load_frames:
            frames_path = '{}/{}/frames/{}'.format(self.base_path, set, sequence_name)
            frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
            frame_list.sort(key=lambda f: int(f[:-4]))
            frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        else:
            frames_list = []
            frame_list = []

        masks = None
        if self.vos_mode:
            seq_mask_path = '{}/{}'.format(self.mask_path, sequence_name)
            masks = [self._load_mask(Path(self._get_anno_frame_path(seq_mask_path, f[:-3] + 'png'))) for f in
                     frame_list[0:1]]

        return Sequence(sequence_name, frames_list, 'trackingnet',
                        ground_truth_rect.reshape(-1, 4), ground_truth_seg=masks)

    @staticmethod
    def _load_mask(path):
        if not path.exists():
            print('Error: Could not read: ', path, flush=True)
            return None
        im = np.array(Image.open(path))
        im = np.atleast_3d(im)[..., 0]
        return im

    def _get_anno_frame_path(self, seq_path, frame_name):
        return os.path.join(seq_path, frame_name)

    def __len__(self):
        return len(self.sequence_list)

    def _list_sequences(self, root, set_ids):
        sequence_list = []

        for s in set_ids:
            anno_dir = os.path.join(root, s, "anno")
            sequences_cur_set = [(s, os.path.splitext(f)[0]) for f in os.listdir(anno_dir) if f.endswith('.txt')]

            sequence_list += sequences_cur_set

        return sequence_list
