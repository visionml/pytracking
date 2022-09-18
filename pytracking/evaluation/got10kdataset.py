import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os
from PIL import Image
from pathlib import Path


class GOT10KDataset(BaseDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, split, vos_mode=False):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test' or split == 'val':
            self.base_path = os.path.join(self.env_settings.got10k_path, split)
        else:
            self.base_path = os.path.join(self.env_settings.got10k_path, 'train')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

        self.vos_mode = vos_mode

        self.mask_path = None
        if self.vos_mode:
            self.mask_path = self.env_settings.got10k_mask_path

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        masks = None
        if self.vos_mode:
            seq_mask_path = '{}/{}'.format(self.mask_path, sequence_name)
            masks = [self._load_mask(Path(self._get_anno_frame_path(seq_mask_path, f[:-3] + 'png'))) for f in
                     frame_list[0:1]]

        return Sequence(sequence_name, frames_list, 'got10k', ground_truth_rect.reshape(-1, 4),
                        ground_truth_seg=masks)

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

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'ltrval':
            with open('{}/got10k_val_split.txt'.format(self.env_settings.dataspec_path)) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list
