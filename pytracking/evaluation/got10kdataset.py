import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os


def GOT10KDatasetTest():
    return GOT10KDatasetClass('test').get_sequence_list()


def GOT10KDatasetVal():
    return GOT10KDatasetClass('val').get_sequence_list()


def GOT10KDatasetLTRVal():
    return GOT10KDatasetClass('ltrval').get_sequence_list()


class GOT10KDatasetClass(BaseDataset):
    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval
        if split == 'test' or split == 'val':
            self.base_path = os.path.join(self.env_settings.got10k_path, split)
        else:
            self.base_path = os.path.join(self.env_settings.got10k_path, 'train')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        '''Overload this function in your evaluation. This should return number of sequences in the evaluation '''
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'ltrval':
            with open('{}/got10k_val_split.txt'.format(self.env_settings.dataspec_path)) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list
