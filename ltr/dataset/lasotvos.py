from pathlib import Path
import os
import numpy as np
import torch
import pandas
import csv
from PIL import Image
from ltr.dataset.lasot import Lasot
from ltr.data.image_loader import jpeg4py_loader, imread_indexed


class LasotVOS(Lasot):
    """ Lasot video object segmentation dataset.
    """

    def __init__(self, anno_path=None, split='train'):
        super().__init__(split=split)
        self.anno_path = anno_path
        self.skip_interval = 5

    @staticmethod
    def _load_anno(path):
        if not path.exists():
            print('path', path, flush=True)
            return None
        im = np.array(Image.open(path))
        im = np.atleast_3d(im)[..., 0]
        # im = imread_indexed(path)
        return im

    def _get_anno_sequence_path(self, seq_id):
        return os.path.join(self.anno_path, self.sequence_list[seq_id])

    def _get_anno_frame_path(self, seq_path, frame_id):
        frame_number = 1 + frame_id * self.skip_interval
        return os.path.join(seq_path, '{:08}.png'.format(frame_number))  # frames start from 1

    #########################
    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        gt = torch.tensor(gt)
        gt = gt[:1000:self.skip_interval]
        return gt

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        target_visible = target_visible[:1000:self.skip_interval]

        return target_visible

    def _get_frame_path(self, seq_path, frame_id):
        frame_number = 1 + frame_id * self.skip_interval
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_number))  # frames start from 1

    #########################
    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        # TODO FIX Me ?? This is not used by the LWL sampler
        obj_meta = None
        # obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        anno_seq_path = self._get_anno_sequence_path(seq_id)

        labels = [self._load_anno(Path(self._get_anno_frame_path(anno_seq_path, f))) for f in frame_ids]
        labels = [torch.Tensor(lb) for lb in labels]
        anno_frames['mask'] = labels

        return frame_list, anno_frames, obj_meta
