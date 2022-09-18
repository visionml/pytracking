from pathlib import Path
import os
import numpy as np
import torch
from PIL import Image
from ltr.dataset.got10k import Got10k
from ltr.data.image_loader import jpeg4py_loader, imread_indexed


class Got10kVOS(Got10k):
    """ Got10K video object segmentation dataset.
    """

    def __init__(self, anno_path=None, split='train'):
        super().__init__(split=split)
        self.anno_path = anno_path

        # TODO this prevents a crash, because that particular sequence does not have masks.
        # Once the missing mask is added, the following code can be removed (handled in base)
        self.sequence_list = [i for i in self.sequence_list if i not in ['GOT-10k_Train_004419']]

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    @staticmethod
    def _load_anno(path):
        if not path.exists():
            print('path', path, flush=True)
            return None
        im = np.array(Image.open(path))
        im = np.atleast_3d(im)[..., 0]
        return im

    def _get_anno_sequence_path(self, seq_id):
        return os.path.join(self.anno_path, self.sequence_list[seq_id])

    def _get_anno_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.png'.format(frame_id + 1))  # frames start from 1

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

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
