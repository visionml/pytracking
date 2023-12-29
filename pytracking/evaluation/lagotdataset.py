import os
import json
import numpy as np
from collections import OrderedDict
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


class LaGOTDataset(BaseDataset):
    def __init__(self, sot_mode=False):
        super().__init__()
        self.sot_mode = sot_mode
        self.base_path = self.env_settings.lasot_path
        if sot_mode:
            self.anno_path = os.path.join(self.env_settings.lagot_path,
                                          'LaGOT_one_object_per_sequence_annotations_final.json')
        else:
            self.anno_path = os.path.join(self.env_settings.lagot_path,
                                          'LaGOT_multiple_object_per_sequence_annotations_final.json')

        self.annos = self._load_annotations()
        self.sequence_list = list(self.annos.keys())

    def _load_annotations(self):
        with open(self.anno_path, 'r') as f:
            anno = json.load(f)

        return anno

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        if self.sot_mode:
            ground_truth_rect = np.array(self.annos[sequence_name]['xywh'])
            frames_list = [os.path.join(self.base_path, p) for p in self.annos[sequence_name]['frames']]
            target_visible = np.ones(ground_truth_rect.shape[0], dtype=np.bool)
            target_visible[::3] = np.all(ground_truth_rect[::3] >= 0, axis=1)
            return Sequence(sequence_name, frames_list, 'LaGOT', ground_truth_rect.reshape(-1, 4),
                            target_visible=target_visible)
        else:
            frames_list = [f'{self.base_path}/{p}' for p in self.annos[sequence_name]['frames']]

            track_ids = list(self.annos[sequence_name]['xywh'].keys())

            gt_bboxes = OrderedDict()

            for tid, boxes in self.annos[sequence_name]['xywh'].items():
                gt_bboxes[tid] = np.array(boxes)

            init_data = dict()
            for tid, boxes in gt_bboxes.items():
                im_id = 0
                init_box = boxes[im_id]

                if np.all(init_box > -1):
                    if im_id not in init_data:
                        init_data[im_id] = {'object_ids': [tid], 'bbox': {tid: np.array(init_box)}}
                    else:
                        init_data[im_id]['object_ids'].append(tid)
                        init_data[im_id]['bbox'][tid] = np.array(init_box)

            assert set(init_data[0]['object_ids']) == set(track_ids)
            gt_bboxes = OrderedDict({key: val for key, val in gt_bboxes.items() if key in track_ids})

            return Sequence(name=sequence_name, frames=frames_list, dataset='LaGOT', ground_truth_rect=gt_bboxes,
                            init_data=init_data, object_ids=track_ids,
                            multiobj_mode=True)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        return list(self.annos.keys())
