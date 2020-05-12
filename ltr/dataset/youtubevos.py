from pathlib import Path
import os
from ltr.dataset.vos_base import VOSDatasetBase, VOSMeta
from pytracking.evaluation import Sequence
import json
from ltr.admin.environment import env_settings
from ltr.data.image_loader import jpeg4py_loader


class YouTubeVOSMeta:
    """ Thin wrapper for YouTubeVOS meta data
    meta.json
    {
        "videos": {
            "<video_id>": {
                "objects": {
                    "<object_id>": {
                        "category": "<category>",
                        "frames": [
                            "<frame_id>",
                            "<frame_id>",
                        ]
                    }
                }
            }
        }
    }
    # <object_id> is the same as the pixel values of object in annotated segmentation PNG files.
    # <frame_id> is the 5-digit index of frame in video, and not necessary to start from 0.
    """

    def __init__(self, dset_split_path):
        self._data = json.load(open(dset_split_path / 'meta.json'))['videos']

    def sequences(self):
        return list(self._data.keys())

    def seq_frames(self, seq_name):
        """ All filename stems of the frames in the sequence """
        frames = set()
        for obj_id in self.object_ids(seq_name):
            for f in self.object_frames(seq_name, obj_id):
                frames.add(f)
        return list(sorted(frames))

    def object_ids(self, seq_name):
        """ All objects in the sequence """
        return list(self._data[seq_name]['objects'].keys())

    def object_category(self, seq_name, obj_id):
        return self._data[seq_name]['objects'][str(obj_id)]['category']

    def object_frames(self, seq_name, obj_id):
        return self._data[seq_name]['objects'][str(obj_id)]['frames']

    def object_first_frame(self, seq_name, obj_id):
        return self.object_frames(seq_name, obj_id)[0]


class YouTubeVOS(VOSDatasetBase):
    """
    YoutubeVOS video object segmentation dataset.

    Publication:
        YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark
        Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang
        ECCV, 2018
        https://arxiv.org/pdf/1809.03327.pdf

    Download dataset from: https://youtube-vos.org/dataset/
    """
    def __init__(self, root=None, version='2019', split='train', cleanup=None, all_frames=False, sequences=None,
                 multiobj=True, vis_threshold=10, image_loader=jpeg4py_loader):
        """
        args:
            root - Dataset root path. If unset, it uses the path in your local.py config.
            version - '2018' or '2019'
            split - 'test', 'train', 'valid', or 'jjtrain', 'jjvalid'. 'jjvalid' corresponds to a custom validation
                    dataset consisting of 300 videos randomly sampled from the train set. 'jjtrain' contains the
                    remaining videos used for training.
            cleanup - List of actions to take to to clean up known problems in the dataset.
                      'aspects': remove frames with weird aspect ratios,
                      'starts': fix up start frames from original meta data
            all_frames - Whether to use an "all_frames" split.
            sequences - List of sequence names. Limit to a subset of sequences if not None.
            multiobj - Whether the dataset will return all objects in a sequence or multiple sequences with one
                       object in each.
            vis_threshold - Minimum number of pixels required to consider a target object "visible".
            image_loader - Image loader.
        """
        root = env_settings().youtubevos_dir if root is None else root
        super().__init__(name="YouTubeVOS", root=Path(root), version=version, split=split, multiobj=multiobj,
                         vis_threshold=vis_threshold, image_loader=image_loader)

        split_folder = self.split
        if self.split.startswith("jj"):
            split_folder = "train"

        dset_path = self.root / self.version / split_folder

        self._anno_path = dset_path / 'Annotations'

        if all_frames:
            self._jpeg_path = self.root / self.version / (split_folder + "_all_frames") / 'JPEGImages'
        else:
            self._jpeg_path = dset_path / 'JPEGImages'

        self.meta = YouTubeVOSMeta(dset_path)
        meta_path = dset_path / "generated_meta.json"
        if meta_path.exists():
            self.gmeta = VOSMeta(filename=meta_path)
        else:
            self.gmeta = VOSMeta.generate('YouTubeVOS', self._jpeg_path, self._anno_path)
            self.gmeta.save(meta_path)

        if all_frames:
            self.gmeta.enable_all_frames(self._jpeg_path)

        if self.split not in ['train', 'valid', 'test']:
            self.gmeta.select_split('youtubevos', self.split)

        if sequences is None:
            sequences = self.gmeta.get_sequence_names()

        to_remove = set()
        cleanup = {} if cleanup is None else set(cleanup)

        if 'aspect' in cleanup:
            # Remove sequences with unusual aspect ratios
            for seq_name in sequences:
                a = self.gmeta.get_aspect_ratio(seq_name)
                if a < 1.45 or a > 1.9:
                    to_remove.add(seq_name)

        if 'starts' in cleanup:
            # Fix incorrect start frames for some objects found with ytvos_start_frames_test()
            bad_start_frames = [("0e27472bea", '2', ['00055', '00060'], '00065'),
                                ("5937b08d69", '4', ['00000'], '00005'),
                                ("5e1ce354fd", '5', ['00010', '00015'], '00020'),
                                ("7053e4f41e", '2', ['00000', '00005', '00010', '00015'], '00020'),
                                ("720e3fa04c", '2', ['00050'], '00055'),
                                ("c73c8e747f", '2', ['00035'], '00040')]
            for seq_name, obj_id, bad_frames, good_frame in bad_start_frames:
                # bad_frames is from meta.json included with the dataset
                # good_frame is from the generated meta - and the first actual frame where the object was seen.
                if seq_name in self.meta._data:
                    frames = self.meta.object_frames(seq_name, obj_id)
                    for f in bad_frames:
                        frames.remove(f)
                    assert frames[0] == good_frame

        sequences = [seq for seq in sequences if seq not in to_remove]

        self.sequence_names = sequences
        self._samples = []

        for seq in sequences:
            obj_ids = self.meta.object_ids(seq)
            if self.multiobj:  # Multiple objects per sample
                self._samples.append((seq, obj_ids))
            else:  # One object per sample
                self._samples.extend([(seq, [obj_id]) for obj_id in obj_ids])

        print("%s loaded." % self.get_name())
        if len(to_remove) > 0:
            print("   %d sequences were removed, (%d remaining)." % (len(to_remove), len(sequences)))

    def _construct_sequence(self, sequence_info):

        seq_name = sequence_info['sequence']
        frame_names = sequence_info['frame_names']
        fname_to_fid = {f: i for i, f in enumerate(frame_names)}
        images, gt_segs, gt_bboxes = self.get_paths_and_bboxes(sequence_info)

        init_data = dict()
        for obj_id in sequence_info['object_ids']:
            if obj_id == '0':
                print("!")
            f_name = self.meta.object_first_frame(seq_name, obj_id)
            f_id = fname_to_fid[f_name]
            if f_id not in init_data:
                init_data[f_id] = {'object_ids': [obj_id],
                                   'bbox': {obj_id: gt_bboxes[obj_id][f_id,:]},
                                   'mask': os.path.join(os.path.dirname(gt_segs[f_id]), (f_name + ".png"))}
                assert init_data[f_id]['mask'] in gt_segs  # If this fails, some file is missing
            else:
                init_data[f_id]['object_ids'].append(obj_id)
                init_data[f_id]['bbox'][obj_id] = gt_bboxes[obj_id][f_id,:]

        return Sequence(name=seq_name, frames=images, dataset='YouTubeVOS', ground_truth_rect=gt_bboxes,
                        init_data=init_data, ground_truth_seg=gt_segs, object_ids=sequence_info['object_ids'],
                        multiobj_mode=self.multiobj)
