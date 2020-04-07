from pathlib import Path
from ltr.dataset.vos_base import VOSDatasetBase, VOSMeta
from pytracking.evaluation import Sequence
from ltr.admin.environment import env_settings
from ltr.data.image_loader import jpeg4py_loader


class Davis(VOSDatasetBase):

    """The DAVIS VOS dataset"""

    def __init__(self, root=None, sequences=None, version='2017', split='train', multiobj=True,
                 vis_threshold=10, image_loader=jpeg4py_loader):
        """
        :param root:            Dataset root path. If unset, it uses the path in your local.py config.
        :param sequences:       List of sequence names. Limit to a subset of sequences if not None.
        :param version:         '2016' or '2017
        :param split:           Any name in DAVIS/ImageSets/<year>
        :param multiobj:        Whether the dataset will return all objects in a sequence or
                                multiple sequences with one object in each.
        :param vis_threshold:   Minimum number of pixels required to consider a target object "visible".
        :param image_loader:    Image loader.
        """
        if version == '2017':
            if split in ['train', 'val']:
                root = env_settings().davis_dir if root is None else root
            elif split in ['test-dev']:
                root = env_settings().davis_testdev_dir if root is None else root
            else:
                raise Exception('Unknown split {}'.format(split))
        else:
            root = env_settings().davis16_dir if root is None else root
            
        super().__init__(name='DAVIS', root=Path(root), version=version, split=split, multiobj=multiobj,
                         vis_threshold=vis_threshold, image_loader=image_loader)

        dset_path = self.root
        self._jpeg_path = dset_path / 'JPEGImages' / '480p'
        self._anno_path = dset_path / 'Annotations' / '480p'

        meta_path = dset_path / "generated_meta.json"
        if meta_path.exists():
            self.gmeta = VOSMeta(filename=meta_path)
        else:
            self.gmeta = VOSMeta.generate('DAVIS', self._jpeg_path, self._anno_path)
            self.gmeta.save(meta_path)

        if sequences is None:
            if self.split != 'all':
                fname = dset_path / 'ImageSets' / self.version / (self.split + '.txt')
                sequences = open(fname).read().splitlines()
            else:
                sequences = [p for p in sorted(self._jpeg_path.glob("*")) if p.is_dir()]

        self.sequence_names = sequences
        self._samples = []

        for seq in sequences:
            obj_ids = self.gmeta.get_obj_ids(seq)
            if self.multiobj:  # Multiple objects per sample
                self._samples.append((seq, obj_ids))
            else:  # One object per sample
                self._samples.extend([(seq, [obj_id]) for obj_id in obj_ids])

        print("%s loaded." % self.get_name())

    def _construct_sequence(self, sequence_info):

        seq_name = sequence_info['sequence']
        images, gt_labels, gt_bboxes = self.get_paths_and_bboxes(sequence_info)

        return Sequence(name=seq_name, frames=images, dataset='DAVIS', ground_truth_rect=gt_bboxes,
                        ground_truth_seg=gt_labels, object_ids=sequence_info['object_ids'],
                        multiobj_mode=self.multiobj)
