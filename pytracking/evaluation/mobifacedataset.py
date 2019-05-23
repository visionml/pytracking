from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import glob
import numpy as np
import os
import os.path as osp
from collections import OrderedDict
import pandas as pd

def MobifaceDatasetTest():
    return MobifaceDataset('test').get_sequence_list()

def MobifaceDatasetAll():
    return MobifaceDataset('all').get_sequence_list()
def MobifaceDatasetTrain():
    return MobifaceDataset('train').get_sequence_list()


class MobifaceDataset(BaseDataset):
    """ Mobiface dataset.
        Publication:
            MobiFace: A Novel Dataset for Mobile Face Tracking in the Wild
            Yiming Lin, Shiyang Cheng, Jie Shen, Maja Pantic
            arXiv:1805.09749, 2018
            https://arxiv.org/pdf/1805.09749v2

        Download dataset from https://mobiface.github.io/
    """
    def __init__(self, split):
        """
        args:
            split - Split to use. Can be i) 'train': official training set, ii) 'test': official test set, iii) 'all': whole dataset.
        """
        super().__init__()
        self.base_path = self.env_settings.mobiface_path
        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])
    def _get_sequence_list(self, split):

        self.train_meta_fn = osp.join(self.base_path, 'train.meta.csv')
        self.test_meta_fn = osp.join(self.base_path, 'test.meta.csv')
        self.train_meta = pd.read_csv(self.train_meta_fn,index_col=0).transpose().to_dict()
        self.test_meta = pd.read_csv(self.test_meta_fn,index_col=0).transpose().to_dict()
        if split == 'train':
            self.meta = self.train_meta
        elif split == 'test':
            self.meta = self.test_meta
        else:
            self.meta = {**self.train_meta, **self.test_meta} # In Python 3.5 or greater
        self.meta = OrderedDict(sorted(self.meta.items(), key=lambda t: t[0]))
        self.anno_files = []
        for k,v in self.meta.items():
            if k in self.train_meta.keys():
                self.anno_files.append(osp.abspath(osp.join(self.base_path,'train', k+'.annot.csv')))
            else:
                self.anno_files.append(osp.abspath(osp.join(self.base_path,'test', k+'.annot.csv')))
        self.seq_names = sorted(list(self.meta.keys()))
        self.seq_dirs = [fn[:-len('.annot.csv')] for fn in self.anno_files]
        return self.seq_names

    def _construct_sequence(self, sequence_name):
        index = self.seq_names.index(sequence_name)
        img_files = sorted(glob.glob(self.seq_dirs[index]+'/*.jpg'))
        if len(img_files) == 0:
            img_files = sorted(glob.glob(self.seq_dirs[index]+'.png'))
        with open(self.anno_files[index], 'r') as f:
            anno = np.loadtxt(f, delimiter=',', skiprows=1, dtype=int)
        anno = anno[:,1:]
        assert anno.shape[1] == 4

        return Sequence(sequence_name, img_files, anno.reshape(-1, 4))
    def __len__(self):
        '''Overload this function in your evaluation. This should return number of sequences in the evaluation '''
        return len(self.sequence_list)
