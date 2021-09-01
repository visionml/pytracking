import os
import numpy as np
import pandas as pd
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
from PIL import Image


class OxUvADataset(BaseDataset):
    """
    OxUva dev and test set consisting of 200 and 166 videos

    Publication:
        Long-term Tracking in the Wild: A Benchmark
        Jack Valmadre, Luca Bertinetto, João F. Henriques, Ran Tao, Andrea Vedaldi, Arnold Smeulders, Philip Torr, Efstratios Gavves
        ECCV 2018
        https://arxiv.org/pdf/1803.09502

    Download the dataset from https://oxuva.github.io/long-term-tracking-benchmark
    """
    def __init__(self, split):
        super().__init__()
        self.base_path = self.env_settings.oxuva_path
        self.split = split

        dev_tasks_df = pd.read_csv(os.path.join(self.base_path, 'tasks', 'dev.csv'), header=None,
                                   names=['video_id', 'object_id', 'init_frame', 'last_frame', 'xmin', 'xmax',
                                          'ymin', 'ymax'],
                                   dtype={'video_id': str, 'object_id': str, 'init_frame': int, 'last_frame': int,
                                          'xmin': float, 'xmax': float, 'ymin': float, 'ymax': float}
                                   )

        test_tasks_df = pd.read_csv(os.path.join(self.base_path, 'tasks', 'test.csv'), header=None,
                                    names=['video_id', 'object_id', 'init_frame', 'last_frame', 'xmin', 'xmax',
                                           'ymin', 'ymax'],
                                    dtype={'video_id': str, 'object_id': str, 'init_frame': int, 'last_frame': int,
                                           'xmin': float, 'xmax': float, 'ymin': float, 'ymax': float}
                                    )

        self.dev_annotations_df = pd.read_csv(os.path.join(self.base_path, 'annotations', 'dev.csv'), header=None,
                                              names=['video_id', 'object_id', 'class_id', 'class_name',
                                                     'contains_cuts', 'always_visible', 'frame_num', 'object_presence',
                                                     'xmin', 'xmax', 'ymin', 'ymax'],
                                              dtype={'video_id': str, 'object_id': str, 'class_id': int,
                                                     'class_name':str, 'contains_cuts': str, 'always_visible': str,
                                                     'frame_num': int, 'object_presence': str, 'xmin': float,
                                                     'xmax': float, 'ymin': float, 'ymax': float}
                                              )
        if self.split == 'dev':
            self.tasks_df = dev_tasks_df
        elif self.split == 'test':
            self.tasks_df = test_tasks_df
        else:
            raise ValueError('Split {} is not a valid option for OxUva'.format(self.split))


    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(i) for i in range(0, self.tasks_df.shape[0])])

    def _construct_sequence(self, sequence_id):
        video_id = self.tasks_df['video_id'][sequence_id]
        object_id = self.tasks_df['object_id'][sequence_id]
        init_frame = self.tasks_df['init_frame'][sequence_id]
        last_frame = self.tasks_df['last_frame'][sequence_id]

        sequence_name = '{}_{}_frames[{:06d}:{:06d}]'.format(video_id, object_id, init_frame, last_frame + 1)


        frames_path = os.path.join('{}', 'images', self.split, '{}').format(self.base_path, video_id)

        frames_list = [os.path.join(frames_path, '{:06d}.jpeg'.format(frame_number)) for frame_number in
                       range(init_frame, last_frame+1)]

        if self.split == 'dev':
            ground_truth_rect = self.build_dev_annotations(frames_list, init_frame, last_frame, object_id, video_id)
        else:
            ground_truth_rect = self.build_test_annotations(frames_list, init_frame, last_frame, object_id, video_id)

        return Sequence(sequence_name, frames_list, 'oxuva', ground_truth_rect.reshape(-1, 4),
                        object_class=None, target_visible=None)

    def build_test_annotations(self, frames_list, init_frame, last_frame, object_id, video_id):
        # Load first image in sequence to extract width and height to compute bbox coordinates in images.
        w, h = Image.open(frames_list[0]).size
        ground_truth_rect = -np.ones((last_frame - init_frame + 1, 4))
        annotation_mask = np.logical_and(self.tasks_df['video_id'] == video_id, self.tasks_df['object_id'] == object_id)
        annotation = self.tasks_df[annotation_mask]
        xxyy = annotation[['xmin', 'xmax', 'ymin', 'ymax']].to_numpy()
        xxyy[:, :2] *= w
        xxyy[:, 2:] *= h
        xywh = np.vstack([xxyy[:, 0], xxyy[:, 2], xxyy[:, 1] - xxyy[:, 0], xxyy[:, 3] - xxyy[:, 2]]).T
        ground_truth_rect[0] = xywh
        return ground_truth_rect

    def build_dev_annotations(self, frames_list, init_frame, last_frame, object_id, video_id):
        # Load first image in sequence to extract width and height to compute bbox coordinates in images.
        w, h = Image.open(frames_list[0]).size
        annotation_mask = np.logical_and(self.dev_annotations_df['video_id'] == video_id,
                                        self.dev_annotations_df['object_id'] == object_id)
        annotations = self.dev_annotations_df[annotation_mask]
        ground_truth_rect = -np.ones((last_frame - init_frame + 1, 4))
        time = annotations['frame_num'].to_numpy()
        time_mask = np.logical_and(time >= init_frame, time <= last_frame)
        xxyy = annotations[['xmin', 'xmax', 'ymin', 'ymax']][time_mask].to_numpy()
        xxyy[:, :2] *= w
        xxyy[:, 2:] *= h
        xywh = np.vstack([xxyy[:, 0], xxyy[:, 2], xxyy[:, 1] - xxyy[:, 0], xxyy[:, 3] - xxyy[:, 2]]).T
        time -= init_frame
        ground_truth_rect[time] = xywh

        return ground_truth_rect

    def old_construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return self.tasks_df.shape[0]

# dev
# sequence_names =   ['vid0029_obj0000_frames[000000:002791]',
#                     'vid0145_obj0000_frames[000000:001351]',
#                     'vid0144_obj0000_frames[000000:002341]',
#                     'vid0143_obj0000_frames[000000:002731]',
#                     'vid0142_obj0000_frames[000000:002431]',
#                     'vid0141_obj0000_frames[000000:002371]',
#                     'vid0140_obj0000_frames[000000:003181]',
#                     'vid0020_obj0000_frames[000000:001381]',
#                     'vid0021_obj0000_frames[000000:001111]',
#                     'vid0022_obj0000_frames[000000:003901]',
#                     'vid0023_obj0000_frames[000000:003181]',
#                     'vid0025_obj0000_frames[000000:037441]',
#                     'vid0025_obj0001_frames[002700:027991]',
#                     'vid0027_obj0000_frames[000000:001441]',
#                     'vid0115_obj0000_frames[000000:005851]',
#                     'vid0058_obj0000_frames[000000:001381]',
#                     'vid0235_obj0000_frames[000000:009481]',
#                     'vid0235_obj0001_frames[000840:008131]',
#                     'vid0236_obj0000_frames[000000:012601]',
#                     'vid0231_obj0000_frames[000000:001471]',
#                     'vid0230_obj0000_frames[000000:004081]',
#                     'vid0233_obj0000_frames[000000:003631]',
#                     'vid0133_obj0000_frames[000000:002341]',
#                     'vid0131_obj0000_frames[000000:001381]',
#                     'vid0137_obj0000_frames[000000:003061]',
#                     'vid0135_obj0000_frames[000000:001381]',
#                     'vid0334_obj0000_frames[000000:002731]',
#                     'vid0335_obj0000_frames[000000:001381]',
#                     'vid0336_obj0000_frames[000000:001381]',
#                     'vid0139_obj0000_frames[000000:001441]',
#                     'vid0330_obj0000_frames[000000:001351]',
#                     'vid0331_obj0000_frames[000000:001381]',
#                     'vid0333_obj0000_frames[000000:001441]',
#                     'vid0155_obj0000_frames[000000:001891]',
#                     'vid0151_obj0000_frames[000000:005041]',
#                     'vid0153_obj0000_frames[000000:013531]',
#                     'vid0258_obj0000_frames[000000:005941]',
#                     'vid0158_obj0000_frames[000000:001441]',
#                     'vid0159_obj0000_frames[000000:004141]',
#                     'vid0228_obj0000_frames[000000:001441]',
#                     'vid0223_obj0000_frames[000000:002281]',
#                     'vid0226_obj0000_frames[000000:003121]',
#                     'vid0252_obj0000_frames[000000:001381]',
#                     'vid0309_obj0000_frames[000000:003331]',
#                     'vid0308_obj0000_frames[000000:001381]',
#                     'vid0301_obj0000_frames[000000:001291]',
#                     'vid0300_obj0000_frames[000000:001321]',
#                     'vid0303_obj0000_frames[000000:002791]',
#                     'vid0302_obj0000_frames[000000:002341]',
#                     'vid0306_obj0000_frames[000000:004141]',
#                     'vid0163_obj0000_frames[000000:002281]',
#                     'vid0162_obj0000_frames[000000:004141]',
#                     'vid0167_obj0000_frames[000000:001381]',
#                     'vid0166_obj0000_frames[000000:001321]',
#                     'vid0169_obj0000_frames[000000:001351]',
#                     'vid0168_obj0000_frames[000000:002251]',
#                     'vid0225_obj0000_frames[000000:001441]',
#                     'vid0316_obj0000_frames[000000:008131]',
#                     'vid0316_obj0001_frames[002700:010831]',
#                     'vid0315_obj0000_frames[000000:001441]',
#                     'vid0315_obj0001_frames[011700:013111]',
#                     'vid0315_obj0002_frames[013950:015391]',
#                     'vid0313_obj0000_frames[000000:005431]',
#                     'vid0311_obj0000_frames[000000:002761]',
#                     'vid0095_obj0000_frames[000000:005491]',
#                     'vid0094_obj0000_frames[000000:001291]',
#                     'vid0097_obj0000_frames[000000:005881]',
#                     'vid0093_obj0000_frames[000000:002671]',
#                     'vid0092_obj0000_frames[000000:001441]',
#                     'vid0014_obj0000_frames[000000:001471]',
#                     'vid0017_obj0000_frames[000000:004081]',
#                     'vid0011_obj0000_frames[000000:001411]',
#                     'vid0012_obj0000_frames[000000:008131]',
#                     'vid0255_obj0000_frames[000000:002281]',
#                     'vid0254_obj0000_frames[000000:003241]',
#                     'vid0253_obj0000_frames[000000:001381]',
#                     'vid0018_obj0000_frames[000000:002341]',
#                     'vid0077_obj0000_frames[000000:003241]',
#                     'vid0076_obj0000_frames[000000:015811]',
#                     'vid0074_obj0000_frames[000000:001441]',
#                     'vid0073_obj0000_frames[000000:006391]',
#                     'vid0072_obj0000_frames[000000:005971]',
#                     'vid0178_obj0000_frames[000000:004441]',
#                     'vid0070_obj0000_frames[000000:003181]',
#                     'vid0176_obj0000_frames[000000:001801]',
#                     'vid0177_obj0000_frames[000000:001441]',
#                     'vid0174_obj0000_frames[000000:001951]',
#                     'vid0175_obj0000_frames[000000:001441]',
#                     'vid0175_obj0001_frames[027000:029341]',
#                     'vid0079_obj0000_frames[000000:005371]',
#                     'vid0146_obj0000_frames[000000:001381]',
#                     'vid0089_obj0000_frames[000000:005491]',
#                     'vid0082_obj0000_frames[000000:014491]',
#                     'vid0080_obj0000_frames[000000:001411]',
#                     'vid0081_obj0000_frames[000000:001231]',
#                     'vid0084_obj0000_frames[000000:002701]',
#                     'vid0085_obj0000_frames[000000:003151]',
#                     'vid0003_obj0000_frames[000000:010861]',
#                     'vid0000_obj0000_frames[000000:004171]',
#                     'vid0001_obj0000_frames[000000:002551]',
#                     'vid0249_obj0000_frames[000000:018541]',
#                     'vid0004_obj0000_frames[000000:003181]',
#                     'vid0005_obj0000_frames[000000:001441]',
#                     'vid0195_obj0000_frames[000000:003241]',
#                     'vid0008_obj0000_frames[000000:003631]',
#                     'vid0241_obj0000_frames[000000:001441]',
#                     'vid0109_obj0000_frames[000000:002281]',
#                     'vid0067_obj0000_frames[000000:007741]',
#                     'vid0062_obj0000_frames[000000:016591]',
#                     'vid0062_obj0001_frames[013410:015241]',
#                     'vid0063_obj0000_frames[000000:008911]',
#                     'vid0103_obj0000_frames[000000:001441]',
#                     'vid0102_obj0000_frames[000000:002581]',
#                     'vid0101_obj0000_frames[000000:003241]',
#                     'vid0100_obj0000_frames[000000:001231]',
#                     'vid0107_obj0000_frames[000000:001351]',
#                     'vid0069_obj0000_frames[000000:004981]',
#                     'vid0281_obj0000_frames[000000:005491]',
#                     'vid0271_obj0000_frames[000000:003241]',
#                     'vid0272_obj0000_frames[000000:005041]',
#                     'vid0274_obj0000_frames[000000:001171]',
#                     'vid0276_obj0000_frames[000000:010951]',
#                     'vid0051_obj0000_frames[000000:001261]',
#                     'vid0299_obj0000_frames[000000:001501]',
#                     'vid0298_obj0000_frames[000000:001441]',
#                     'vid0298_obj0001_frames[005850:007291]',
#                     'vid0054_obj0000_frames[000000:001981]',
#                     'vid0056_obj0000_frames[000000:003241]',
#                     'vid0284_obj0000_frames[000000:008131]',
#                     'vid0291_obj0000_frames[000000:001441]',
#                     'vid0290_obj0000_frames[000000:001441]',
#                     'vid0296_obj0000_frames[000000:001441]',
#                     'vid0295_obj0000_frames[000000:004531]',
#                     'vid0043_obj0000_frames[000000:001051]',
#                     'vid0040_obj0000_frames[000000:001441]',
#                     'vid0205_obj0000_frames[000000:007741]',
#                     'vid0266_obj0000_frames[000000:001441]',
#                     'vid0180_obj0000_frames[000000:013981]',
#                     'vid0186_obj0000_frames[000000:001441]',
#                     'vid0260_obj0000_frames[000000:004111]',
#                     'vid0184_obj0000_frames[000000:005011]',
#                     'vid0189_obj0000_frames[000000:005941]',
#                     'vid0269_obj0000_frames[000000:001381]',
#                     'vid0288_obj0000_frames[000000:001441]',
#                     'vid0289_obj0000_frames[000000:004981]',
#                     'vid0048_obj0000_frames[000000:000991]',
#                     'vid0046_obj0000_frames[000000:009481]',
#                     'vid0047_obj0000_frames[000000:004141]',
#                     'vid0044_obj0000_frames[000000:009031]',
#                     'vid0042_obj0000_frames[000000:010891]',
#                     'vid0053_obj0000_frames[000000:008641]',
#                     'vid0286_obj0000_frames[000000:004141]',
#                     'vid0287_obj0000_frames[000000:004141]',
#                     'vid0190_obj0000_frames[000000:001381]',
#                     'vid0191_obj0000_frames[000000:001441]',
#                     'vid0211_obj0000_frames[000000:005431]',
#                     'vid0210_obj0000_frames[000000:001441]',
#                     'vid0216_obj0000_frames[000000:003181]',
#                     'vid0196_obj0000_frames[000000:002371]',
#                     'vid0196_obj0001_frames[008550:009991]',
#                     'vid0197_obj0000_frames[000000:004981]',
#                     'vid0197_obj0001_frames[000870:004081]',
#                     'vid0292_obj0000_frames[000000:002791]',
#                     'vid0111_obj0000_frames[000000:004591]',
#                     'vid0066_obj0000_frames[000000:012211]',
#                     'vid0213_obj0000_frames[000000:001231]',
#                     'vid0038_obj0000_frames[000000:005221]',
#                     'vid0329_obj0000_frames[000000:001381]',
#                     'vid0212_obj0000_frames[000000:002131]',
#                     'vid0033_obj0000_frames[000000:013591]',
#                     'vid0033_obj0001_frames[004500:010891]',
#                     'vid0032_obj0000_frames[000000:006841]',
#                     'vid0032_obj0001_frames[004080:006511]',
#                     'vid0031_obj0000_frames[000000:003691]',
#                     'vid0030_obj0000_frames[000000:001441]',
#                     'vid0037_obj0000_frames[000000:012511]',
#                     'vid0192_obj0000_frames[000000:001501]',
#                     'vid0034_obj0000_frames[000000:002191]',
#                     'vid0193_obj0000_frames[000000:011281]',
#                     'vid0245_obj0000_frames[000000:004621]',
#                     'vid0246_obj0000_frames[000000:001291]',
#                     'vid0098_obj0000_frames[000000:001381]',
#                     'vid0202_obj0000_frames[000000:001411]',
#                     'vid0203_obj0000_frames[000000:007171]',
#                     'vid0204_obj0000_frames[000000:004591]',
#                     'vid0215_obj0000_frames[000000:002311]',
#                     'vid0215_obj0001_frames[005370:010861]',
#                     'vid0215_obj0002_frames[011670:013111]',
#                     'vid0208_obj0000_frames[000000:003181]',
#                     'vid0214_obj0000_frames[000000:002791]',
#                     'vid0121_obj0000_frames[000000:002341]',
#                     'vid0120_obj0000_frames[000000:019051]',
#                     'vid0123_obj0000_frames[000000:001171]',
#                     'vid0122_obj0000_frames[000000:001441]',
#                     'vid0327_obj0000_frames[000000:011701]',
#                     'vid0327_obj0001_frames[007230:010891]',
#                     'vid0326_obj0000_frames[000000:001891]',
#                     'vid0324_obj0000_frames[000000:003241]',
#                     'vid0129_obj0000_frames[000000:001381]',
#                     'vid0128_obj0000_frames[000000:004081]']
