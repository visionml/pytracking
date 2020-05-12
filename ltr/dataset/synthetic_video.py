from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.bounding_box_utils import masks_to_bboxes


class SyntheticVideo(BaseVideoDataset):
    """
    Create a synthetic video dataset from an image dataset by applying a random transformation to images.
    """
    def __init__(self, base_image_dataset, transform=None):
        """
        args:
            base_image_dataset - Image dataset used for generating synthetic videos
            transform - Set of transforms to be applied to the images to generate synthetic video.
        """
        super().__init__(base_image_dataset.get_name() + '_syn_vid', base_image_dataset.root,
                         base_image_dataset.image_loader)
        self.base_image_dataset = base_image_dataset
        self.transform = transform

    def get_name(self):
        return self.name

    def is_video_sequence(self):
        return False

    def has_class_info(self):
        return self.base_image_dataset.has_class_info()

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return self.base_image_dataset.get_num_images()

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.get_images_in_class[class_name]

    def get_sequence_info(self, seq_id):
        image_info = self.base_image_dataset.get_image_info(seq_id)

        image_info = {k: v.unsqueeze(0) for k, v in image_info.items()}
        return image_info

    def get_class_name(self, seq_id):
        return self.base_image_dataset.get_class_name(seq_id)

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame, anno, object_meta = self.base_image_dataset.get_image(seq_id, anno=anno)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0].clone() for f_id in frame_ids]

        if self.transform is not None:
            if 'mask' in anno_frames.keys():
                frame_list, anno_frames['bbox'], anno_frames['mask'] = self.transform(image=frame_list,
                                                                                      bbox=anno_frames['bbox'],
                                                                                      mask=anno_frames['mask'],
                                                                                      joint=False)

                anno_frames['bbox'] = [masks_to_bboxes(m, fmt='t') for m in anno_frames['mask']]
            else:
                frame_list, anno_frames['bbox'] = self.transform(image=frame_list,
                                                                 bbox=anno_frames['bbox'],
                                                                 joint=False)

        object_meta = OrderedDict({'object_class_name': self.get_class_name(seq_id),
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
