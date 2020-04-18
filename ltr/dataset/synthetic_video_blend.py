from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.bounding_box_utils import masks_to_bboxes
import random
import torch


class SyntheticVideoBlend(BaseVideoDataset):
    """
    Create a synthetic video by applying random transformations to an object (foreground) and pasting it in a
    background image.  Currently, the foreground object is pasted at random locations in different frames.
    """
    def __init__(self, foreground_image_dataset, background_image_dataset, foreground_transform=None,
                 background_transform=None):
        """
        args:
            foreground_image_dataset - A segmentation dataset from which foreground objects are cropped using the
                                       segmentation mask
            background_image_dataset - Dataset used to sample background image for the synthetic video
            foreground_transform - Random transformations to be applied to the foreground object in every frame
            background_transform - Random transformations to be applied to the background image in every frame
        """
        assert foreground_image_dataset.has_segmentation_info()

        super().__init__(foreground_image_dataset.get_name() + '_syn_vid_blend', foreground_image_dataset.root,
                         foreground_image_dataset.image_loader)
        self.foreground_image_dataset = foreground_image_dataset
        self.background_image_dataset = background_image_dataset

        self.foreground_transform = foreground_transform
        self.background_transform = background_transform

    def get_name(self):
        return self.name

    def is_video_sequence(self):
        return False

    def has_class_info(self):
        return self.foreground_image_dataset.has_class_info()

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return self.foreground_image_dataset.get_num_images()

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.get_images_in_class[class_name]

    def get_sequence_info(self, seq_id):
        image_info = self.foreground_image_dataset.get_image_info(seq_id)

        image_info = {k: v.unsqueeze(0) for k, v in image_info.items()}
        return image_info

    def get_class_name(self, seq_id):
        return self.foreground_image_dataset.get_class_name(seq_id)

    def _paste_target(self, fg_image, fg_box, fg_mask, bg_image, paste_loc):
        fg_mask = fg_mask.view(fg_mask.shape[0], fg_mask.shape[1], 1)
        fg_box = fg_box.long().tolist()

        x1 = int(paste_loc[0] - 0.5 * fg_box[2])
        x2 = x1 + fg_box[2]

        y1 = int(paste_loc[1] - 0.5 * fg_box[3])
        y2 = y1 + fg_box[3]

        x1_pad = max(-x1, 0)
        y1_pad = max(-y1, 0)

        x2_pad = max(x2 - bg_image.shape[1], 0)
        y2_pad = max(y2 - bg_image.shape[0], 0)

        bg_mask = torch.zeros((bg_image.shape[0], bg_image.shape[1], 1), dtype=fg_mask.dtype,
                              device=fg_mask.device)

        if x1_pad >= fg_mask.shape[1] or x2_pad >= fg_mask.shape[1] or y1_pad >= fg_mask.shape[0] or y2_pad >= \
                fg_mask.shape[0]:
            return bg_image, bg_mask.squeeze(-1)

        fg_mask_patch = fg_mask[fg_box[1] + y1_pad:fg_box[1] + fg_box[3] - y2_pad,
                                fg_box[0] + x1_pad:fg_box[0] + fg_box[2] - x2_pad, :]

        fg_image_patch = fg_image[fg_box[1] + y1_pad:fg_box[1] + fg_box[3] - y2_pad,
                         fg_box[0] + x1_pad:fg_box[0] + fg_box[2] - x2_pad, :]

        bg_image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] = \
            bg_image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] * (1 - fg_mask_patch.numpy()) \
            + fg_mask_patch.numpy() * fg_image_patch

        bg_mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] = fg_mask_patch

        return bg_image, bg_mask.squeeze(-1)

    def get_frames(self, seq_id, frame_ids, anno=None):
        # Handle foreground
        fg_frame, fg_anno, fg_object_meta = self.foreground_image_dataset.get_image(seq_id, anno=anno)

        fg_frame_list = [fg_frame.copy() for _ in frame_ids]

        fg_anno_frames = {}
        for key, value in fg_anno.items():
            fg_anno_frames[key] = [value[0].clone() for f_id in frame_ids]

        if self.foreground_transform is not None:
            fg_frame_list, fg_anno_frames['bbox'], fg_anno_frames['mask'] = self.foreground_transform(
                image=fg_frame_list,
                bbox=fg_anno_frames['bbox'],
                mask=fg_anno_frames['mask'],
                joint=False)

        # Sample a random background
        bg_seq_id = random.randint(0, self.background_image_dataset.get_num_images() - 1)

        bg_frame, bg_anno, _ = self.background_image_dataset.get_image(bg_seq_id)

        bg_frame_list = [bg_frame.copy() for _ in frame_ids]

        bg_anno_frames = {}
        for key, value in bg_anno.items():
            # Note: Since we get bg anno from image dataset, it does not has frame dimension
            bg_anno_frames[key] = [value.clone() for f_id in frame_ids]

        if self.background_transform is not None:
            if 'mask' in bg_anno_frames.keys():
                bg_frame_list, bg_anno_frames['bbox'], bg_anno_frames['mask'] = self.background_transform(
                    image=bg_frame_list,
                    bbox=bg_anno_frames['bbox'],
                    mask=bg_anno_frames['mask'],
                    joint=False)
            else:
                bg_frame_list, bg_anno_frames['bbox'] = self.background_transform(
                    image=bg_frame_list,
                    bbox=bg_anno_frames['bbox'],
                    joint=False)

        for i in range(len(frame_ids)):
            # To be safe, get target bb for the mask
            bbox = masks_to_bboxes(fg_anno_frames['mask'][i], fmt='t')

            loc_y = random.randint(0, bg_frame_list[i].shape[0] - 1)
            loc_x = random.randint(0, bg_frame_list[i].shape[1] - 1)

            paste_loc = (loc_x, loc_y)
            fg_frame_list[i], fg_anno_frames['mask'][i] = self._paste_target(fg_frame_list[i], bbox,
                                                                             fg_anno_frames['mask'][i],
                                                                             bg_frame_list[i], paste_loc)

            fg_anno_frames['bbox'][i] = masks_to_bboxes(fg_anno_frames['mask'][i], fmt='t')

        object_meta = OrderedDict({'object_class_name': self.get_class_name(seq_id),
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return fg_frame_list, fg_anno_frames, object_meta
