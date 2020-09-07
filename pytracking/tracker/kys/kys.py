import torch
import torch.nn
import torch.nn.functional as F
import math
import time
from pytracking.tracker.base import BaseTracker
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
from ltr.models.target_classifier.initializer import FilterInitializerZero
from ltr.models.kys.utils import CenterShiftFeatures, shift_features


class PrevStateHandler:
    def __init__(self):
        self.info_dict = {}

    def set_data(self, frame_number, feat, state, im, bb, label, bb_patch):
        self.info_dict = {'frame_number': frame_number, 'feat': feat, 'state': state, 'im': im, 'bb': bb,
                          'label': label, 'bb_patch': bb_patch}

    def get_data(self):
        return self.info_dict['feat'], self.info_dict['state'], self.info_dict['im'], self.info_dict['label'], \
               self.info_dict['bb_patch']

    def reset_state(self):
        # Reset state in case of long occlusions
        if self.info_dict['state'] is not None:
            self.info_dict['state'] = self.info_dict['state'] * 0.0


class KYS(BaseTracker):
    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        sz = self.params.image_sample_size
        self.img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale = math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Convert image
        im = numpy_to_torch(image)

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize dimp classifier
        self.init_classifier(init_backbone_feat)

        # Initialize IoUNet
        if self.params.get('use_iou_net', True):
            self.init_iou_net(init_backbone_feat)

        self.label_sz = self.feature_sz / self.params.score_downsample_factor

        output_sigma_factor = self.params.output_sigma_factor
        self.sigma = (self.label_sz / self.img_support_sz *
                      self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        self.prev_state_handler = PrevStateHandler()

        self.init_motion_module(im)

        out = {'time': time.time() - tic}
        return out

    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)
        self.im = im

        # ------- LOCALIZATION ------- #
        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                                  self.target_scale * self.params.scale_factors,
                                                                                  self.img_sample_sz)
        # Extract classification features
        test_patch = im_patches[0].int()
        self.test_patch = test_patch

        test_x = self.get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_dimp = self.classify_target(test_x)

        # Compute fused score using the motion module
        scores_fused, motion_feat, new_state_vector = self.get_response_prediction(backbone_feat, scores_dimp)

        # Localize the target
        translation_vec, scale_ind, s, flag = self.localize_target(scores_fused, scores_dimp, sample_scales)
        new_pos = sample_pos[scale_ind, :] + translation_vec

        self.debug_info['flag' + self.id_str] = flag
        self.search_area_box = torch.cat((sample_coords[scale_ind, [1, 0]],
                                          sample_coords[scale_ind, [3, 2]] - sample_coords[scale_ind, [1, 0]] - 1))

        # Debug information
        dimp_score_at_loc = self.debug_info['dimp_score_at_loc']

        # Update position and scale
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                update_scale_flag = update_scale_flag and (dimp_score_at_loc > self.params.get('min_dimp_score_for_scale_update', -1.0))

                self.debug_info['update_scale_flag'] = update_scale_flag

                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(backbone_feat, sample_pos[scale_ind, :], sample_scales[scale_ind], scale_ind,
                                       update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])

        # ------- UPDATE ------- #
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if dimp_score_at_loc < self.params.get('min_dimp_score_update', -1.0):
            update_flag = False

        if update_flag and self.params.get('update_classifier', False):
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind, :],
                                             sample_scales[scale_ind])

            # Update the classifier model
            self.update_classifier(train_x, target_box, learning_rate, s[scale_ind, ...])

        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]]-1)/2, self.target_sz[[1, 0]]))

        # Update motion state
        if flag != 'not_found':
            box_patch = self.get_iounet_box(self.pos, self.target_sz, sample_pos[0, :], sample_scales[0])

            prev_label = self.get_label_function(sample_pos[scale_ind, :], sample_scales[scale_ind]).to(self.params.device)[0]
            self.prev_state_handler.set_data(self.frame_num, motion_feat, new_state_vector, test_patch,
                                             new_state, prev_label, box_patch)
            self.prev_anno = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind, :], sample_scales[scale_ind])
        elif self.params.get('reset_state_during_occlusion', False):
            self.prev_state_handler.reset_state()

        if self.visdom is not None:
            self.visdom.register(scores_dimp[0], 'heatmap', 2, 'Dimp')
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
            self.visdom.register(test_patch, 'image', 2, 'im current')
        out = {'target_bbox': new_state.tolist()}

        return out

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.classifier.classify(self.target_filter, sample_x)
        scores = scores[..., :-1, :-1].contiguous()
        return scores

    def get_motion_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.get_motion_feat(backbone_feat)

    def init_motion_module(self, im):
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                                  self.target_scale * self.params.scale_factors,
                                                                                  self.img_sample_sz)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)
        test_patch = im_patches[0].int()

        motion_feat = self.get_motion_features(backbone_feat)

        sample_pos = sample_pos.view(-1)
        prev_label = self.get_label_function(sample_pos, sample_scales[0]).to(self.params.device)[0]

        box_patch = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scales[0])
        current_bb = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

        self.prev_state_handler.set_data(0, motion_feat, None, test_patch, current_bb, prev_label,
                                         box_patch)
        self.prev_anno = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scales[0])

    def get_response_prediction(self, backbone_feat, scores_dimp):
        motion_feat = self.get_motion_features(backbone_feat)

        prev_motion_feat, prev_motion_state_vector, prev_test_patch, prev_label, box_patch = self.prev_state_handler.get_data()

        box_patch = box_patch.to(prev_motion_feat.device)
        box_c = box_patch[0:2] + 0.5 * box_patch[2:]

        box_c_max = self.img_sample_sz[0] * (0.5 + 1.0 / self.params.search_area_scale)
        box_c_min = self.img_sample_sz[0] * (0.5 - 1.0 / self.params.search_area_scale)

        if prev_motion_state_vector is not None and self.params.get('move_feat_to_center', False) and not \
                ((box_c < box_c_max.to(box_c.device)) * (box_c > box_c_min.to(box_c.device))).all():
            # In case the target was not near the center in the prev. frame, shift the feature map so that the target
            # is in the center.
            prev_motion_feat = CenterShiftFeatures(feature_stride=16)(prev_motion_feat.clone(), box_patch)
            prev_motion_state_vector = CenterShiftFeatures(feature_stride=16)(prev_motion_state_vector.clone(), box_patch)
        elif self.params.get('prev_feat_remove_subpixel_shift', False) and prev_motion_state_vector is not None:
            box_c_feat = box_c / 16.0
            box_c_feat_round = box_c_feat.round() + 0.5 * self.net.predictor.fix_coordinate_shift
            feat_trans = (box_c_feat_round - box_c_feat).view(-1, 2) / self.output_sz.view(-1, 2).to(box_c_feat.device)

            prev_motion_feat = shift_features(prev_motion_feat.clone(), feat_trans)

            if prev_motion_state_vector is not None:
                prev_motion_state_vector = shift_features(prev_motion_state_vector.clone(), feat_trans)
            else:
                prev_label = shift_features(prev_label, feat_trans)

        if self.output_window is not None and self.params.get('apply_window_to_dimp_score', True):
            scores_dimp_win = scores_dimp * self.output_window
        else:
            scores_dimp_win = scores_dimp

        predictor_input = {'dimp_score_cur': scores_dimp_win,
                           'label_prev': prev_label,
                           'feat1': prev_motion_feat, 'feat2': motion_feat,
                           'anno_prev': self.prev_anno.to(self.params.device),
                           'state_prev': prev_motion_state_vector}

        window_fn = self.output_window

        with torch.no_grad():
            resp_output = self.net.predictor.predict_response(predictor_input, dimp_thresh=self.params.get('dimp_threshold', None),
                                                              output_window=window_fn)

        scores_am = F.relu(resp_output['response'])
        new_state_vector = resp_output['state_cur']

        return scores_am, motion_feat, new_state_vector

    def localize_target(self, score_fused, score_dimp, sample_scales):
        """Run the target localization."""
        if score_fused is not None:
            score_fused = score_fused[0]
        score_dimp = score_dimp[0]

        # Apply window function
        if self.output_window is not None and score_fused is not None:
            score_dimp_win = score_dimp * self.output_window
        else:
            score_dimp_win = score_dimp

        max_dimp_score = score_dimp.max().item()
        max_id = score_fused.view(-1).argmax()

        dimp_score_at_loc = score_dimp_win.view(-1)[max_id].item()
        self.debug_info['dimp_score_at_loc'] = dimp_score_at_loc
        self.debug_info['max_dimp_score'] = max_dimp_score

        loc_params = {'target_not_found_threshold': self.params.target_not_found_threshold_fused}

        translation_vec, scale_ind, scores, max_dimp_score, flag, max_disp1 = self.compute_target_location(
            score_fused, loc_params, sample_scales, score_dimp_win)

        self.debug_info['fused_score'] = max_dimp_score
        self.debug_info['fused_flag'] = flag

        if self.params.get('perform_hn_mining_dimp', False) and flag != 'not_found':
            hn_flag = self.perform_hn_mining_dimp(score_dimp, max_disp1, sample_scales)

            if hn_flag:
                flag = 'hard_negative'

        return translation_vec, scale_ind, scores, flag

    def compute_target_location(self, scores, loc_params, sample_scales, scores_dimp=None):
        sample_scale = sample_scales[0]

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)

        if scores_dimp is not None:
            max_score1_dimp, max_disp1_dimp = dcf.max2d(scores_dimp)
            max_disp1_dimp = max_disp1_dimp[scale_ind, ...].float().cpu().view(-1)

            # In case dimp score peak and fused score peak only have a small offset between them, go with dimp peak to
            # avoid drift due to state propagation
            if self.params.get('remove_offset_in_fused_score', False):
                if (max_disp1 - max_disp1_dimp).abs().max() == 1:
                    max_disp1 = max_disp1_dimp

        target_disp1 = max_disp1 - self.output_sz // 2
        translation_vec1 = target_disp1 * (self.img_support_sz / self.output_sz) * sample_scale

        if max_score1.item() < loc_params['target_not_found_threshold']:
            return translation_vec1, scale_ind, scores, max_score1.item(), 'not_found', max_disp1
        else:
            return translation_vec1, scale_ind, scores, max_score1.item(), 'normal', max_disp1

    def perform_hn_mining_dimp(self, score_dimp, max_disp1, sample_scales):
        # Compute hard negatives using the dimp score
        sample_scale = sample_scales[0]
        sz = score_dimp.shape[-2:]

        max_score1 = score_dimp[0, max_disp1[0].long(), max_disp1[1].long()]

        target_neigh_sz = self.params.target_neighborhood_scale_safe * (
                    self.target_sz.prod().sqrt().repeat(2) / sample_scale) * (
                                  self.output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])

        scale_ind = 0
        scores_masked = score_dimp[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > 0.1:
            return True
        return False

    def get_label_function(self, sample_pos, sample_scale, output_sz=None):
        train_y = TensorList()
        target_center_norm = (self.pos - sample_pos) / (sample_scale * self.img_support_sz)

        if output_sz is None:
            output_sz = self.label_sz

        ksz_even = (self.kernel_size + 1) % 2
        center = output_sz * target_center_norm + 0.5 * ksz_even

        train_y.append(dcf.label_function_spatial(output_sz, self.sigma, center))
        return train_y

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x

    def update_memory(self, sample_x: TensorList, target_box, learning_rate=None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def update_state(self, new_pos, new_scale=None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_iou_net(self, backbone_feat):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical, augmentation.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor([self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])

    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Set regularization weight and initializer
        if hasattr(self.net, 'classifier'):
            pred_module = getattr(self.net.classifier.filter_optimizer, 'score_predictor', self.net.classifier.filter_optimizer)
        elif hasattr(self.net, 'dimp_classifier'):
            self.net.classifier = self.net.dimp_classifier
            pred_module = getattr(self.net.dimp_classifier.filter_optimizer, 'score_predictor',
                                  self.net.dimp_classifier.filter_optimizer)
        else:
            raise NotImplementedError

        if self.params.get('label_threshold', None) is not None:
            self.net.classifier.filter_optimizer.label_threshold = self.params.label_threshold
        if self.params.get('label_shrink', None) is not None:
            self.net.classifier.filter_optimizer.label_shrink = self.params.label_shrink
        if self.params.get('softmax_reg', None) is not None:
            self.net.classifier.filter_optimizer.softmax_reg = self.params.softmax_reg
        if self.params.get('filter_reg', None) is not None:
            pred_module.filter_reg[0] = self.params.filter_reg
            pred_module.min_filter_reg = self.params.filter_reg
        if self.params.get('filter_init_zero', False):
            self.net.classifier.filter_initializer = FilterInitializerZero(self.net.classifier.filter_size, x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz #+ (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            score_map_sz = self.feature_sz + (self.kernel_size + 1)%2
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(score_map_sz.long(), (score_map_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(score_map_sz.long(), centered=True).to(self.params.device)

            self.output_window = self.output_window.squeeze(0)[:, :-1, :-1]
            if self.params.get('windom_clamp_factor', None) is not None:
                self.output_window = (self.output_window * (1.0 / self.params.get('windom_clamp_factor'))).clamp(0.0, 1.0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, num_iter=num_iter,
                                                                           compute_losses=plot_loss)

        # Init memory
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([x]))

        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.stack(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter,
                                                                                     num_iter=num_iter, feat=samples,
                                                                                     bb=target_boxes,
                                                                                     sample_weight=sample_weights,
                                                                                     compute_losses=plot_loss)

            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.stack(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1,4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min()/3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

        # Optimize the boxes
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = self.params.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            if self.params.has('target_scale_update_rate'):
                self.target_scale = new_scale*self.params.target_scale_update_rate + \
                                    self.target_scale*(1 - self.params.target_scale_update_rate)
            else:
                self.target_scale = new_scale

    def optimize_boxes(self, iou_features, init_boxes):
        return self.optimize_boxes_default(iou_features, init_boxes)

    def optimize_boxes_default(self, iou_features, init_boxes):
        # Optimize iounet boxes
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]],
                                       device=self.params.device).view(1,1,4)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')