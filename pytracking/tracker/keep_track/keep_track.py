from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ltr.models.layers import activation
import ltr.data.processing_utils as prutils
from .candidates import CandidateCollection

from collections import defaultdict
import numpy as np


class KeepTrack(BaseTracker):

    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        # Initialize IoUNet
        if self.params.get('use_iou_net', True):
            self.init_iou_net(init_backbone_feat)

        self.logging_dict = defaultdict(list)

        self.params.target_candidate_matching_net.initialize()
        self.target_candidate_matching_net = self.params.target_candidate_matching_net
        self.previous_candidates = None
        self.candidate_collection = None

        if self.visdom is not None:
            self.previous_im_patches = None
            self.previous_score_map = None

        self.target_scales = []
        self.target_not_found_counter = 0

        self.mem_sort_indices = torch.arange(0, self.num_init_samples[0])

        out = {'time': time.time() - tic}
        return out

    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)
        # Extract classification features
        test_x = self.get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores = self.classify_target(test_x)
        if self.params.get('window_output', False):
            scores = self.output_window * scores

        # Localize the target
        search_area_box = torch.cat((sample_coords[0, [1, 0]], sample_coords[0, [3, 2]] - sample_coords[0, [1, 0]] - 1))

        out = self.localize_target_by_candidate_matching(im_patches, backbone_feat, scores, search_area_box,
                                                         sample_pos, sample_scales, im.shape[2:])
        translation_vec, scale_ind, s, flag, candidate_score, matching_visualization_data = out

        object_presence_score = scores.max()
        if (self.candidate_collection is None or self.candidate_collection.object_id_of_selected_candidate == 0):
            object_presence_score = torch.max(object_presence_score, torch.sqrt(object_presence_score.clone().detach()))

        new_pos = sample_pos[scale_ind,:] + translation_vec

        self.debug_info['flag' + self.id_str] = flag

        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))

        # Update position and scale
        if flag == 'not_found':
            self.search_area_rescaling()
        else:
            self.target_not_found_counter = 0
            self.target_scales.append(self.target_scale)

            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])


        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()
        self.debug_info['max_score' + self.id_str] = max_score

        # ------- Compute target certainty ------ #
        target_label_certainty = score_map.max()

        # ------- UPDATE ------- #
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])
            train_y = self.get_label_function(self.pos, sample_pos[scale_ind,:], sample_scales[scale_ind]).to(self.params.device)

            self.update_classifier(train_x, train_y, target_box, learning_rate, s[scale_ind,...], target_label_certainty)

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        out = {
            'target_bbox': output_state,
            'object_presence_score': object_presence_score.cpu().item()
        }

        if self.visdom is not None:
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
            self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
            if self.params.get('visualize_candidate_matching', False):
                self.visualize_candidate_matching(matching_visualization_data)
            if self.params.get('visualize_candidate_assignment_matrix', False):
                self.visualize_candidate_assignment_matrix(matching_visualization_data)

        elif self.params.debug >= 2:
            show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

        return out

    def search_area_rescaling(self):
        if len(self.target_scales) > 0:
            min_scales, max_scales, max_history = 2, 30, 60
            self.target_not_found_counter += 1
            num_scales = max(min_scales, min(max_scales, self.target_not_found_counter))
            target_scales = torch.tensor(self.target_scales)[-max_history:]
            target_scales = target_scales[target_scales >= target_scales[-1]]  # only boxes that are bigger than the `not found`
            target_scales = target_scales[-num_scales:]  # look as many samples into past as not found endures.
            self.target_scale = torch.mean(target_scales) # average bigger boxes from the past

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

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.classifier.classify(self.target_filter, sample_x)
        return scores


    def localize_target_by_candidate_matching(self, im_patches1, backbone_feat1, score_map1, search_area_box1, sample_pos, sample_scales, img_shape):
        matching_visualization_data = None

        max_score1 = score_map1.max()
        candidate_score = max_score1

        if max_score1 < self.params.get('local_max_candidate_score_th', 0.05):
            translation_vec, scale_ind, scores, flag = self.localize_target(score_map1, sample_pos, sample_scales)
            return translation_vec, scale_ind, scores, flag, max_score1, matching_visualization_data

        current_candidates = self.extract_descriptors_and_keypoints(backbone_feat1, score_map1, search_area_box1)

        if self.previous_candidates is None or (self.frame_num - self.previous_candidates['frame_num']) > 1:
            translation_vec, scale_ind, s, flag = self.localize_target(score_map1, sample_pos, sample_scales)
            self.candidate_collection = None

        else:
            # Check if candidate matching can be skipped.
            if (self.previous_candidates['scores'].shape[0] == 1 and current_candidates['scores'].shape[0] == 1 and
                    self.previous_candidates['scores'].max() > 0.5 and current_candidates['scores'].max() > 0.5):
                match_preds = {'matches1': torch.zeros(1).long(), 'match_scores1': torch.ones(1)}

            else:
                match_preds = self.extract_matches(descriptors0=self.previous_candidates['descriptors'],
                                                   img_coords0=self.previous_candidates['img_coords'],
                                                   scores0=self.previous_candidates['scores'],
                                                   descriptors1=current_candidates['descriptors'],
                                                   img_coords1=current_candidates['img_coords'],
                                                   scores1=current_candidates['scores'],
                                                   image_shape=img_shape)

                if self.visdom is not None:
                    matching_visualization_data = dict(match_preds=match_preds, im_patches0=self.previous_im_patches,
                                                       im_patches1=im_patches1, score_map0=self.previous_score_map,
                                                       score_map1=score_map1, frameid0=self.previous_candidates['frame_num'],
                                                       frameid1=self.frame_num)

            self.candidate_collection.update(scores=current_candidates['scores'],
                                             tsm_coords=current_candidates['tsm_coords'],
                                             matches=match_preds['matches1'],
                                             match_scores=match_preds['match_scores1'])

            candidate_coord = current_candidates['tsm_coords'][self.candidate_collection.candidate_id_of_selected_candidate]
            candidate_score = current_candidates['scores'][self.candidate_collection.candidate_id_of_selected_candidate]
            flag = self.candidate_collection.flag

            scale_ind = 0
            s = score_map1.squeeze(1)
            sz = score_map1.shape[-2:]
            score_sz = torch.Tensor(list(sz))
            output_sz = score_sz - (self.kernel_size + 1) % 2
            score_center = (score_sz - 1) / 2

            target_disp = candidate_coord.cpu() - score_center
            translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        # Setup peak collection
        if self.candidate_collection is None:
            self.candidate_collection = CandidateCollection(scores=current_candidates['scores'],
                                                            tsm_coords=current_candidates['tsm_coords'],
                                                            candidate_selection_is_certain=(self.frame_num < 10))

        self.previous_candidates = dict(frame_num=self.frame_num,
                                        descriptors=current_candidates['descriptors'],
                                        img_coords=current_candidates['img_coords'],
                                        scores=current_candidates['scores'])

        if self.visdom is not None:
            self.previous_im_patches = im_patches1
            self.previous_score_map = score_map1

        return translation_vec, scale_ind, s, flag, candidate_score, matching_visualization_data

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None

    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'

    def extract_descriptors_and_keypoints(self, backbone_feat, score_map, search_area_box):
        th = self.params.get('local_max_candidate_score_th', 0.05)
        score_map = score_map.squeeze()
        frame_feat_clf = self.target_candidate_matching_net.get_backbone_clf_feat(backbone_feat)
        tsm_coords, scores = prutils.find_local_maxima(score_map, ks=5, th=th)

        with torch.no_grad():
            descriptors = self.target_candidate_matching_net.descriptor_extractor.get_descriptors(frame_feat_clf, tsm_coords)

        if self.params.get('matching_coordinate_system_reference', 'full') == 'full':
            x, y, w, h = search_area_box.tolist()
            img_coords = torch.stack([
                h * (tsm_coords[:, 0].float() / (score_map.shape[0] - 1)) + y,
                w * (tsm_coords[:, 1].float() / (score_map.shape[1] - 1)) + x
            ]).permute(1, 0)
        else:
            img_coords = torch.stack([
                self.params.image_sample_size * (tsm_coords[:, 0].float() / (score_map.shape[0] - 1)),
                self.params.image_sample_size * (tsm_coords[:, 1].float() / (score_map.shape[1] - 1))
            ]).permute(1, 0)

        candidates = dict(descriptors=descriptors, img_coords=img_coords, scores=scores, tsm_coords=tsm_coords)
        return candidates

    def extract_matches(self, descriptors0, descriptors1, img_coords0, img_coords1, scores0, scores1,
                        image_shape):
        data = {
            'descriptors0': descriptors0,
            'descriptors1': descriptors1,
            'img_coords0': img_coords0,
            'img_coords1': img_coords1,
            'scores0': scores0.unsqueeze(0),
            'scores1': scores1.unsqueeze(0),
            'image_size0': image_shape[-2:],
            'image_size1': image_shape[-2:],
        }
        with torch.no_grad():
            pred = self.target_candidate_matching_net.matcher(data)

        return pred

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

    def init_target_label_certainties(self, train_x: TensorList):
        num_train_samples = train_x[0].shape[0]
        self.target_label_certainties = train_x[0].new_zeros(self.params.sample_memory_size, 1, 1, 1)
        self.target_label_certainties[:num_train_samples] = 1.

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

    def init_target_labels(self, train_x: TensorList):
        self.target_labels = TensorList([x.new_zeros(self.params.sample_memory_size, 1,
                                                     x.shape[2] + (int(self.kernel_size[0].item()) + 1) % 2,
                                                     x.shape[3] + (int(self.kernel_size[1].item()) + 1) % 2)
                                         for x in train_x])
        # Output sigma factor
        output_sigma_factor = self.params.get('output_sigma_factor', 1/4)
        self.sigma = (self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Center pos in normalized img_coords
        target_center_norm = (self.pos - self.init_sample_pos) / (self.init_sample_scale * self.img_support_sz)

        for target, x in zip(self.target_labels, train_x):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center_pos = self.feature_sz * target_center_norm + 0.5 * ksz_even
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * self.feature_sz
                target[i, 0, ...] = dcf.label_function_spatial(self.feature_sz, self.sigma, sample_center, end_pad=ksz_even)

        return self.target_labels[0][:train_x[0].shape[0]]

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

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, target_box, learning_rate=None, target_label_certainty=None):
        # Update weights and get replace ind
        if (self.candidate_collection is None or self.candidate_collection.object_id_of_selected_candidate == 0):
            target_label_certainty = torch.max(target_label_certainty, torch.sqrt(target_label_certainty.clone().detach()))

        certainties = [self.target_label_certainties.view(-1) * self.sample_weights[0].view(-1)]

        replace_ind = self.update_sample_weights_based_on_certainty(certainties, self.sample_weights,
                                                                    self.previous_replace_ind, self.num_stored_samples,
                                                                    self.num_init_samples, learning_rate)

        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        for y_memory, y, ind in zip(self.target_labels, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y

        # Update target label certainties memory

        self.target_label_certainties[replace_ind[0]] = target_label_certainty

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        if replace_ind[0] >= len(self.mem_sort_indices):
            self.mem_sort_indices = torch.cat([self.mem_sort_indices, torch.zeros(1, dtype=torch.long)])
            self.mem_sort_indices[replace_ind[0]] = torch.max(self.mem_sort_indices) + 1
        else:
            idx = torch.nonzero(self.mem_sort_indices == replace_ind[0])
            mem_temp = self.mem_sort_indices.clone()
            mem_temp[idx:-1] = self.mem_sort_indices[idx+1:]
            mem_temp[-1] = replace_ind[0]
            self.mem_sort_indices = mem_temp

        self.num_stored_samples += 1

    def update_sample_weights_based_on_certainty(self, certainties, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate=None):
        # Update weights and get index to replace
        replace_ind = []
        for cert, sw, prev_ind, num_samp, num_init in zip(certainties, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
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
                    _, r_ind = torch.min(cert[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                elif r_ind == prev_ind:
                    pass
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, pos, sample_pos, sample_scale):
        train_y = TensorList()
        target_center_norm = (pos - sample_pos) / (sample_scale * self.img_support_sz)

        for sig, sz, ksz in zip([self.sigma], [self.feature_sz], [self.kernel_size]):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center = sz * target_center_norm + 0.5*ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))

        return train_y

    def update_state(self, new_pos, new_scale = None):
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
        if torch.is_tensor(self.iou_modulation[0]):
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
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Get target boxes for the different augmentations
        target_labels = self.init_target_labels(TensorList([x]))

        # Init target label certainties, init gth samples as 1.0
        self.init_target_label_certainties(TensorList([x]))

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        self.net.classifier.compute_losses = plot_loss

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes,
                                                                           train_label=target_labels,
                                                                           num_iter=num_iter)

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

    def update_classifier(self, train_x, train_y, target_box, learning_rate=None, scores=None, target_label_certainty=None):
        if target_label_certainty is None:
            target_label_certainty = 1.

        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), train_y, target_box, learning_rate, target_label_certainty)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)

            # do not update if certainty of hn_sample is lower than ths it won't be considered during update anyway.
            ths_cert = self.params.get('certainty_for_weight_computation_ths', 0.5)
            if (self.params.get('use_certainty_for_weight_computation', False) and ths_cert > target_label_certainty):
                num_iter = 0

        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        if self.params.get('net_opt_every_frame', False):
            num_iter = self.params.get('net_opt_every_frame_iter', 1)

        plot_loss = self.params.debug > 0

        self.logging_dict['num_iters'].append(num_iter)

        # Compute sample weights either fully on age or mix with correctness certainty of target lables.
        # Supress memory sample if certainty is below certain threshold.
        sample_weights = self.sample_weights[0][:self.num_stored_samples[0]].view(-1, 1, 1, 1)

        if self.params.get('use_certainty_for_weight_computation', False):
            target_label_certainties = self.target_label_certainties[:self.num_stored_samples[0]].view(-1, 1, 1, 1)

            ths_cert = self.params.get('certainty_for_weight_computation_ths', 0.5)
            weights = target_label_certainties
            weights[weights < ths_cert] = 0.0
            weights = weights*sample_weights
        else:
            weights = sample_weights.clone()


        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_labels = self.target_labels[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()

            self.net.classifier.compute_losses = plot_loss


            # Run the filter optimizer module
            with torch.no_grad():
                target_filter, _, losses = self.net.classifier.filter_optimizer(TensorList([self.target_filter]),
                                                                                num_iter=num_iter, feat=samples,
                                                                                bb=target_boxes, train_label=target_labels,
                                                                                sample_weight=weights)
                self.target_filter = target_filter[0]


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

        if hasattr(self.net.bb_regressor, 'predict_bb'):
            return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

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
        self.predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

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
            self.target_scale = new_scale

    def optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            return self.optimize_boxes_default(iou_features, init_boxes)
        if box_refinement_space == 'relative':
            return self.optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))

    def optimize_boxes_default(self, iou_features, init_boxes):
        # Optimize iounet boxes
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]], device=self.params.device).view(1,1,4)

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

    def optimize_boxes_relative(self, iou_features, init_boxes):
        # Optimize iounet boxes
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(self.params.device).view(1,1,4)

        sz_norm = output_boxes[:,:1,2:].clone()
        output_boxes_rel = bbutils.rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = bbutils.rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        output_boxes = bbutils.rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()

    def direct_box_regression(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        output_boxes = self.net.bb_regressor.predict_bb(self.iou_modulation, iou_features, init_boxes).view(-1,4).cpu()

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)

        predicted_box = output_boxes[0, :]

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale_bbr = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
        new_scale = new_scale_bbr

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')

    def visualize_candidate_assignment_matrix(self, data):
        if data is not None:
            assignment_probs = data['match_preds']['log_assignment'][0].exp().cpu().numpy()

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(assignment_probs, vmin=0, vmax=1)

            ax.set_xticks(np.arange(assignment_probs.shape[1]))
            ax.set_yticks(np.arange(assignment_probs.shape[0]))
            ax.set_xticklabels(['{}'.format(i) for i in range(assignment_probs.shape[1] - 1)] + ['NM'])
            ax.set_yticklabels(['{}'.format(i) for i in range(assignment_probs.shape[0] - 1)] + ['NM'])

            for i in range(assignment_probs.shape[0]):
                for j in range(assignment_probs.shape[1]):
                    if assignment_probs[i, j] < 0.5:
                        ax.text(j, i, '{:0.2f}'.format(assignment_probs[i, j]), ha="center", va="center", color="w")
                    else:
                        ax.text(j, i, '{:0.2f}'.format(assignment_probs[i, j]), ha="center", va="center", color="k")
            ax.set_title('Assignment Matrix Probs {},{}'.format(data['frameid0'], data['frameid1']))
            self.visdom.visdom.matplot(plt, opts={'title': 'assignment matrix'}, win='assignment matrix')
            plt.close(fig)

    def visualize_candidate_matching(self, data):

        def add_circle(axis, x, y, id, color):
            axis.add_patch(patches.Circle((x, y), 20, linewidth=3, edgecolor=color, facecolor='none'))
            axis.text(x, y, '{}'.format(id), ha="center", va="center", color=color, fontsize=15, weight='bold')

        def add_connection(axis, x0, y0, x1, y1, color):
            axis.plot([x0, x1], [y0, y1], color=color, linewidth=2)

        def add_score_value(axis, score, id, offset, color):
            x = 10 + 100 * (id // 3) + offset
            y = (int(id) % 3) * 20 + 10 + img_sz
            axis.text(x, y, '{}: {:0.3f}'.format(id, score), ha="left", va="center", color=color, fontsize=10)

        def add_matching_probability(axis, prob, id0, id1, num_entry, offset, color):
            x = offset + 10
            y = num_entry * 20
            axis.text(x, y, '{}--[{:0.1f}]--{}'.format(id0, 100 * prob, id1), ha="left", va="center", color=color, fontsize=10)

        if data is not None:
            gap = 50

            assignment_probs = data['match_preds']['log_assignment'][0].exp().cpu().numpy()

            im_patches = torch.cat([
                data['im_patches0'],
                255 * torch.ones((1, 3, data['im_patches0'].shape[3], gap)),
                data['im_patches1']
            ], dim=3).permute(0, 2, 3, 1)

            th = self.params.get('local_max_candidate_score_th', 0.05)
            coords0, scores0 = prutils.find_local_maxima(data['score_map0'].squeeze(), ks=5, th=th)
            coords1, scores1 = prutils.find_local_maxima(data['score_map1'].squeeze(), ks=5, th=th)

            img_sz = self.params.image_sample_size
            sm0_sz = data['score_map0'].squeeze().shape
            sm1_sz = data['score_map1'].squeeze().shape

            img_coords0 = torch.stack([
                img_sz*(coords0[:, 0].float()/(sm0_sz[0] - 1)),
                img_sz*(coords0[:, 1].float()/(sm0_sz[1] - 1))
            ]).permute(1, 0).cpu().numpy()

            img_coords1 = torch.stack([
                img_sz*(coords1[:, 0].float()/(sm1_sz[0] - 1)),
                img_sz*(coords1[:, 1].float()/(sm1_sz[1] - 1)) + (gap + img_sz)
            ]).permute(1, 0).cpu().numpy()

            matches1 = data['match_preds']['matches1'][0].cpu().tolist()
            colors = ['red', 'limegreen', 'deepskyblue', 'darkorange', 'darkviolet', 'grey', 'black', 'blue', 'gold', 'pink']

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.imshow(im_patches[0].numpy().astype(np.uint8))

            for i, m in enumerate(matches1):
                color = colors[i % len(colors)]

                add_circle(ax, x=img_coords1[i, 1], y=img_coords1[i, 0], id=i, color=color)
                add_score_value(ax, score=scores1[i], id=i, offset=img_sz + gap, color=color)

                if m >= 0:
                    add_circle(ax, x=img_coords0[m, 1], y=img_coords0[m, 0], id=m, color=color)
                    add_connection(ax, x0=img_coords0[m, 1], y0=img_coords0[m, 0], x1=img_coords1[i, 1],
                                   y1=img_coords1[i, 0], color=color)
                    add_score_value(ax, score=scores0[m], id=m, offset=0, color=color)
                    add_matching_probability(ax, prob=assignment_probs[m, i], id0=m, id1=i, offset=2 * img_sz + gap,
                                             num_entry=sum(torch.tensor(matches1[:i]) >= 0), color=color)


            ax.set_title('Matching between Frames {} {}'.format(data['frameid0'], data['frameid1']))
            ax.axis('off')
            self.visdom.visdom.matplot(plt, opts={'title': 'matching between frames'}, win='matching between frames')
            plt.close(fig)
