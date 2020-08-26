from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from pytracking import TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed, sample_patch
from pytracking.features import augmentation
from collections import OrderedDict


class LWL(BaseTracker):
    multiobj_mode = 'parallel'

    def predicts_segmentation_mask(self):
        return True

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Learn the initial target model. Initialize memory etc.
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The segmentation network
        self.net = self.params.net

        # Convert image
        im = numpy_to_torch(image)

        # Time initialization
        tic = time.time()

        # Output
        out = {}

        # Get target position and size
        state = info['init_bbox']
        init_mask = info.get('init_mask', None)

        if init_mask is not None and not self.params.get('init_with_box', False):
            # shape 1 , 1, h, w (frames, seq, h, w)
            init_mask = torch.tensor(init_mask).unsqueeze(0).unsqueeze(0).float()
        elif hasattr(self.net, 'box_label_encoder'):
            # Generate initial mask from the box
            with torch.no_grad():
                init_backbone_feat = self.net.extract_backbone(im)
                init_feat_clf = self.net.extract_target_model_features(init_backbone_feat)

                init_box = torch.tensor(state).unsqueeze(dim=0).to(init_feat_clf.device)
                init_mask_enc = self.net.box_label_encoder(init_box, init_feat_clf, im.shape[-2:])

                if isinstance(init_mask_enc, (list, tuple)):
                    init_mask_enc = init_mask_enc[0]
                init_mask_raw, _ = self.net.decoder(init_mask_enc, init_backbone_feat, im.shape[-2:])
                init_mask = torch.sigmoid(init_mask_raw)

            out['init_mask'] = init_mask.squeeze().cpu()
            out['segmentation_raw'] = init_mask_raw.squeeze().cpu().numpy()
            out['segmentation'] = init_mask.squeeze().cpu().numpy()
        else:
            raise Exception('No mask provided')

        # Set target center and target size
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object ids
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        sz = self.params.image_sample_size
        self.img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_support_sz = self.img_sample_sz

        # Set search area.
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale = math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Extract and transform sample
        init_backbone_feat, init_masks = self.generate_init_samples(im, init_mask)

        # Initialize target model
        self.init_target_model(init_backbone_feat, init_masks)

        out['time'] = time.time() - tic
        return out

    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Obtain the merged segmentation prediction for the previous frames. This is used to update the target model
        # and determine the search region for the current frame
        if self.object_id is None:
            prev_segmentation_prob_im = info['previous_output']['segmentation_raw']
        else:
            prev_segmentation_prob_im = info['previous_output']['segmentation_raw'][self.object_id]

        prev_segmentation_prob_im = torch.from_numpy(prev_segmentation_prob_im).unsqueeze(0).unsqueeze(0).float()

        # ********************************************************************************** #
        # ------- Update the target model using merged masks from the previous frame ------- #
        # ********************************************************************************** #
        if self.frame_num > 2:
            # Crop the segmentation mask for the previous search area
            if self.params.get('update_target_model', True):
                prev_segmentation_prob_crop, _ = sample_patch(prev_segmentation_prob_im, self.prev_pos,
                                                              self.prev_scale * self.img_sample_sz,
                                                              self.img_sample_sz,
                                                              mode=self.params.get('border_mode', 'replicate'),
                                                              max_scale_change=self.params.get('patch_max_scale_change'),
                                                              is_mask=True)

                # Update the target model
                self.update_target_model(self.prev_test_x, prev_segmentation_prob_crop.clone())

        # ****************************************************************************************** #
        # -------- Estimate target box using the merged segmentation mask from prev. frame --------- #
        # --- The estimated target box is used to obtain the search region for the current frame --- #
        # ****************************************************************************************** #
        self.pos, self.target_sz = self.get_target_state(prev_segmentation_prob_im.squeeze())

        new_target_scale = torch.sqrt(self.target_sz.prod() / self.base_target_sz.prod())

        if self.params.get('max_scale_change') is not None:
            # Do not allow drastic scale change, as this might be caused due to occlusions or incorrect mask
            # prediction
            new_target_scale = self.clip_scale_change(new_target_scale)

        # Update target size and scale using the filtered target size
        self.target_scale = new_target_scale
        self.target_sz = self.base_target_sz * self.target_scale

        # ********************************************************************** #
        # ---------- Predict segmentation mask for the current frame ----------- #
        # ********************************************************************** #

        # Convert image
        im = numpy_to_torch(image)

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                                  self.target_scale,
                                                                                  self.img_sample_sz)

        # Save the search region information as it is needed to merge the segmentation masks for the next frame update
        self.prev_pos = self.get_centered_sample_pos()
        self.prev_scale = self.target_scale

        # Extract features input to the target model
        test_x = self.get_target_model_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scale = self.get_sample_location(sample_coords)

        # Predict the segmentation mask. Note: These are raw scores, before the sigmoid
        segmentation_scores = self.segment_target(test_x, backbone_feat)

        self.prev_test_x = test_x

        # Get the segmentation scores for the full image.
        # Regions outside the search region are assigned low scores (-100)
        segmentation_scores_im = self.convert_scores_crop_to_image(segmentation_scores, im, sample_scale, sample_pos)

        segmentation_mask_im = (segmentation_scores_im > 0.0).float()   # Binary segmentation mask
        segmentation_prob_im = torch.sigmoid(segmentation_scores_im)    # Probability of being target at each pixel

        # ************************************************************************ #
        # ---------- Output estimated segmentation mask and target box ----------- #
        # ************************************************************************ #

        # Get target box from the predicted segmentation
        pred_pos, pred_target_sz = self.get_target_state(segmentation_prob_im.squeeze())
        new_state = torch.cat((pred_pos[[1, 0]] - (pred_target_sz[[1, 0]] - 1) / 2, pred_target_sz[[1, 0]]))
        output_state = new_state.tolist()

        if self.object_id is None:
            # In single object mode, no merge called. Hence return the probabilities
            segmentation_output = segmentation_prob_im
        else:
            # In multi-object mode, return raw scores
            segmentation_output = segmentation_scores_im

        segmentation_mask_im = segmentation_mask_im.view(*segmentation_mask_im.shape[-2:]).cpu().numpy()
        segmentation_output = segmentation_output.cpu().numpy()

        if self.visdom is not None:
            self.visdom.register(segmentation_scores_im, 'heatmap', 2, 'Seg Scores' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

        out = {'segmentation': segmentation_mask_im, 'target_bbox': output_state,
               'segmentation_raw': segmentation_output}
        return out

    def merge_results(self, out_all):
        """ Merges the predictions of individual targets"""
        out_merged = OrderedDict()

        obj_ids = list(out_all.keys())

        # Merge segmentation scores using the soft-aggregation approach from RGMP
        segmentation_scores = []
        for id in obj_ids:
            if 'segmentation_raw' in out_all[id].keys():
                segmentation_scores.append(out_all[id]['segmentation_raw'])
            else:
                # If 'segmentation_raw' is not present, then this is the initial frame for the target. Convert the
                # GT Segmentation mask to raw scores (assign 100 to target region, -100 to background)
                segmentation_scores.append((out_all[id]['segmentation'] - 0.5) * 200.0)

        segmentation_scores = np.stack(segmentation_scores)
        segmentation_scores = torch.from_numpy(segmentation_scores).float()
        segmentation_prob = torch.sigmoid(segmentation_scores)

        # Obtain seg. probability and scores for background label
        eps = 1e-7
        bg_p = torch.prod(1 - segmentation_prob, dim=0).clamp(eps, 1.0 - eps)  # bg prob
        bg_score = (bg_p / (1.0 - bg_p)).log()

        segmentation_scores_all = torch.cat((bg_score.unsqueeze(0), segmentation_scores), dim=0)

        out = []
        for s in segmentation_scores_all:
            s_out = 1.0 / (segmentation_scores_all - s.unsqueeze(0)).exp().sum(dim=0)
            out.append(s_out)

        segmentation_maps_t_agg = torch.stack(out, dim=0)
        segmentation_maps_np_agg = segmentation_maps_t_agg.numpy()

        # Obtain segmentation mask
        obj_ids_all = np.array([0, *map(int, obj_ids)], dtype=np.uint8)
        merged_segmentation = obj_ids_all[segmentation_maps_np_agg.argmax(axis=0)]

        out_merged['segmentation'] = merged_segmentation
        out_merged['segmentation_raw'] = OrderedDict({key: segmentation_maps_np_agg[i + 1]
                                                      for i, key in enumerate(obj_ids)})

        # target_bbox
        out_first = list(out_all.values())[0]
        out_types = out_first.keys()

        for key in out_types:
            if 'segmentation' in key:
                pass
            elif 'target_bbox' in key:
                # Update the target box using the merged segmentation mask
                merged_boxes = {}
                for obj_id, out in out_all.items():
                    segmentation_prob = torch.from_numpy(out_merged['segmentation_raw'][obj_id])
                    pred_pos, pred_target_sz = self.get_target_state(segmentation_prob)
                    new_state = torch.cat((pred_pos[[1, 0]] - (pred_target_sz[[1, 0]] - 1) / 2, pred_target_sz[[1, 0]]))
                    merged_boxes[obj_id] = new_state.tolist()
                out_merged['target_bbox'] = merged_boxes
            else:
                # For fields other than segmentation predictions or target box, only convert the data structure
                out_merged[key] = {obj_id: out[key] for obj_id, out in out_all.items()}

        return out_merged

    def get_target_state(self, segmentation_prob_im):
        """ Estimate target bounding box using the predicted segmentation probabilities """

        # If predicted mask area is too small, target might be occluded. In this case, just return prev. box
        if segmentation_prob_im.sum() < self.params.get('min_mask_area', -10):
            return self.pos, self.target_sz

        if self.params.get('seg_to_bb_mode') == 'var':
            # Target center is the center of mass of the predicted per-pixel seg. probability scores
            prob_sum = segmentation_prob_im.sum()
            e_y = torch.sum(segmentation_prob_im.sum(dim=-1) *
                            torch.arange(segmentation_prob_im.shape[-2], dtype=torch.float32)) / prob_sum
            e_x = torch.sum(segmentation_prob_im.sum(dim=-2) *
                            torch.arange(segmentation_prob_im.shape[-1], dtype=torch.float32)) / prob_sum

            # Target size is obtained using the variance of the seg. probability scores
            e_h = torch.sum(segmentation_prob_im.sum(dim=-1) *
                            (torch.arange(segmentation_prob_im.shape[-2], dtype=torch.float32) - e_y)**2) / prob_sum
            e_w = torch.sum(segmentation_prob_im.sum(dim=-2) *
                            (torch.arange(segmentation_prob_im.shape[-1], dtype=torch.float32) - e_x)**2) / prob_sum

            sz_factor = self.params.get('seg_to_bb_sz_factor', 4)
            return torch.Tensor([e_y, e_x]), torch.Tensor([e_h.sqrt() * sz_factor, e_w.sqrt() * sz_factor])
        else:
            raise Exception('Unknown seg_to_bb_mode mode {}'.format(self.params.get('seg_to_bb_mode')))

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:2] + sample_coord[2:] - 1)
        sample_scales = ((sample_coord[2:] - sample_coord[:2]) / self.img_sample_sz).prod().sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def clip_scale_change(self, new_target_scale):
        """ Limit scale change """
        if not isinstance(self.params.get('max_scale_change'), (tuple, list)):
            max_scale_change = (self.params.get('max_scale_change'), self.params.get('max_scale_change'))
        else:
            max_scale_change = self.params.get('max_scale_change')

        scale_change = new_target_scale / self.target_scale

        if scale_change < max_scale_change[0]:
            new_target_scale = self.target_scale * max_scale_change[0]
        elif scale_change > max_scale_change[1]:
            new_target_scale = self.target_scale * max_scale_change[1]

        return new_target_scale

    def convert_scores_crop_to_image(self, segmentation_scores, im, sample_scale, sample_pos):
        """ Obtain segmentation scores for the full image using the scores for the search region crop. This is done by
            assigning a low score (-100) for image regions outside the search region """

        # Resize the segmention scores to match the image scale
        segmentation_scores_re = F.interpolate(segmentation_scores, scale_factor=sample_scale.item(), mode='bilinear')
        segmentation_scores_re = segmentation_scores_re.view(*segmentation_scores_re.shape[-2:])

        # Regions outside search area get very low score
        segmentation_scores_im = torch.ones(im.shape[-2:], dtype=segmentation_scores_re.dtype) * (-100.0)

        # Find the co-ordinates of the search region in the image scale
        r1 = int(sample_pos[0].item() - 0.5*segmentation_scores_re.shape[-2])
        c1 = int(sample_pos[1].item() - 0.5*segmentation_scores_re.shape[-1])

        r2 = r1 + segmentation_scores_re.shape[-2]
        c2 = c1 + segmentation_scores_re.shape[-1]

        r1_pad = max(0, -r1)
        c1_pad = max(0, -c1)

        r2_pad = max(r2 - im.shape[-2], 0)
        c2_pad = max(c2 - im.shape[-1], 0)

        # Copy the scores for the search region
        shape = segmentation_scores_re.shape
        segmentation_scores_im[r1 + r1_pad:r2 - r2_pad, c1 + c1_pad:c2 - c2_pad] = \
            segmentation_scores_re[r1_pad:shape[0] - r2_pad, c1_pad:shape[1] - c2_pad]

        return segmentation_scores_im

    def segment_target(self, sample_tm_feat, sample_x):
        with torch.no_grad():
            segmentation_scores, mask_encoding_pred = self.net.segment_target(self.target_filter, sample_tm_feat,
                                                                              sample_x)

        return segmentation_scores

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scale, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scale.unsqueeze(0), sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords[0], im_patches[0]

    def get_target_model_features(self, backbone_feat):
        """ Extract features input to the target model"""
        with torch.no_grad():
            return self.net.extract_target_model_features(backbone_feat)

    def generate_init_samples(self, im: torch.Tensor, init_mask):
        """ Generate initial training sample."""

        mode = self.params.get('border_mode', 'replicate')
        if 'inside' in mode:
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
            init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / init_sample_scale
        else:
            init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = 2.0
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Can be extended to include data augmentation on the initial frame
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        # Extract image patches
        im_patches = sample_patch_transformed(im, init_sample_pos, init_sample_scale, aug_expansion_sz, self.transforms)

        init_masks = sample_patch_transformed(init_mask,
                                              init_sample_pos, init_sample_scale, aug_expansion_sz, self.transforms,
                                              is_mask=True)
        init_masks = init_masks.to(self.params.device)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat, init_masks

    def init_memory(self, train_x: TensorList, masks):
        """ Initialize the sample memory used to update the target model """
        assert masks.dim() == 4

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

        self.target_masks = masks.new_zeros(self.params.sample_memory_size, masks.shape[-3], masks.shape[-2],
                                            masks.shape[-1])
        self.target_masks[:masks.shape[0], :, :, :] = masks

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x

    def update_memory(self, sample_x: TensorList, mask, learning_rate=None):
        """ Add a new sample to the memory. If the memory is full, an old sample are removed"""
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        self.target_masks[replace_ind[0], :, :, :] = mask[0, ...]

        self.num_stored_samples = [n + 1 if n < self.params.sample_memory_size else n for n in self.num_stored_samples]


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        """ Update weights and get index to replace """
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
                    if self.params.get('lower_init_weight', False):
                        sw[r_ind] = 1
                    else:
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

    def init_target_model(self, init_backbone_feat, init_masks):
        # Get target model features
        x = self.get_target_model_features(init_backbone_feat)

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))

        ksz = self.net.target_model.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Set number of iterations
        num_iter = self.params.get('net_opt_iter', None)

        with torch.no_grad():
            few_shot_label, few_shot_sw = self.net.label_encoder(init_masks, x.unsqueeze(1))

        # Get the target module parameters using the few-shot learner
        with torch.no_grad():
            self.target_filter, _, losses = self.net.target_model.get_filter(x.unsqueeze(1), few_shot_label,
                                                                             few_shot_sw,
                                                                             num_iter=num_iter)

        # Init memory
        if self.params.get('update_target_model', True):
            self.init_memory(TensorList([x]), masks=init_masks.view(-1, 1, *init_masks.shape[-2:]))

    def update_target_model(self, train_x, mask, learning_rate=None):
        # Set flags and learning rate
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), mask, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        if (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        if num_iter > 0:
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            masks = self.target_masks[:self.num_stored_samples[0], ...]

            with torch.no_grad():
                few_shot_label, few_shot_sw = self.net.label_encoder(masks, samples.unsqueeze(1))

            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            if few_shot_sw is not None:
                # few_shot_sw provides spatial weights, while sample_weights contains temporal weights.
                sample_weights = few_shot_sw * sample_weights.view(-1, 1, 1, 1, 1)

            # Run the filter optimizer module
            with torch.no_grad():
                target_filter, _, losses = self.net.target_model.filter_optimizer(TensorList([self.target_filter]),
                                                                                  num_iter=num_iter,
                                                                                  feat=samples.unsqueeze(1),
                                                                                  label=few_shot_label.unsqueeze(1),
                                                                                  sample_weight=sample_weights)

            self.target_filter = target_filter[0]

