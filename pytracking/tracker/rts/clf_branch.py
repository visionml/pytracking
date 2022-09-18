import torch
import torch.nn.functional as F
import math

from pytracking import dcf, TensorList
from pytracking.features.preprocessing import sample_patch_transformed
from pytracking.features import augmentation
from pytracking.utils.plotting import plot_graph

from ltr.models.layers import activation


class ClassifierBranch:
    def __init__(self, parent_tracker):
        self.params = parent_tracker.params
        self.visdom = parent_tracker.visdom
        self.net =    parent_tracker.net
        self.id_str = parent_tracker.id_str

        assert not self.params.get('use_iou_net', False)


    def initialize(self, image, pos, target_sz):

        self.frame_num = 1

        sz = self.params.clf_image_sample_size
        self.clf_img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.clf_img_support_sz = self.clf_img_sample_sz

        # Setup scale bounds
        self.clf_pos       = pos
        self.clf_target_sz = target_sz

        # Set search area.
        search_area = torch.prod(self.clf_target_sz * self.params.clf_search_area_scale).item()
        self.clf_target_scale = math.sqrt(search_area) / self.clf_img_sample_sz.prod().sqrt()
        # Target size in base scale
        self.clf_base_target_sz = self.clf_target_sz / self.clf_target_scale

        self.clf_image_sz = torch.Tensor([image.shape[2], image.shape[3]])

        self.clf_min_scale_factor  = torch.max(10 / self.clf_base_target_sz)
        self.clf_max_scale_factor  = torch.min(self.clf_image_sz / self.clf_base_target_sz)

        # Extract and transform init samples
        init_backbone_feat = self.generate_init_samples(image)

        # Initialize classifier
        self.init_classifier(init_backbone_feat)


    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('clf_border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.clf_target_scale * self.clf_img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('clf_patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.clf_init_sample_scale = (sample_sz / self.clf_img_sample_sz).prod().sqrt()
            tl = self.clf_pos - (sample_sz - 1) / 2
            br = self.clf_pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.clf_init_sample_scale
        else:
            self.clf_init_sample_scale = self.clf_target_scale
            global_shift = torch.zeros(2)

        self.clf_init_sample_pos = self.clf_pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('clf_augmentation_expansion_factor', None)
        aug_expansion_sz = self.clf_img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.clf_img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.clf_img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.clf_img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('clf_random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.clf_img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.clf_transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.clf_augmentation if self.params.get('clf_use_augmentation', False) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.clf_transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.clf_img_sample_sz/2).long().tolist()
            self.clf_transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.clf_transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.clf_transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.clf_transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.clf_transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.clf_init_sample_pos, self.clf_init_sample_scale, aug_expansion_sz, self.clf_transforms)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat


    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)


    def classify(self, backbone_feat):
        self.debug_info = {}
        self.frame_num += 1

        track_test_x = self.get_classification_features(backbone_feat)

        with torch.no_grad():
            clf_scores = self.net.classifier.classify(self.clf_target_filter, track_test_x)

        return track_test_x, clf_scores


    def update_state(self, track_test_x, clf_scores, sample_coords, lwl_pos, lwl_scale, lwl_size,
                     is_lost_seg, is_seg_too_small):

        sample_coords = sample_coords.unsqueeze(0)
        clf_sample_pos, clf_sample_scale = self.get_classifier_sample_location(sample_coords)
        ################################################
        # Localize the target coarsly, determine if it is found at all or not
        translation_vec, scale_ind, s, flag = self.localize_target(clf_scores, clf_sample_pos, clf_sample_scale)

        # Update position and scale
        if not is_lost_seg and not is_seg_too_small and self.params.get('use_seg_pos_sz_for_clf_update', True):
            self.clf_pos = lwl_pos
            self.clf_target_scale = lwl_scale
            self.clf_target_sz = lwl_size
        elif flag != 'not_found' and self.params.get('clf_use_clf_when_lwl_lost', True):
            new_pos = clf_sample_pos[scale_ind,:] + translation_vec
            new_scale = clf_sample_scale
            if new_scale is not None:
                self.clf_target_scale = new_scale.clamp(self.clf_min_scale_factor, self.clf_max_scale_factor)
                self.clf_target_sz = self.clf_base_target_sz * self.clf_target_scale
                self.clf_target_scale = self.clf_target_scale[0]
            # Update pos
            inside_ratio = self.params.get('target_inside_ratio', 0.2)
            inside_offset = (inside_ratio - 0.5) * self.clf_target_sz
            self.clf_pos = torch.max(torch.min(new_pos, self.clf_image_sz - inside_offset), inside_offset)

        clf_pos, clf_target_scale, clf_target_sz = self.clf_pos, self.clf_target_scale, self.clf_target_sz

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()
        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]],
                                          sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))

        self.debug_info['DiMP flag'] = flag
        self.debug_info['DiMP max_score'] = max_score
        if self.visdom is not None:
            self.visdom.register(self.debug_info, 'info_dict', 1, self.id_str + ' Status')
        # Compute output bounding box
        clf_new_state = torch.cat((self.clf_pos[[1,0]] - (self.clf_target_sz[[1,0]]-1)/2, self.clf_target_sz[[1,0]]))

        if self.params.get('clf_output_not_found_box', False) and flag == 'not_found':
            self.clf_output_state = [-1, -1, -1, -1]
        else:
            self.clf_output_state = clf_new_state.tolist()

        self.current_flag = flag

        # Update model
        self.update_classifier_model(track_test_x, sample_coords, scale_ind, s, flag)

        return self.clf_output_state, clf_pos, clf_target_sz


    def update_classifier_model(self, track_test_x, sample_coords, scale_ind, s, flag):

        clf_sample_pos, clf_sample_scale = self.get_classifier_sample_location(sample_coords)

        # ------- UPDATE MODEL ------- #
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('clf_hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):

            # Get train sample
            train_x = track_test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.clf_pos, self.clf_target_sz, clf_sample_pos[scale_ind,:],
                                             clf_sample_scale[scale_ind])

            train_y = self.get_label_function(self.clf_pos, clf_sample_pos[scale_ind,:],
                                              clf_sample_scale[scale_ind]).to(self.params.device)

            # Update the classifier model
            self.update_classifier(train_x, train_y, target_box, learning_rate, s[scale_ind,...])
        ################################################

        if self.visdom is not None:
            self.visdom.register(self.debug_info, 'info_dict', 1, self.id_str + ' Status')


    def get_classifier_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.clf_img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales


    def init_classifier_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.clf_num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.clf_num_stored_samples = self.clf_num_init_samples.copy()
        self.clf_previous_replace_ind = [None] * len(self.clf_num_stored_samples)
        self.clf_sample_weights = TensorList([x.new_zeros(self.params.clf_sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.clf_sample_weights, init_sample_weights, self.clf_num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.clf_training_samples = TensorList(
            [x.new_zeros(self.params.clf_sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.clf_training_samples, train_x):
            ts[:x.shape[0],...] = x


    def update_classifier_memory(self, sample_x: TensorList, sample_y: TensorList, target_box, learning_rate=None):
        # Update weights and get replace ind
        replace_ind = self.update_classifier_sample_weights(
            self.clf_sample_weights, self.clf_previous_replace_ind, self.clf_num_stored_samples,
            self.clf_num_init_samples, learning_rate)
        self.clf_previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.clf_training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        for y_memory, y, ind in zip(self.clf_target_labels, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y

        # Update bb memory
        self.clf_target_boxes[replace_ind[0],:] = target_box

        self.clf_num_stored_samples += 1


    def update_classifier_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples,
                                         num_init_samples, learning_rate=None):
        """ Update weights and get index to replace """
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind,
                                                    num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.clf_learning_rate

            init_samp_weight = self.params.get('clf_init_samples_minimum_weight', None)
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
                    if self.params.get('clf_lower_init_weight', False):
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


    def init_classifier(self, init_backbone_feat):

        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if self.params.get('clf_use_augmentation', False) and 'dropout' in self.params.clf_augmentation:
            num, prob = self.params.clf_augmentation['dropout']
            self.clf_transforms.extend(self.clf_transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.clf_feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.classifier.filter_size
        self.clf_kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.clf_output_sz = self.clf_feature_sz + (self.clf_kernel_size + 1)%2

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Get target boxes for the different augmentations
        target_labels = self.init_target_labels(TensorList([x]))

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('clf_net_opt_iter', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():

            self.clf_target_filter, _, losses = self.net.classifier.get_filter(
                x, target_boxes, train_label=target_labels, num_iter=num_iter)

        # Init memory
        if self.params.get('update_classifier', True):
            self.init_classifier_memory(TensorList([x]))

        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.stack(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3,
                                     self.id_str + ' Classifier Training Loss')
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title=self.id_str + ' Classifier Training Loss')


    def update_classifier(self, train_x, train_y, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.clf_learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('clf_train_sample_interval', 1) == 0:
            self.update_classifier_memory(TensorList([train_x]), train_y, target_box, learning_rate)

        # Decide the number of iterations to run
        alr_init_buff = self.params.get('clf_alr_init_buff', 100)
        force_train = alr_init_buff > 0 and self.frame_num < alr_init_buff

        num_iter = 0
        low_score_th = self.params.get('clf_low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('clf_net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('clf_net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.clf_train_skipping == 0 or force_train:
            num_iter = self.params.get('clf_net_opt_update_iter', None)

        if num_iter <= 0:
            return

        # Get inputs for the DiMP filter optimizer module
        samples = self.clf_training_samples[0][:self.clf_num_stored_samples[0],...]
        target_boxes = self.clf_target_boxes[:self.clf_num_stored_samples[0],:].clone()
        target_labels = self.clf_target_labels[0][:self.clf_num_stored_samples[0],...]
        sample_weights = self.clf_sample_weights[0][:self.clf_num_stored_samples[0]].view(-1, 1, 1, 1).clone()

        # Run the filter optimizer module
        with torch.no_grad():
            self.clf_target_filter, _, losses = self.net.classifier.filter_optimizer(
                self.clf_target_filter,
                num_iter=num_iter, feat=samples,
                bb=target_boxes, train_label=target_labels,
                sample_weight=sample_weights)

        if self.params.debug > 0:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.cat((self.losses, torch.stack(losses)))
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3,
                                     self.id_str + ' Classifier Training Loss')
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title=self.id_str + ' Classifier Training Loss')

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'clf_softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('clf_score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('clf_advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.clf_kernel_size + 1) % 2
        translation_vec = target_disp * (self.clf_img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.clf_img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(
            self.clf_pos, self.clf_target_sz, self.clf_init_sample_pos, self.clf_init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.clf_transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)

        self.clf_target_boxes = init_target_boxes.new_zeros(self.params.clf_sample_memory_size, 4)
        self.clf_target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes


    def init_target_labels(self, train_x: TensorList):
        self.clf_target_labels = TensorList([x.new_zeros(self.params.clf_sample_memory_size, 1,
                                                     x.shape[2] + (int(self.clf_kernel_size[0].item()) + 1) % 2,
                                                     x.shape[3] + (int(self.clf_kernel_size[1].item()) + 1) % 2)
                                         for x in train_x])
        # Output sigma factor
        output_sigma_factor = self.params.get('clf_output_sigma_factor', 1/4)
        self.clf_sigma = (self.clf_feature_sz / self.clf_img_support_sz * self.clf_base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Center pos in normalized img_coords
        target_center_norm = (self.clf_pos - self.clf_init_sample_pos) / (self.clf_init_sample_scale * self.clf_img_support_sz)

        for target, x in zip(self.clf_target_labels, train_x):
            ksz_even = torch.Tensor([(self.clf_kernel_size[0] + 1) % 2, (self.clf_kernel_size[1] + 1) % 2])
            center_pos = self.clf_feature_sz * target_center_norm + 0.5 * ksz_even
            for i, T in enumerate(self.clf_transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.clf_img_support_sz * self.clf_feature_sz
                target[i, 0, ...] = dcf.label_function_spatial(self.clf_feature_sz, self.clf_sigma, sample_center, end_pad=ksz_even)

        return self.clf_target_labels[0][:train_x[0].shape[0]]



    def get_label_function(self, pos, sample_pos, sample_scale):
        train_y = TensorList()
        target_center_norm = (pos - sample_pos) / (sample_scale * self.clf_img_support_sz)

        for sig, sz, ksz in zip([self.clf_sigma], [self.clf_feature_sz], [self.clf_kernel_size]):
            ksz_even = torch.Tensor([(self.clf_kernel_size[0] + 1) % 2, (self.clf_kernel_size[1] + 1) % 2])
            center = sz * target_center_norm + 0.5*ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))

        return train_y


    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""
        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.clf_kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.clf_img_support_sz / output_sz) * sample_scale
        if max_score1.item() < self.params.clf_target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('clf_uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('clf_hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.clf_target_neighborhood_scale * (self.clf_target_sz / sample_scale) * (output_sz / self.clf_img_support_sz)
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
        translation_vec2 = target_disp2 * (self.clf_img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.clf_pos - sample_pos[scale_ind,:]) / ((self.clf_img_support_sz / output_sz) * sample_scale)
        # Handle the different cases
        if max_score2 > self.params.clf_distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.clf_displacement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score2 > self.params.clf_hard_negative_threshold * max_score1 and max_score2 > self.params.clf_target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'
        return translation_vec1, scale_ind, scores_hn, 'normal'