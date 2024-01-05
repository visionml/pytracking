import os
from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from ltr.models.layers import activation

from collections import defaultdict, OrderedDict


def remap_object_ids(init_info):
    if 'init_object_ids' not in init_info:
        return init_info, {'1': '1'}

    init_info_new = {}
    loc_to_glob_map = dict(zip([str(i) for i in range(1,len(init_info['object_ids'])+1)], init_info['object_ids']))
    glob_to_loc_map = dict(zip(init_info['object_ids'], [str(i) for i in range(1,len(init_info['object_ids'])+1)]))

    init_info_new['init_object_ids'] = [glob_to_loc_map[i] for i in init_info['init_object_ids']]
    init_info_new['object_ids'] = [glob_to_loc_map[i] for i in init_info['object_ids']]
    init_info_new['init_bbox'] = OrderedDict({glob_to_loc_map[i]: val for i, val in init_info['init_bbox'].items()})
    return init_info_new, loc_to_glob_map


class TaMOs(BaseTracker):

    multiobj_mode = os.environ.get('MULTIOBJ_MODE', 'default')

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

        # self.max_num_objs = 10
        self.max_num_objs = self.net.net.head.filter_predictor.num_tokens

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        info, self.id_map = remap_object_ids(info)

        # Get target position and size
        if 'init_object_ids' in info:
            self.mot_dataset = True
            states = info['init_bbox']
            self.object_ids = [int(oid) for oid in info['init_object_ids']]
            self.pos = OrderedDict({int(oid): torch.Tensor([states[oid][1] + (states[oid][3] - 1)/2, states[oid][0] + (states[oid][2] - 1)/2]) for oid in info['init_object_ids']})
            self.target_sz = OrderedDict({int(oid): torch.Tensor([states[oid][3], states[oid][2]]) for oid in info['init_object_ids']})
            self.target_bbox = OrderedDict({int(oid): torch.Tensor(states[oid]) for oid in info['init_object_ids']})
            self.pos_prev = OrderedDict({int(oid): state.clone() for oid, state in self.pos.items()})
            self.conf_ths = self.params.get('conf_ths', 0.0)
        else:
            self.mot_dataset = False
            self.object_ids = [1]
            state = info['init_bbox']
            self.pos = OrderedDict({1: torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])})
            self.target_sz = OrderedDict({1: torch.Tensor([state[3], state[2]])})
            self.target_bbox = OrderedDict({1: torch.Tensor(state)})
            self.pos_prev = OrderedDict({1: self.pos[1].clone()})
            self.conf_ths = self.params.get('conf_ths', 0.0) + 0.1

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.img_sample_sz = torch.Tensor(self.params.image_sample_size)
        self.img_support_sz = self.img_sample_sz
        tfs = self.params.get('train_feature_size')
        stride = self.params.get('feature_stride', 16)
        self.train_img_sample_sz = torch.Tensor([tfs[0]*stride, tfs[1]*stride])


        # Extract and transform sample
        init_backbone_feat = self.extract_backbone_features(im)[0]

        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        self.logging_dict = defaultdict(list)

        self.cls_weights_avg = None
        self.prev_boxes = OrderedDict(self.target_bbox)

        self.object_was_lost_once = dict(zip(self.object_ids, len(self.object_ids)*[False]))
        self.object_was_lost_reset_counter = dict(zip(self.object_ids, len(self.object_ids)*[0]))

        out = {'time': time.time() - tic}
        return out

    def clip_bbox_to_image_area(self, bbox, image, minwidth=10, minheight=10):
        H, W = image.shape[:2]
        x1 = max(0, min(bbox[0], W - minwidth))
        y1 = max(0, min(bbox[1], H - minheight))
        x2 = max(x1 + minwidth, min(bbox[0] + bbox[2], W))
        y2 = max(y1 + minheight, min(bbox[1] + bbox[3], H))
        return torch.Tensor([x1, y1, x2 - x1, y2 - y1])

    def encode_bbox(self, bboxes):
        mask = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)
        bbox = bboxes[mask]

        stride = self.params.get('feature_stride')
        output_sz = self.params.get('image_sample_size')

        shifts_x = torch.arange(
            0, output_sz[1], step=stride,
            dtype=torch.float32, device=bbox.device
        )
        shifts_y = torch.arange(
            0, output_sz[0], step=stride,
            dtype=torch.float32, device=bbox.device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        xs, ys = locations[:, 0], locations[:, 1]

        xyxy = torch.stack([bbox[:, 0], bbox[:, 1], bbox[:, 0] + bbox[:, 2],
                            bbox[:, 1] + bbox[:, 3]], dim=1)

        l = xs[:, None] - xyxy[:, 0][None]
        t = ys[:, None] - xyxy[:, 1][None]
        r = xyxy[:, 2][None] - xs[:, None]
        b = xyxy[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2).reshape(-1, 4)

        s = torch.tensor([output_sz[1], output_sz[0], output_sz[1], output_sz[0]]).reshape(1, 4).to(bbox.device)
        reg_targets_per_im = reg_targets_per_im / s

        nb = bbox.shape[0]
        reg_targets_per_im = reg_targets_per_im.reshape(output_sz[0] // stride, output_sz[1] // stride, nb, 4).permute(2, 3, 0, 1)

        reg_targets_per_im_all = torch.zeros(bboxes.shape[0], reg_targets_per_im.shape[1], reg_targets_per_im.shape[2], reg_targets_per_im.shape[3], device=bbox.device)
        reg_targets_per_im_all[mask] = reg_targets_per_im

        return reg_targets_per_im_all

    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #
        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im)
        # Extract classification features
        test_x = self.get_backbone_head_feat(backbone_feat)

        # Compute classification scores
        scores_raw, bbox_preds = self.classify_target(test_x)

        if self.params.get('normalize_scores', False):
            scores_raw = torch.sigmoid(scores_raw)

        output_states = OrderedDict()
        object_presence_scores = OrderedDict()
        target_boxes = OrderedDict()
        train_ys = OrderedDict()

        for oid in self.object_ids:
            scale_ind, s, flag, score_loc = self.localize_target(scores_raw[:,:,oid-1], self.scale_factor, oid)

            bbox_raw = self.direct_bbox_regression(bbox_preds[:,:,oid-1], sample_coords, score_loc, scores_raw[:,:,oid-1], stride=8)
            bbox = self.clip_bbox_to_image_area(bbox_raw, image)

            if flag != 'not_found':
                self.pos_prev[oid] = self.pos[oid].clone()
                self.pos[oid] = bbox[:2].flip(0) + bbox[2:].flip(0)/2  # [y + h/2, x + w/2]
                self.target_sz[oid] = bbox[2:].flip(0)

            # ------- UPDATE ------- #

            update_flag = flag not in ['not_found', 'uncertain']
            hard_negative = (flag == 'hard_negative')
            learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

            if update_flag and self.params.get('update_classifier', False) and scores_raw[:,:,oid-1].max() > self.conf_ths:
                # Get train sample
                train_x = test_x['layer3'][scale_ind:scale_ind+1, ...] if 'layer3' in test_x else test_x['2'][scale_ind:scale_ind+1, ...]

                # Create target_box and label for spatial sample
                target_boxes[oid] = self.get_iounet_box(self.pos[oid], self.target_sz[oid], self.scale_factor)
                train_ys[oid] = self.get_label_function(self.pos[oid], self.scale_factor).to(self.params.device)

            score_map = s[scale_ind, ...]

            # Compute output bounding box
            new_state = torch.cat((self.pos[oid][[1, 0]] - (self.target_sz[oid][[1, 0]] - 1) / 2, self.target_sz[oid][[1, 0]]))

            if self.params.get('output_not_found_box', False):
                output_states[str(oid)] = [-1, -1, -1, -1]
            else:
                output_states[str(oid)] = new_state.tolist()

            object_presence_scores[str(oid)] = score_map.max().cpu().item()

        # Update the classifier model
        if len(target_boxes) == len(self.object_ids):
            self.update_memory(TensorList([train_x]), train_ys, target_boxes, learning_rate)


        out = {'target_bbox': output_states,
               'object_presence_score': object_presence_scores}

        if self.visdom is not None:
            self.visualize_raw_results(scores_raw, output_states, object_presence_scores)

        out['target_bbox'] = OrderedDict({self.id_map[id]: val for id, val in out['target_bbox'].items()})
        out['object_presence_score'] = OrderedDict({self.id_map[id]: val for id, val in out['object_presence_score'].items()})

        if not self.mot_dataset:
            d = {}
            for key, val in out.items():
                for oid, states in val.items():
                    d[key] = states
            out = d

        return out

    def visualize_raw_results(self, score_map, new_state, object_presence_scores):
        for i in self.object_ids:
            self.visdom.register(score_map[0, 0, i-1], 'heatmap', 2, 'Score Map' + str(i))
            self.logging_dict['max_score_{}'.format(i)].append(score_map[0, 0, i-1].max())
            self.debug_info['max_score_{}'.format(i)] = object_presence_scores[str(i)]
        self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

    def direct_bbox_regression(self, bbox_preds, sample_coords, score_loc, scores_raw, stride=16):
        shifts_x = torch.arange(
            0, self.img_sample_sz[1], step=stride,
            dtype=torch.float32
        )
        shifts_y = torch.arange(
            0, self.img_sample_sz[0], step=stride,
            dtype=torch.float32
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        xs, ys = locations[:, 0], locations[:, 1]
        s1, s2 = scores_raw.shape[2:]
        xs = xs.reshape(s1, s2)
        ys = ys.reshape(s1, s2)

        ltrb = bbox_preds.permute(0,1,3,4,2)[0,0].cpu() * self.train_img_sample_sz[[1, 0, 1, 0]]
        xs1 = xs - ltrb[:, :, 0]
        xs2 = xs + ltrb[:, :, 2]
        ys1 = ys - ltrb[:, :, 1]
        ys2 = ys + ltrb[:, :, 3]
        sl = score_loc.int()

        x1 = xs1[sl[0], sl[1]] / self.img_sample_sz[1] * (sample_coords[0, 3] - sample_coords[0, 1]) + sample_coords[0, 1]
        y1 = ys1[sl[0], sl[1]] / self.img_sample_sz[0] * (sample_coords[0, 2] - sample_coords[0, 0]) + sample_coords[0, 0]
        x2 = xs2[sl[0], sl[1]] / self.img_sample_sz[1] * (sample_coords[0, 3] - sample_coords[0, 1]) + sample_coords[0, 1]
        y2 = ys2[sl[0], sl[1]] / self.img_sample_sz[0] * (sample_coords[0, 2] - sample_coords[0, 0]) + sample_coords[0, 0]
        w = x2 - x1
        h = y2 - y1

        return torch.Tensor([x1, y1, w, h])

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            train_samples = self.training_samples[0][:self.num_stored_samples[0], ...]
            target_labels = self.target_labels[0][:self.num_stored_samples[0], ...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0], :]

            test_feat = self.net.head.extract_head_feat(sample_x)
            train_feat = self.net.head.extract_head_feat({'layer3': train_samples, '2': train_samples})

            train_ltrb = torch.cat([self.encode_bbox(box).unsqueeze(0) for box in target_boxes], dim=0)
            train_bb = [{i: target_boxes[k,i].unsqueeze(1) for i in range(target_boxes.shape[1]) if torch.all(target_boxes[k, i, 2:] > 0)}
                        for k in range(target_boxes.shape[0])]

            cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc = \
                self.net.head.get_filter_and_features_in_parallel(train_feat, test_feat, num_gth_frames=1,
                                                                  train_label=target_labels, train_ltrb_target=train_ltrb,
                                                                  train_bb=train_bb)

            test_feat_enc_fpn_cls = self.net.head.fpn(cls_test_feat_enc, sample_x)
            test_feat_enc_fpn_bbreg = self.net.head.fpn(bbreg_test_feat_enc, sample_x)
            # fuse encoder and decoder features to one feature map
            if self.params.get('cls_feature_type', 'trafo') == 'trafo':
                target_scores = self.net.head.classifier(cls_test_feat_enc, cls_weights)
            else:
                target_scores = self.net.head.classifier(test_feat_enc_fpn_cls['feat2'], cls_weights)

            # compute the final prediction using the output module
            bbox_preds = self.net.head.bb_regressor(test_feat_enc_fpn_bbreg['feat2'], bbreg_weights)

            if target_scores.shape[-1] != bbox_preds.shape[-1] and target_scores.shape[-2] != bbox_preds.shape[-2]:
                target_scores = F.interpolate(target_scores[0], bbox_preds.shape[-2:], mode='bicubic').unsqueeze(0)

        return target_scores, bbox_preds

    def localize_target(self, scores, scale_factor, obj_id):
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
            return self.localize_advanced(scores, scale_factor, obj_id)

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)

        return scale_ind, scores, None, max_disp

    def construct_hn_window(self, size, pos):
        x1 = torch.arange(0, size[0]) + (size[0]//2 - pos[0])
        x2 = torch.arange(0, size[1]) + (size[1]//2 - pos[1])
        hn1 = 0.5*(1 - torch.cos(2*math.pi*(x1)/size[0]))*((0 < x1) & (x1 < size[0])).float()
        hn2 = 0.5*(1 - torch.cos(2*math.pi*(x2)/size[1]))*((0 < x2) & (x2 < size[1])).float()

        return hn1.reshape(1,-1,1)*hn2.reshape(1,1,-1)

    def localize_advanced(self, scores, scale_factor, obj_id):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = self.pos[obj_id]/((self.img_support_sz/output_sz)/scale_factor)

        scores_hn = scores
        if self.params.get('window_output', False):
            output_window = self.construct_hn_window(self.output_sz, score_center).to(scores.device)
            scores_hn = scores.clone()
            scores *= output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center

        if max_score1.item() < self.params.target_not_found_threshold:
            return scale_ind, scores_hn, 'not_found', max_disp1
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return scale_ind, scores_hn, 'uncertain', max_disp1
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return scale_ind, scores_hn, 'hard_negative', max_disp1

        # Mask out target neighborhood
        target_neigh_sz = self.target_sz[obj_id]/((self.img_support_sz/output_sz)/scale_factor)*self.params.target_neighborhood_scale

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

        prev_target_vec = (self.pos[obj_id] - self.pos_prev[obj_id])/((self.img_support_sz/output_sz)/scale_factor)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return scale_ind, scores_hn, 'hard_negative', max_disp1
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return scale_ind, scores_hn, 'hard_negative', max_disp2
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return scale_ind, scores_hn, 'uncertain', max_disp1

            # If also the distractor is close, return with highest score
            return scale_ind, scores_hn, 'uncertain', max_disp1

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return scale_ind, scores_hn, 'hard_negative', max_disp1

        return scale_ind, scores_hn, 'normal', max_disp1

    def extract_backbone_features(self, im: torch.Tensor):
        target_aspect_ratio = float(self.img_sample_sz[0])/float(self.img_sample_sz[1])

        w, h = float(im.shape[3]), float(im.shape[2])
        aspect_ratio = h/w
        padding = [0, 0] # h, w

        if aspect_ratio <= target_aspect_ratio:
            # scale such that width matches
            self.scale_factor = float(self.img_sample_sz[1])/w
            target_sz = (int(h*self.scale_factor), int(self.img_sample_sz[1]))
            padding[0] = int((self.img_sample_sz[0] - target_sz[0]))
        else:
            # scale such that height matches
            self.scale_factor = float(self.img_sample_sz[0])/h
            target_sz = (int(self.img_sample_sz[0]), int(w*self.scale_factor))
            padding[1] = int((self.img_sample_sz[1] - target_sz[1]))

        im_scale = F.interpolate(im, target_sz, mode='bilinear')

        im_patches = F.pad(im_scale, [0, padding[1], 0, padding[0]], mode='replicate')
        self.im_patches = im_patches

        patch_coords = torch.Tensor([[0, 0, self.img_sample_sz[0]/self.scale_factor, self.img_sample_sz[1]/self.scale_factor]])

        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)

        return backbone_feat, patch_coords, im_patches

    def get_backbone_head_feat(self, backbone_feat):
        with torch.no_grad():
            return self.net.get_backbone_head_feat(backbone_feat)

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = OrderedDict({obj_id: self.get_iounet_box(self.pos[obj_id], self.target_sz[obj_id], self.scale_factor) for obj_id in self.pos.keys()})
        init_target_boxes = torch.zeros(self.max_num_objs, 4).to(self.params.device)
        for obj_id, box in self.classifier_target_box.items():
            init_target_boxes[obj_id - 1] = box.to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, self.max_num_objs, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_target_labels(self, train_x: TensorList):
        self.target_labels = TensorList([x.new_zeros(self.params.sample_memory_size, self.max_num_objs,
                                                     x.shape[2] + (int(self.kernel_size[0].item()) + 1) % 2,
                                                     x.shape[3] + (int(self.kernel_size[1].item()) + 1) % 2)
                                         for x in train_x])
        # Output sigma factor
        output_sigma_factor = self.params.get('output_sigma_factor', 1/4)
        self.sigma = output_sigma_factor/self.params.search_area_scale*self.feature_sz.prod().sqrt().item()*torch.ones(2)


        # Center pos in normalized img_coords
        for label, x in zip(self.target_labels, train_x):
            for obj_id in self.pos.keys():
                target_center_norm = (self.pos[obj_id]*self.scale_factor - self.img_support_sz/2)/self.img_support_sz
                ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
                center_pos = self.feature_sz * target_center_norm + 0.5 * ksz_even
                label[0, obj_id - 1, ...] = dcf.label_function_spatial(self.feature_sz, self.sigma, center_pos, end_pad=ksz_even)

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

    def update_memory(self, sample_x: TensorList, sample_ys_dict, target_boxes_dict, learning_rate = None):
        sample_y = TensorList([torch.zeros(1, self.target_labels[0].shape[1], self.target_labels[0].shape[2], self.target_labels[0].shape[3], device=sample_x[0].device)])
        target_box = torch.zeros(self.target_boxes.shape[1], self.target_boxes.shape[2], device=sample_x[0].device)

        for oid, label in sample_ys_dict.items():
            for s, l in zip(sample_y, label):
                s[:, oid-1:oid] = l

        for oid, box in target_boxes_dict.items():
            target_box[oid-1:oid] = box

        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        for y_memory, y, ind in zip(self.target_labels, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y

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

    def get_label_function(self, pos, scale_factor):
        train_y = TensorList()
        # target_center_norm = (pos - sample_pos) / (sample_scale * self.img_support_sz)
        target_center_norm = (pos*scale_factor - self.img_support_sz/2)/self.img_support_sz

        for sig, sz, ksz in zip([self.sigma], [self.feature_sz], [self.kernel_size]):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center = sz * target_center_norm + 0.5*ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))

        return train_y

    def get_iounet_box(self, pos, sz, scale_factor):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        target_ul = torch.Tensor([pos[1] - (sz[1] - 1)/2, pos[0] - (sz[0] - 1)/2, sz[1], sz[0]])
        target_ul = target_ul.unsqueeze(0)*scale_factor
        return target_ul

    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_backbone_head_feat(init_backbone_feat)
        x = x['layer3'] if 'layer3' in x else x['2']

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = getattr(self.net.head.filter_predictor, 'filter_size', 1)
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Get target boxes for the different augmentations
        self.init_target_boxes()

        # Get target labels for the different augmentations
        self.init_target_labels(TensorList([x]))

        if hasattr(self.net.head.filter_predictor, 'num_gth_frames'):
            self.net.head.filter_predictor.num_gth_frames = self.num_gth_frames

        self.init_memory(TensorList([x]))

    def visdom_draw_tracking(self, image, boxes, segmentation=None):
        parsed_boxes = []
        for bb in boxes:
            if isinstance(bb, (OrderedDict, dict)):
                parsed_boxes.extend([v for k, v in bb.items()])
            else:
                parsed_boxes.append(bb)
        self.visdom.register((image, *parsed_boxes), 'Tracking', 1, 'Tracking')
