from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from pytracking import TensorList
from pytracking.features.preprocessing_sta import numpy_to_torch, torch_to_numpy
from pytracking.features.preprocessing_sta import sample_patch_multiscale, sample_patch_transformed, sample_patch
from pytracking.features import augmentation
from collections import OrderedDict

def plot_image(image, savePath="out.png"):
    from PIL import Image, ImageDraw
    ##image shape: h, w, 1
    im = Image.fromarray(image.astype(np.uint8))
    drawer = ImageDraw.Draw(im)
    im.save(savePath)
    print("image saved")

class STA(BaseTracker):
    multiobj_mode = 'parallel'

    def predicts_segmentation_mask(self):
        return True

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Learn the initial target model. Initialize memory etc.
        self.frame_num = 0
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The segmentation network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Get target position and size
        state = info['init_bbox']
        init_bbox = info.get('init_bbox', None)

        if init_bbox is not None:
            # shape 1, 1, 4 (frames, seq, 4)
            init_bbox = torch.tensor(init_bbox).unsqueeze(0).unsqueeze(0).float()

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
        self.search_area_scale = [self.params.search_area_scale] if isinstance(self.params.search_area_scale, (int, float)) else self.params.search_area_scale
        search_area = [torch.prod(self.target_sz * s).item() for s in self.search_area_scale]
        self.target_scale = [math.sqrt(s) / self.img_sample_sz.prod().sqrt() for s in search_area]

        # Convert image
        im = numpy_to_torch(image)

        # Extract and transform sample
        self.feature_sz = self.img_sample_sz / 16
        ksz = self.net.target_model.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        im_patches_ms = []
        init_bboxes_ms = []
        for p, t in zip(self.get_centered_sample_pos(), self.target_scale):
            _, _, im_patches, init_bboxes = self.extract_backbone_features(im, init_bbox, p, t, self.img_sample_sz)
            im_patches_ms.append(im_patches.unsqueeze(0))
            init_bboxes_ms.append(init_bboxes.unsqueeze(0))
        
        self.init_memory(im_patches_ms, init_bboxes_ms)
        out = {'time': time.time() - tic}
        
        # If object is visible in the i-th aved frame
        self.visible = [1]

        return out

    def store_seq(self, image, bbox, info):
        if self.object_id is None:
            bbox = bbox
        else:
            bbox = bbox[self.object_id]
        if bbox[0] == -1 and bbox[1] == -1 and bbox[2] == -1 and bbox[3] == -1:
            self.visible.append(0)
        else:
            self.visible.append(1)
        
        self.pos = torch.Tensor([bbox[1] + (bbox[3] - 1)/2, bbox[0] + (bbox[2] - 1)/2])
        self.target_sz = torch.Tensor([bbox[3], bbox[2]])
        bbox = torch.tensor(bbox).unsqueeze(0).unsqueeze(0).float()

        search_area = [torch.prod(self.target_sz * s).item() for s in self.search_area_scale]
        self.target_scale = [math.sqrt(s) / self.img_sample_sz.prod().sqrt() for s in search_area]

        # Convert image
        im = numpy_to_torch(image)

        # Extract backbone features
        im_patches_ms = []
        init_bboxes_ms = []
        for p, t in zip(self.get_centered_sample_pos(), self.target_scale):
            _, _, im_patches, patch_bboxes = self.extract_backbone_features(im, bbox, p, t, self.img_sample_sz)
            im_patches_ms.append(im_patches.unsqueeze(0))
            init_bboxes_ms.append(patch_bboxes.unsqueeze(0))

        # Update the tracker memory
        self.update_memory(im_patches_ms, init_bboxes_ms)
        
    def track(self, image, bbox, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        if self.object_id is None:
            bbox = bbox
        else:
            bbox = bbox[self.object_id]
        
        if bbox[0] == -1 and bbox[1] == -1 and bbox[2] == -1 and bbox[3] == -1:
            segmentation_mask_im = np.full(image.shape[:2], 0)
            segmentation_output = np.full(image.shape[:2], -100.0)
            if self.object_id is None:
                segmentation_output = 1 / (1 + np.exp(-segmentation_output))
            out = {'segmentation': segmentation_mask_im, 'target_bbox': bbox,
               'segmentation_raw': segmentation_output}
            return out

        self.pos = torch.Tensor([bbox[1] + (bbox[3] - 1)/2, bbox[0] + (bbox[2] - 1)/2])
        self.target_sz = torch.Tensor([bbox[3], bbox[2]])
        bbox = torch.tensor(bbox).unsqueeze(0).unsqueeze(0).float()

        search_area = [torch.prod(self.target_sz * s).item() for s in self.search_area_scale]
        self.target_scale = [math.sqrt(s) / self.img_sample_sz.prod().sqrt() for s in search_area]

        # ********************************************************************** #
        # ---------- Predict segmentation mask for the current frame ----------- #
        # ********************************************************************** #

        # Convert image
        im = numpy_to_torch(image)

        segmentation_scores_im_ms = []
        for i, (p, t) in enumerate(zip(self.get_centered_sample_pos(), self.target_scale)):
            _, sample_coords, im_patches, patch_bboxes = self.extract_backbone_features(im, bbox, p, t, self.img_sample_sz)
        
            # predict segmentation masks
            segmentation_scores = self.update_target_model(im_patches, patch_bboxes, i)
        
            # Location of sample
            sample_pos, sample_scale = self.get_sample_location(sample_coords)

            # Get the segmentation scores for the full image.
            # Regions outside the search region are assigned low scores (-100)
            segmentation_scores_im_ms.append(self.convert_scores_crop_to_image(segmentation_scores, im, sample_scale, sample_pos))
        segmentation_scores_im_ms = torch.stack(segmentation_scores_im_ms, dim=0)
        segmentation_scores_im = torch.mean(segmentation_scores_im_ms, dim=0)
        
        bbox = bbox.round().squeeze(0).squeeze(0).numpy().astype(np.int)
        segmentation_scores_im[..., :bbox[0]] = -100
        segmentation_scores_im[..., bbox[0]+bbox[2]:] = -100
        segmentation_scores_im[..., :bbox[1], :] = -100
        segmentation_scores_im[..., bbox[1]+bbox[3]:, :] = -100

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
        return [self.pos + ((self.feature_sz + self.kernel_size) % 2) * t * \
               self.img_support_sz / (2*self.feature_sz) for t in self.target_scale]

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

    def segment_target(self, bbox_mask, sample_tm_feat, sample_x, segm=False):
        with torch.no_grad():
            segmentation_scores = self.net.segment_target_add_bbox_encoder(bbox_mask, self.target_filter, sample_tm_feat, sample_x, segm)

        return segmentation_scores

    def extract_backbone_features(self, im: torch.Tensor, bbox, pos: torch.Tensor, scale, sz: torch.Tensor):
        im_patches, patch_coords, patch_bboxes = sample_patch_multiscale(im, bbox, pos, scale.unsqueeze(0), sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords[0], im_patches[0], patch_bboxes[0].to(self.params.device)

    def get_target_model_features(self, backbone_feat):
        """ Extract features input to the target model"""
        with torch.no_grad():
            return self.net.extract_target_model_features(backbone_feat)

    def init_memory(self, train_x, bboxes):
        """ Initialize the sample memory used to update the target model """
        # Initialize memory
        self.training_samples = train_x
        self.target_bboxes = bboxes
        self.num_stored_samples = [x.shape[0] for x in self.training_samples]

    def update_memory(self, sample_x, bboxes):
        """ Add a new sample to the memory"""
        for i in range(len(self.training_samples)):
            self.training_samples[i] = torch.cat([self.training_samples[i], sample_x[i]], dim=0)
            self.target_bboxes[i] = torch.cat([self.target_bboxes[i], bboxes[i]], dim=0)
        self.num_stored_samples = [x.shape[0] for x in self.training_samples]

    def update_target_model(self, train_x, bbox, i, learning_rate=None):
        # Set flags and learning rate
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Decide the number of iterations to run
        num_iter = 0
        if (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        if num_iter > 0:
            samples = train_x.unsqueeze(0)
            bboxes = bbox.unsqueeze(0)

            sample_interval = self.params.get('test_sample_interval', 5)
            test_num_frames = self.params.get('test_num_frames', 1)
            num_append_left = (test_num_frames-1)//2
            num_append_right = (test_num_frames-1)//2

            while self.frame_num-1-num_append_left*sample_interval < 0:
                num_append_left -= 1
                num_append_right += 1
            while self.frame_num-1+num_append_right*sample_interval >= self.num_stored_samples[0]:
                num_append_left += 1
                num_append_right -= 1
            
            if self.params.get('casual', False) == True:
                num_append_left += num_append_right
                num_append_right = 0
            
            # add samples in the past
            ind = self.frame_num-1-sample_interval
            while ind >= 0 and num_append_left > 0:
                if self.visible[ind]:
                    samples = torch.cat((samples, self.training_samples[i][ind:ind+1]), dim=0)
                    bboxes = torch.cat((bboxes, self.target_bboxes[i][ind:ind+1]), dim=0)
                ind -= sample_interval
                num_append_left -= 1
            
            # add samples in the future
            ind = self.frame_num-1+sample_interval
            while ind < self.num_stored_samples[0] and num_append_right > 0:
                if self.visible[ind]:
                    samples = torch.cat((samples, self.training_samples[i][ind:ind+1]), dim=0)
                    bboxes = torch.cat((bboxes, self.target_bboxes[i][ind:ind+1]), dim=0)
                ind += sample_interval
                num_append_right -= 1
            
            # print("sample shape", samples.shape, flush=True)    
            with torch.no_grad():
                backbone_feat = self.net.extract_backbone(samples)
                test_x = self.get_target_model_features(backbone_feat)
                train_bbox_enc, _ = self.net.label_encoder(bboxes, test_x, list(self.params.image_sample_size))
                
                few_shot_label, few_shot_sw = self.net.bbox_encoder(bboxes, test_x.unsqueeze(1), list(self.params.image_sample_size))
                self.target_filter, _, _ = self.net.target_model.get_filter(test_x.unsqueeze(1), few_shot_label, few_shot_sw,
                                                                        num_iter=num_iter)        
                segmentation_scores = self.segment_target(train_bbox_enc, test_x, backbone_feat)
                train_segm_enc, train_segm_sw = self.net.segm_encoder(torch.sigmoid(segmentation_scores), test_x.unsqueeze(1))
                
                # print("train_segm_enc shape", train_segm_enc.shape, flush=True)
                # for i in range(train_segm_enc.shape[2]):
                #     vis_embedding = train_segm_enc[0:1,0,i:i+1,...]
                #     vis_embedding = F.interpolate(vis_embedding, size=self.params.image_sample_size, mode='bilinear')
                #     vis_embedding = vis_embedding[0,0,...].cpu().numpy()
                #     print("vis_embedding shape", vis_embedding.shape, flush=True)
                #     print("max value", np.max(vis_embedding), flush=True)
                #     print("min value", np.min(vis_embedding), flush=True)
                #     vis_embedding = vis_embedding / np.max(vis_embedding) * 255.0
                #     plot_image(vis_embedding, savePath="output"+str(i)+".jpg")
                # exit()

                self.target_filter, _, _ = self.net.target_model_segm.get_filter(test_x.unsqueeze(1), train_segm_enc, train_segm_sw,
                                                                    num_iter=num_iter)  
                segmentation_scores = self.segment_target(train_bbox_enc, test_x, backbone_feat, True)
                segmentation_score = segmentation_scores[0:1]

                augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}
                if 'fliplr' in augs:
                    sample_width = samples.shape[3]
                    samples = samples.flip((3))
                    bboxes[:, :, 0] = sample_width - bboxes[:, :, 0] - bboxes[:, :, 2]

                    backbone_feat = self.net.extract_backbone(samples)
                    test_x = self.get_target_model_features(backbone_feat)
                    train_bbox_enc, _ = self.net.label_encoder(bboxes, test_x, list(self.params.image_sample_size))
                    
                    few_shot_label, few_shot_sw = self.net.bbox_encoder(bboxes, test_x.unsqueeze(1), list(self.params.image_sample_size))
                    self.target_filter, _, _ = self.net.target_model.get_filter(test_x.unsqueeze(1), few_shot_label, few_shot_sw,
                                                                            num_iter=num_iter)        
                    segmentation_scores = self.segment_target(train_bbox_enc, test_x, backbone_feat)

                    train_segm_enc, train_segm_sw = self.net.segm_encoder(torch.sigmoid(segmentation_scores), test_x.unsqueeze(1))
                    self.target_filter, _, _ = self.net.target_model_segm.get_filter(test_x.unsqueeze(1), train_segm_enc, train_segm_sw,
                                                                        num_iter=num_iter)  
                    segmentation_scores = self.segment_target(train_bbox_enc, test_x, backbone_feat, True)
                    segmentation_score_flip = segmentation_scores[0:1]
                    segmentation_score = (segmentation_score + segmentation_score_flip.flip((3))) / 2.0
                
                return segmentation_score
