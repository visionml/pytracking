from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
from pytracking import complex, dcf, fourier, TensorList
from pytracking.libs.tensorlist import tensor_operation
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor
from pytracking.libs.optimization import GaussNewtonCG
from .optim import FilterOptim, FactorizedConvProblem
from pytracking.features import augmentation



class ECO(BaseTracker):

    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.features.initialize()
        self.features_initialized = True


    def initialize(self, image, info: dict) -> dict:
        state = info['init_bbox']

        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize features
        self.initialize_features()

        # Chack if image is color
        self.params.features.set_is_color(image.shape[2] == 3)

        # Get feature specific params
        self.fparams = self.params.features.get_fparams('feature_params')

        # Get position and size
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Set search area
        self.target_scale = 1.0
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        if search_area > self.params.max_image_sample_size:
            self.target_scale =  math.sqrt(search_area / self.params.max_image_sample_size)
        elif search_area < self.params.min_image_sample_size:
            self.target_scale =  math.sqrt(search_area / self.params.min_image_sample_size)

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Use odd square search area and set sizes
        feat_max_stride = max(self.params.features.stride())
        self.img_sample_sz = torch.round(torch.sqrt(torch.prod(self.base_target_sz * self.params.search_area_scale))) * torch.ones(2)
        self.img_sample_sz += feat_max_stride - self.img_sample_sz % (2 * feat_max_stride)

        # Set other sizes (corresponds to ECO code)
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = self.params.features.size(self.img_sample_sz)
        self.filter_sz = self.feature_sz + (self.feature_sz + 1) % 2
        self.output_sz = self.params.score_upsample_factor * self.img_support_sz    # Interpolated size of the output
        self.compressed_dim = self.fparams.attribute('compressed_dim')

        # Number of filters
        self.num_filters = len(self.filter_sz)

        # Get window function
        self.window = TensorList([dcf.hann2d(sz).to(self.params.device) for sz in self.feature_sz])

        # Get interpolation function
        self.interp_fs = TensorList([dcf.get_interp_fourier(sz, self.params.interpolation_method,
                                                self.params.interpolation_bicubic_a, self.params.interpolation_centering,
                                                self.params.interpolation_windowing, self.params.device) for sz in self.filter_sz])

        # Get regularization filter
        self.reg_filter = TensorList([dcf.get_reg_filter(self.img_support_sz, self.base_target_sz, fparams).to(self.params.device)
                                      for fparams in self.fparams])
        self.reg_energy = self.reg_filter.view(-1) @ self.reg_filter.view(-1)

        # Get label function
        output_sigma_factor = self.fparams.attribute('output_sigma_factor')
        sigma = (self.filter_sz / self.img_support_sz) * torch.sqrt(self.base_target_sz.prod()) * output_sigma_factor
        self.yf = TensorList([dcf.label_function(sz, sig).to(self.params.device) for sz, sig in zip(self.filter_sz, sigma)])

        # Optimization options
        self.params.precond_learning_rate = self.fparams.attribute('learning_rate')
        if self.params.CG_forgetting_rate is None or max(self.params.precond_learning_rate) >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (1 - max(self.params.precond_learning_rate))**self.params.CG_forgetting_rate


        # Convert image
        im = numpy_to_torch(image)

        # Setup bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        x = self.generate_init_samples(im)

        # Initialize projection matrix
        x_mat = TensorList([e.permute(1,0,2,3).reshape(e.shape[1], -1).clone() for e in x])
        x_mat -= x_mat.mean(dim=1, keepdim=True)
        cov_x = x_mat @ x_mat.t()
        self.projection_matrix = TensorList([torch.svd(C)[0][:,:cdim].clone() for C, cdim in zip(cov_x, self.compressed_dim)])

        # Transform to get the training sample
        train_xf = self.preprocess_sample(x)

        # Shift the samples back
        if 'shift' in self.params.augmentation:
            for xf in train_xf:
                if xf.shape[0] == 1:
                    continue
                for i, shift in enumerate(self.params.augmentation['shift']):
                    shift_samp = 2 * math.pi * torch.Tensor(shift) / self.img_support_sz
                    xf[1+i:2+i,...] = fourier.shift_fs(xf[1+i:2+i,...], shift=shift_samp)

        # Shift sample
        shift_samp = 2*math.pi * (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)
        train_xf = fourier.shift_fs(train_xf, shift=shift_samp)

        # Initialize first-frame training samples
        num_init_samples = train_xf.size(0)
        self.init_sample_weights = TensorList([xf.new_ones(1) / xf.shape[0] for xf in train_xf])
        self.init_training_samples = train_xf.permute(2, 3, 0, 1, 4)


        # Sample counters and weights
        self.num_stored_samples = num_init_samples
        self.previous_replace_ind = [None]*len(self.num_stored_samples)
        self.sample_weights = TensorList([xf.new_zeros(self.params.sample_memory_size) for xf in train_xf])
        for sw, init_sw, num in zip(self.sample_weights, self.init_sample_weights, num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [xf.new_zeros(xf.shape[2], xf.shape[3], self.params.sample_memory_size, cdim, 2) for xf, cdim in zip(train_xf, self.compressed_dim)])

        # Initialize filter
        self.filter = TensorList(
            [xf.new_zeros(1, cdim, xf.shape[2], xf.shape[3], 2) for xf, cdim in zip(train_xf, self.compressed_dim)])

        # Do joint optimization
        self.joint_problem = FactorizedConvProblem(self.init_training_samples, self.yf, self.reg_filter, self.projection_matrix, self.params, self.init_sample_weights)
        joint_var = self.filter.concat(self.projection_matrix)
        self.joint_optimizer = GaussNewtonCG(self.joint_problem, joint_var, debug=(self.params.debug>=1), visdom=self.visdom)

        if self.params.update_projection_matrix:
            self.joint_optimizer.run(self.params.init_CG_iter // self.params.init_GN_iter, self.params.init_GN_iter)

        # Re-project samples with the new projection matrix
        compressed_samples = complex.mtimes(self.init_training_samples, self.projection_matrix)
        for train_samp, init_samp in zip(self.training_samples, compressed_samples):
            train_samp[:,:,:init_samp.shape[2],:,:] = init_samp

        # Initialize optimizer
        self.filter_optimizer = FilterOptim(self.params, self.reg_energy)
        self.filter_optimizer.register(self.filter, self.training_samples, self.yf, self.sample_weights, self.reg_filter)
        self.filter_optimizer.sample_energy = self.joint_problem.sample_energy
        self.filter_optimizer.residuals = self.joint_optimizer.residuals.clone()

        if not self.params.update_projection_matrix:
            self.filter_optimizer.run(self.params.init_CG_iter)

        # Post optimization
        self.filter_optimizer.run(self.params.post_init_CG_iter)

        self.symmetrize_filter()



    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.pos.round()
        sample_scales = self.target_scale * self.params.scale_factors
        test_xf = self.extract_fourier_sample(im, self.pos, sample_scales, self.img_sample_sz)

        # Compute scores
        sf = self.apply_filter(test_xf)
        translation_vec, scale_ind, s = self.localize_target(sf)
        scale_change_factor = self.params.scale_factors[scale_ind]

        # Update position and scale
        self.update_state(sample_pos + translation_vec, self.target_scale * scale_change_factor)

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()
        self.debug_info['max_score'] = max_score

        if self.visdom is not None:
            self.visdom.register(score_map, 'heatmap', 2, 'Score Map')
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
        elif self.params.debug >= 2:
            show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

        # if self.params.debug >= 3:
        #     for i, hf in enumerate(self.filter):
        #         show_tensor(fourier.sample_fs(hf).abs().mean(1), 6+i)


        # ------- UPDATE ------- #

        # Get train sample
        train_xf = TensorList([xf[scale_ind:scale_ind+1, ...] for xf in test_xf])

        # Shift the sample
        shift_samp = 2*math.pi * (self.pos - sample_pos) / (sample_scales[scale_ind] * self.img_support_sz)
        train_xf = fourier.shift_fs(train_xf, shift=shift_samp)

        # Update memory
        self.update_memory(train_xf)

        # Train filter
        if self.frame_num % self.params.train_skipping == 1:
            self.filter_optimizer.run(self.params.CG_iter, train_xf)
            self.symmetrize_filter()

        # Return new state
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        out = {'target_bbox': new_state.tolist()}
        return out


    def apply_filter(self, sample_xf: TensorList) -> torch.Tensor:
        return complex.mult(self.filter, sample_xf).sum(1, keepdim=True)

    def localize_target(self, sf: TensorList):
        if self.params.score_fusion_strategy == 'sum':
            scores = fourier.sample_fs(fourier.sum_fs(sf), self.output_sz)
        elif self.params.score_fusion_strategy == 'weightedsum':
            weight = self.fparams.attribute('translation_weight')
            scores = fourier.sample_fs(fourier.sum_fs(weight * sf), self.output_sz)
        elif self.params.score_fusion_strategy == 'transcale':
            alpha = self.fparams.attribute('scale_weight')
            beta = self.fparams.attribute('translation_weight')
            sample_sz = torch.round(self.output_sz.view(1,-1) * self.params.scale_factors.view(-1,1))
            scores = 0
            for sfe, a, b in zip(sf, alpha, beta):
                sfe = fourier.shift_fs(sfe, math.pi*torch.ones(2))
                scores_scales = []
                for sind, sz in enumerate(sample_sz):
                    pd = (self.output_sz-sz)/2
                    scores_scales.append(F.pad(fourier.sample_fs(sfe[sind:sind+1,...], sz),
                                        (math.floor(pd[1].item()), math.ceil(pd[1].item()),
                                         math.floor(pd[0].item()), math.ceil(pd[0].item()))))
                scores_cat = torch.cat(scores_scales)
                scores = scores + (b - a) * scores_cat.mean(dim=0, keepdim=True) + a * scores_cat
        else:
            raise ValueError('Unknown score fusion strategy.')

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp.float().cpu()

        # Convert to displacements in the base scale
        if self.params.score_fusion_strategy in ['sum', 'weightedsum']:
            disp = (max_disp + self.output_sz / 2) % self.output_sz - self.output_sz / 2
        elif self.params.score_fusion_strategy == 'transcale':
            disp = max_disp - self.output_sz / 2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind, ...].view(-1) * (self.img_support_sz / self.output_sz) * self.target_scale
        if self.params.score_fusion_strategy in ['sum', 'weightedsum']:
            translation_vec *= self.params.scale_factors[scale_ind]

        return translation_vec, scale_ind, scores


    def extract_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        return self.params.features.extract(im, pos, scales, sz)[0]

    def extract_fourier_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> TensorList:
        x = self.extract_sample(im, pos, scales, sz)
        return self.preprocess_sample(self.project_sample(x))

    def preprocess_sample(self, x: TensorList) -> TensorList:
        x *= self.window
        sample_xf = fourier.cfft2(x)
        return TensorList([dcf.interpolate_dft(xf, bf) for xf, bf in zip(sample_xf, self.interp_fs)])

    def project_sample(self, x: TensorList):
        @tensor_operation
        def _project_sample(x: torch.Tensor, P: torch.Tensor):
            if P is None:
                return x
            return torch.matmul(x.permute(2, 3, 0, 1), P).permute(2, 3, 0, 1)

        return _project_sample(x, self.projection_matrix)

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        # Do data augmentation
        transforms = [augmentation.Identity()]
        if 'shift' in self.params.augmentation:
            transforms.extend([augmentation.Translation(shift) for shift in self.params.augmentation['shift']])
        if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
            transforms.append(augmentation.FlipHorizontal())
        if 'rotate' in self.params.augmentation:
            transforms.extend([augmentation.Rotate(angle) for angle in self.params.augmentation['rotate']])
        if 'blur' in self.params.augmentation:
            transforms.extend([augmentation.Blur(sigma) for sigma in self.params.augmentation['blur']])

        init_samples = self.params.features.extract_transformed(im, self.pos, self.target_scale, self.img_sample_sz, transforms)

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        if 'dropout' in self.params.augmentation:
            num, prob = self.params.augmentation['dropout']
            for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
                if use_aug:
                    init_samples[i] = torch.cat([init_samples[i], F.dropout2d(init_samples[i][0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        return init_samples


    def update_memory(self, sample_xf: TensorList):
        # Update weights and get index to replace
        replace_ind = self.update_sample_weights()
        for train_samp, xf, ind in zip(self.training_samples, sample_xf, replace_ind):
            train_samp[:,:,ind:ind+1,:,:] = xf.permute(2, 3, 0, 1, 4)


    def update_sample_weights(self):
        replace_ind = []
        for sw, prev_ind, num_samp, fparams in zip(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.fparams):
            if num_samp == 0 or fparams.learning_rate == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw, 0)
                r_ind = r_ind.item()

                # Update weights
                if prev_ind is None:
                    sw /= 1 - fparams.learning_rate
                    sw[r_ind] = fparams.learning_rate
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - fparams.learning_rate)

            sw /= sw.sum()
            replace_ind.append(r_ind)

        self.previous_replace_ind = replace_ind.copy()
        self.num_stored_samples += 1
        return replace_ind

    def update_state(self, new_pos, new_scale):
        # Update scale
        self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
        self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def symmetrize_filter(self):
        for hf in self.filter:
            hf[:,:,:,0,:] /= 2
            hf[:,:,:,0,:] += complex.conj(hf[:,:,:,0,:].flip((2,)))