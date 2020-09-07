from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.image_sample_size = 18*16
    params.search_area_scale = 5

    # Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20
    params.output_sigma_factor = 1/4

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3,1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                           'dropout': (2, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Localization parameters
    params.window_output = True
    params.use_clipped_window = True
    params.effective_search_area = 10.0
    params.apply_window_to_dimp_score = True

    params.dimp_threshold = 0.05
    params.target_not_found_threshold_fused = 0.05

    params.reset_state_during_occlusion = False
    params.prev_feat_remove_subpixel_shift = True
    params.move_feat_to_center = True

    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.use_iou_net = True
    params.iounet_augmentation = False
    params.iounet_use_log_scale = True
    params.iounet_k = 3
    params.num_init_random_boxes = 9
    params.box_jitter_pos = 0.1
    params.box_jitter_sz = 0.5
    params.maximal_aspect_ratio = 6
    params.box_refinement_iter = 5
    params.box_refinement_step_length = 1
    params.box_refinement_step_decay = 1

    params.remove_offset_in_fused_score = True
    params.score_downsample_factor = 1

    params.net = NetWithBackbone(net_path='kys.pth',
                                 use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'
    return params
