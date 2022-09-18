from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def parameters():
    params = TrackerParams()

    ##########################################
    # General parameters
    ##########################################

    params.debug = 0
    params.visualization = False
    params.multiobj_mode = 'parallel'
    params.use_gpu = True

    ##########################################
    # Bounding box init network
    ##########################################
    params.sta_image_sample_size = (30 * 16, 52 * 16)
    params.sta_search_area_scale = 4.0

    params.sta_net = NetWithBackbone(net_path='sta.pth.tar',
                                     use_gpu=params.use_gpu,
                                     image_format='bgr255',
                                     mean=[102.9801, 115.9465, 122.7717],
                                     std=[1.0, 1.0, 1.0]
                                     )

    params.sta_net.load_network()

    ##########################################
    # Segmentation Branch parameters
    ##########################################
    params.seg_to_bb_mode = 'var'
    params.min_mask_area = 100

    params.image_sample_size = (30 * 16, 52 * 16)
    params.search_area_scale = 6.0
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = None
    params.max_scale_change = (0.8, 1.2)

    # Learning parameters
    params.sample_memory_size = 32
    params.learning_rate = 0.1
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_target_model = True
    params.net_opt_iter = 20
    params.net_opt_update_iter = 3

    # Main network
    params.net = NetWithBackbone(net_path='rts50.pth',
                                 use_gpu=params.use_gpu,
                                 image_format='bgr255',
                                 mean=[102.9801, 115.9465, 122.7717],
                                 std=[1.0, 1.0, 1.0],
                                 clf_filter_size=4,
                                 fusion_type="add"
                                 )
    params.net.load_network()
    
    ##########################################
    # Classifier Branch parameters
    ##########################################

    # General parameters
    params.clf_image_sample_size = params.image_sample_size
    params.clf_search_area_scale = params.search_area_scale
    params.clf_border_mode = params.border_mode
    params.clf_patch_max_scale_change = params.patch_max_scale_change

    # Learning parameters
    params.clf_sample_memory_size = 50
    params.clf_learning_rate = 0.01
    params.clf_train_skipping = 20

    # Net optimization
    params.update_classifier = True
    params.clf_net_opt_iter = 10
    params.clf_net_opt_update_iter = 2
    params.clf_net_opt_hn_iter = 1
    params.clf_output_sigma_factor = 0.25

    # Advanced localization parameters
    params.clf_advanced_localization = True
    params.clf_target_not_found_threshold = 0.30
    params.clf_target_not_found_threshold_too_small = 0.50
    params.clf_distractor_threshold = 10000
    params.clf_hard_negative_threshold = 10000
    params.clf_target_neighborhood_scale = 2.2
    params.clf_displacement_scale = 0.8
    params.clf_hard_negative_learning_rate = 0.02

    # Augmentations parameters
    params.clf_use_augmentation = True
    params.clf_augmentation = {
        'fliplr': True,
        'blur': [(3, 1), (1, 3), (2, 2)],
    }

    return params
