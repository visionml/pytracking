from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.seg_to_bb_mode = 'var'
    params.max_scale_change = (0.95, 1.1)
    params.min_mask_area = 100

    params.return_raw_scores = True
    params.use_gpu = True

    params.image_sample_size = (30 * 16, 52 * 16)
    params.search_area_scale = 4.0
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = None

    # Learning parameters
    params.use_merged_mask_for_memory = True
    params.sample_memory_size = 32
    params.learning_rate = 0.1
    params.init_samples_minimum_weight = 0
    params.train_skipping = 1
    params.train_sample_interval = 1
    params.test_sample_interval = 5
    params.test_num_frames = 9

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 5
    params.net_opt_update_iter = 15
    params.net_opt_hn_iter = 1

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = False
    params.augmentation = {}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # IoUnet parameters
    params.use_iou_net = False      # Use the augmented samples to compute the modulation vector

    params.net = NetWithBackbone(net_path='STANet_ep0080.pth.tar',
                                 use_gpu=params.use_gpu,
                                 image_format='bgr255',
                                 mean=[102.9801, 115.9465, 122.7717],
                                 std=[1.0, 1.0, 1.0]
                                 )

    params.vot_anno_conversion_type = 'preserve_area'

    return params




    
