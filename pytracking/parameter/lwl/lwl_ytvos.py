from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.seg_to_bb_mode = 'var'
    params.max_scale_change = (0.95, 1.1)
    params.min_mask_area = 100

    params.use_gpu = True

    params.image_sample_size = (30 * 16, 52 * 16)
    params.search_area_scale = 5.0
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = None

    # Learning parameters
    params.sample_memory_size = 32
    params.learning_rate = 0.1
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 1

    # Net optimization params
    params.update_target_model = True
    params.net_opt_iter = 20
    params.net_opt_update_iter = 3

    params.net = NetWithBackbone(net_path='lwl_stage2.pth',
                                 use_gpu=params.use_gpu,
                                 image_format='bgr255',
                                 mean=[102.9801, 115.9465, 122.7717],
                                 std=[1.0, 1.0, 1.0]
                                 )

    return params
