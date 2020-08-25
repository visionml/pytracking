import torch.optim as optim
from ltr.dataset import YouTubeVOS, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
import ltr.models.lwtl.lwtl_box_net as lwtl_box
import ltr.actors.segmentation as lwtl_actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.models.loss.segmentation import LovaszSegLoss
import ltr.admin.loading as network_loading


def run(settings):
    settings.description = 'Default train settings for training VOS with box initialization.'
    settings.batch_size = 8
    settings.num_workers = 4
    settings.multi_gpu = False
    settings.print_interval = 1
    settings.normalize_mean = [102.9801, 115.9465, 122.7717]
    settings.normalize_std = [1.0, 1.0, 1.0]

    settings.feature_sz = (52, 30)
    settings.output_sz = (settings.feature_sz[0] * 16, settings.feature_sz[1] * 16)
    settings.search_area_factor = 5.0
    settings.crop_type = 'inside_major'
    settings.max_scale_change = None
    settings.device = "cuda:0"
    settings.center_jitter_factor = {'train': 3, 'test': (5.5, 4.5)}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}

    settings.min_target_area = 500

    ytvos_train = YouTubeVOS(version="2019", multiobj=False, split='jjtrain')
    ytvos_valid = YouTubeVOS(version="2019", multiobj=False, split='jjvalid')
    coco_train = MSCOCOSeq()

    # Data transform
    transform_joint = tfm.Transform(tfm.ToBGR(),
                                    tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2, normalize=False),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=False),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    data_processing_train = processing.LWTLProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      crop_type=settings.crop_type,
                                                      max_scale_change=settings.max_scale_change,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint,
                                                      new_roll=True)

    data_processing_val = processing.LWTLProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    crop_type=settings.crop_type,
                                                    max_scale_change=settings.max_scale_change,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint,
                                                    new_roll=True)
    # Train sampler and loader
    dataset_train = sampler.LWTLSampler([ytvos_train, coco_train], [1, 1],
                                        samples_per_epoch=settings.batch_size * 1000, max_gap=100,
                                        num_test_frames=1,
                                        num_train_frames=1,
                                        processing=data_processing_train)
    dataset_val = sampler.LWTLSampler([ytvos_valid], [1],
                                      samples_per_epoch=settings.batch_size * 100, max_gap=100,
                                      num_test_frames=1,
                                      num_train_frames=1,
                                      processing=data_processing_val)

    loader_train = LTRLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                             stack_dim=1, batch_size=settings.batch_size)
    loader_val = LTRLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                           epoch_interval=5, stack_dim=1, batch_size=settings.batch_size)

    net = lwtl_box.steepest_descent_resnet50(filter_size=3, num_filters=16, optim_iter=5,
                                             backbone_pretrained=True,
                                             out_feature_dim=512,
                                             frozen_backbone_layers=['conv1', 'bn1', 'layer1'],
                                             label_encoder_dims=(16, 32, 64),
                                             use_bn_in_label_enc=False,
                                             clf_feat_blocks=0,
                                             final_conv=True,
                                             backbone_type='mrcnn',
                                             box_label_encoder_dims=(64, 64,),
                                             final_bn=False)

    base_net_weights = network_loading.load_trained_network(settings.env.workspace_dir,
                                                            'ltr/segm/lwl_stage2/LWTLNet_ep0080.pth.tar')

    # Copy weights
    net.feature_extractor.load_state_dict(base_net_weights.feature_extractor.state_dict())
    net.target_model.load_state_dict(base_net_weights.target_model.state_dict())
    net.decoder.load_state_dict(base_net_weights.decoder.state_dict())
    net.label_encoder.load_state_dict(base_net_weights.label_encoder.state_dict())

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {
        'segm':  LovaszSegLoss(per_image=False),
    }

    loss_weight = {
        'segm': 100.0,
        'segm_box': 10.0,
        'segm_train': 10,
    }

    actor = lwtl_actors.LWTLBoxActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    optimizer = optim.Adam([{'params': actor.net.box_label_encoder.parameters(), 'lr': 1e-3}],
                           lr=2e-4)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(50, load_latest=True, fail_safe=True)
