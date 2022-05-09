import torch.optim as optim
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import tompnet
import ltr.models.loss as ltr_losses
import ltr.actors.tracking as actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.models.loss.bbr_loss import GIoULoss


def run(settings):
    settings.description = 'ToMP50'
    settings.batch_size = 36
    settings.num_workers = 16
    settings.multi_gpu = True

    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 1
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 0., 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0., 'test': 0.5}
    settings.hinge_threshold = 0.05
    settings.num_train_frames = 2
    settings.num_test_frames = 1
    settings.num_encoder_layers = 6
    settings.num_decoder_layers = 6
    settings.frozen_backbone_layers = ['conv1', 'bn1', 'layer1', 'layer2']
    settings.freeze_backbone_bn_layers = True

    settings.crop_type = 'inside_major'
    settings.max_scale_change = 1.5
    settings.max_gap = 200
    settings.train_samples_per_epoch = 40000
    settings.val_samples_per_epoch = 10000
    settings.val_epoch_interval = 5
    settings.num_epochs = 300

    settings.weight_giou = 1.0
    settings.weight_clf = 100.0
    settings.normalized_bbreg_coords = True
    settings.center_sampling_radius = 1.0
    settings.use_test_frame_encoding = False  # Set to True to use the same as in the paper but is less stable to train.

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    coco_train = MSCOCOSeq(settings.env.coco_dir)

    # Validation datasets
    got10k_val = Got10k(settings.env.got10k_dir, split='votval')


    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}

    data_processing_train = processing.LTRBDenseRegressionProcessing(search_area_factor=settings.search_area_factor,
                                                                     output_sz=settings.output_sz,
                                                                     center_jitter_factor=settings.center_jitter_factor,
                                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                                     crop_type=settings.crop_type,
                                                                     max_scale_change=settings.max_scale_change,
                                                                     mode='sequence',
                                                                     label_function_params=label_params,
                                                                     transform=transform_train,
                                                                     joint_transform=transform_joint,
                                                                     use_normalized_coords=settings.normalized_bbreg_coords,
                                                                     center_sampling_radius=settings.center_sampling_radius)

    data_processing_val = processing.LTRBDenseRegressionProcessing(search_area_factor=settings.search_area_factor,
                                                                   output_sz=settings.output_sz,
                                                                   center_jitter_factor=settings.center_jitter_factor,
                                                                   scale_jitter_factor=settings.scale_jitter_factor,
                                                                   crop_type=settings.crop_type,
                                                                   max_scale_change=settings.max_scale_change,
                                                                   mode='sequence',
                                                                   label_function_params=label_params,
                                                                   transform=transform_val,
                                                                   joint_transform=transform_joint,
                                                                   use_normalized_coords=settings.normalized_bbreg_coords,
                                                                   center_sampling_radius=settings.center_sampling_radius)

    # Train sampler and loader
    dataset_train = sampler.DiMPSampler([lasot_train, got10k_train, trackingnet_train, coco_train], [1, 1, 1, 1],
                                        samples_per_epoch=settings.train_samples_per_epoch, max_gap=settings.max_gap,
                                        num_test_frames=settings.num_test_frames, num_train_frames=settings.num_train_frames,
                                        processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    dataset_val = sampler.DiMPSampler([got10k_val], [1], samples_per_epoch=settings.val_samples_per_epoch,
                                      max_gap=settings.max_gap, num_test_frames=settings.num_test_frames,
                                      num_train_frames=settings.num_train_frames, processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=settings.val_epoch_interval, stack_dim=1)

    # Create network and actor
    net = tompnet.tompnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True, head_feat_blocks=0,
                            head_feat_norm=True, final_conv=True, out_feature_dim=256, feature_sz=settings.feature_sz,
                            frozen_backbone_layers=settings.frozen_backbone_layers,
                            num_encoder_layers=settings.num_encoder_layers,
                            num_decoder_layers=settings.num_decoder_layers,
                            use_test_frame_encoding=settings.use_test_frame_encoding)


    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {'giou': GIoULoss(), 'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold)}

    loss_weight = {'giou': settings.weight_giou, 'test_clf': settings.weight_clf}

    actor = actors.ToMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    optimizer = optim.AdamW([
        {'params': actor.net.head.parameters(), 'lr': 1e-4},
        {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 2e-5}
    ], lr=2e-4, weight_decay=0.0001)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler,
                         freeze_backbone_bn_layers=settings.freeze_backbone_bn_layers)

    trainer.train(settings.num_epochs, load_latest=True, fail_safe=True)
