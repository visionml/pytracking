import torch.optim as optim
from ltr.dataset import YouTubeVOS, Davis, Got10k, Got10kVOS, LasotVOS
from ltr.data import processing, sampler, LTRLoader
import ltr.models.rts.rts_net as rts_networks
import ltr.actors.segmentation as segm_actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import ltr.models.loss as ltr_losses
from ltr.admin.loading import load_pretrained

import os


def run(settings):
    settings.description = 'Default train settings for training full network'
    settings.batch_size = 15
    settings.num_workers = 45
    settings.multi_gpu = False # RAN WITH 1 A100 GPU
    settings.print_interval = 50
    settings.normalize_mean = [102.9801, 115.9465, 122.7717]
    settings.normalize_std = [1.0, 1.0, 1.0]

    settings.feature_sz = (52, 30)

    settings.output_sz = (settings.feature_sz[0] * 16, settings.feature_sz[1] * 16)
    settings.search_area_factor = 5.0
    settings.crop_type = 'inside_major'
    settings.max_scale_change = None

    settings.center_jitter_factor = {'train': 3, 'test': (5.5, 4.5)}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}

    settings.clf_encoder_add = True

    # The tracking pairs processing module
    settings.clf_feature_sz = (52//2, 30//2)
    settings.clf_target_filter_sz = 4
    settings.clf_hinge_threshold = 0.05
    settings.clf_output_sigma_factor = 1/4

    clf_output_sigma = settings.clf_output_sigma_factor / settings.search_area_factor
    label_params = {'feature_sz': settings.clf_feature_sz,
                    'sigma_factor': clf_output_sigma,
                    'kernel_sz': settings.clf_target_filter_sz}


    # Datasets
    ytvos_train = YouTubeVOS(version="2019", multiobj=False, split='jjtrain')
    davis_train = Davis(version='2017', multiobj=False, split='train')

    anno_path = os.path.join(settings.env.pregenerated_masks, "got10k_masks")
    got10k_train = Got10kVOS(anno_path=anno_path, split='vottrain')

    lasot_anno_path = os.path.join(settings.env.pregenerated_masks, "lasot_masks")
    lasot_train = LasotVOS(anno_path=lasot_anno_path, split='train')

    ytvos_val = YouTubeVOS(version="2019", multiobj=False, split='jjvalid')
    got10k_val = Got10k(settings.env.got10k_dir, split='votval')

    # Data transform
    transform_joint = tfm.Transform(
        tfm.ToBGR(),
        tfm.ToGrayscale(probability=0.05),
        tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(
        tfm.RandomAffine(p_flip=0.0, max_rotation=15.0,
                         max_shear=0.0, max_ar_factor=0.0,
                         max_scale=0.2, pad_amount=0),
        tfm.ToTensorAndJitter(0.2, normalize=False),
        tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(
        tfm.ToTensorAndJitter(0.0, normalize=False),
        tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    data_processing_train = processing.RTSProcessing(
        search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        mode='sequence',
        crop_type=settings.crop_type,
        max_scale_change=settings.max_scale_change,
        transform=transform_train,
        joint_transform=transform_joint,
        label_function_params=label_params,
        new_roll=True)

    data_processing_val = processing.RTSProcessing(
        search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        mode='sequence',
        crop_type=settings.crop_type,
        max_scale_change=settings.max_scale_change,
        transform=transform_val,
        joint_transform=transform_joint,
        label_function_params=label_params,
        new_roll=True)

    # Train sampler and loader
    dataset_train = sampler.LWLSampler(
        [ytvos_train, davis_train, got10k_train, lasot_train], [6, 1, 6, 6],
        samples_per_epoch=settings.batch_size * 1000, max_gap=100,
        num_test_frames=3,
        num_train_frames=1,
        processing=data_processing_train)
    dataset_val = sampler.LWLSampler(
        [ytvos_val], [1],
        samples_per_epoch=settings.batch_size * 100, max_gap=100,
        num_test_frames=3,
        num_train_frames=1,
        processing=data_processing_val)

    loader_train = LTRLoader(
        'train', dataset_train, training=True, num_workers=settings.num_workers,
        stack_dim=1, batch_size=settings.batch_size)

    loader_val = LTRLoader(
        'val', dataset_val, training=False, num_workers=settings.num_workers,
        epoch_interval=5, stack_dim=1, batch_size=settings.batch_size)

    # Network
    frozen_backbone_layers = ['conv1', 'bn1', 'layer1']
    # frozen_backbone_layers = 'all'
    net = rts_networks.steepest_descent_resnet50_with_clf_encoder(
        filter_size=3, num_filters=16, optim_iter=5,
        backbone_pretrained=True,
        out_feature_dim=512,
        frozen_backbone_layers=frozen_backbone_layers,
        label_encoder_dims=(16, 32, 64),
        use_bn_in_label_enc=False,
        clf_feat_blocks=0,
        final_conv=True,
        backbone_type='mrcnn',
        clf_filter_size=settings.clf_target_filter_sz,
        clf_score_act='relu',
        clf_hinge_threshold=settings.clf_hinge_threshold,
        clf_activation_leak=0.1,
        clf_with_extractor=True,
        clf_enc_input='sc')

    # Load pretrained
    net_pre, _ = load_pretrained(
        'lwl', 'lwl_ytvos', backbone_pretrained=True, frozen_backbone_layers=frozen_backbone_layers,
        checkpoint=os.path.join(settings.env.pretrained_networks, "lwl_stage2.pth"))

    net.target_model      = net_pre.target_model
    net.label_encoder     = net_pre.label_encoder

    # net.clf_encoder       = net_pre.clf_encoder
    # net.fusion_module     = net_pre.fusion_module
    # net.classifier        = net_pre.classifier

    net.feature_extractor = net_pre.feature_extractor
    net.decoder           = net_pre.decoder

    for p in net.parameters():
        p.requires_grad_(True)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    # Loss function
    objective = {
        'segm':  ltr_losses.LovaszSegLoss(per_image=False),
        'test_clf': ltr_losses.LBHinge(threshold=settings.clf_hinge_threshold),
    }

    loss_weight = {
        'segm': 10.0,
        'test_clf': 100.0,
        'test_init_clf': 100.0,
        'test_iter_clf': 400.0,
    }

    actor = segm_actors.RTSActor(net=net, objective=objective, loss_weight=loss_weight,
                                  num_refinement_iter=2, disable_all_bn=True)

    # Optimizer
    optimizer = optim.Adam([
        {'params': actor.net.feature_extractor.layer2.parameters(), 'lr': 4e-5},
        {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 4e-5},
        {'params': actor.net.feature_extractor.layer4.parameters(), 'lr': 4e-5},
        {'params': actor.net.target_model.filter_initializer.parameters(), 'lr': 8e-5},
        {'params': actor.net.target_model.filter_optimizer.parameters(), 'lr': 8e-5},
        {'params': actor.net.target_model.feature_extractor.parameters(), 'lr': 8e-5},
        {'params': actor.net.label_encoder.parameters(), 'lr': 8e-5},
        {'params': actor.net.decoder.parameters(), 'lr': 8e-5},

        {'params': actor.net.clf_encoder.parameters(), 'lr': 2e-4},
        {'params': actor.net.fusion_module.parameters(), 'lr': 2e-4},
        {'params': actor.net.classifier.filter_initializer.parameters(), 'lr': 2e-4},
        {'params': actor.net.classifier.filter_optimizer.parameters(), 'lr': 2e-4},
        {'params': actor.net.classifier.feature_extractor.parameters(), 'lr': 2e-4},
    ], lr=4e-5)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 115, 160], gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(200, load_latest=True, fail_safe=True)