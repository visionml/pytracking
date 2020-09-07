import torch.optim as optim
import os
from ltr.dataset import Lasot, Got10k, TrackingNet
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.kysnet as kysnet_models
import ltr.models.loss as ltr_losses
from ltr import actors
from ltr.trainers import LTRTrainer
from ltr.models.kys.utils import DiMPScoreJittering
import ltr.data.transforms as tfm
import ltr.admin.loading as network_loading


def run(settings):
    settings.move_data_to_gpu = False
    settings.description = ''
    settings.batch_size = 10
    settings.test_sequence_length = 50
    settings.num_workers = 8
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_param = {'train_mode': 'uniform', 'train_factor': 3.0, 'train_limit_motion': False,
                                    'test_mode': 'uniform', 'test_factor': 4.5, 'test_limit_motion': True}
    settings.scale_jitter_param = {'train_factor': 0.25, 'test_factor': 0.3}
    settings.hinge_threshold = 0.05
    settings.print_stats = ['Loss/total', 'Loss/raw/test_clf', 'Loss/raw/test_clf_acc', 'Loss/raw/dimp_clf_acc',
                            'Loss/raw/is_target', 'Loss/raw/is_target_after_prop',
                            'Loss/raw/test_seq_acc',
                            'Loss/raw/dimp_seq_acc']

    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=[0, 1, 2, 3, 4])

    # Validation datasets
    got10k_val = Got10k(settings.env.got10k_dir, split='votval')

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = None

    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma,
                    'kernel_sz': settings.target_filter_sz,
                    'end_pad_if_even': True}

    data_processing_train = processing.KYSProcessing(search_area_factor=settings.search_area_factor,
                                                     output_sz=settings.output_sz,
                                                     center_jitter_param=settings.center_jitter_param,
                                                     scale_jitter_param=settings.scale_jitter_param,
                                                     proposal_params=proposal_params,
                                                     label_function_params=label_params,
                                                     transform=transform_train,
                                                     joint_transform=transform_joint,
                                                     min_crop_inside_ratio=0.1)

    data_processing_val = processing.KYSProcessing(search_area_factor=settings.search_area_factor,
                                                   output_sz=settings.output_sz,
                                                   center_jitter_param=settings.center_jitter_param,
                                                   scale_jitter_param=settings.scale_jitter_param,
                                                   proposal_params=proposal_params,
                                                   label_function_params=label_params,
                                                   transform=transform_val,
                                                   joint_transform=transform_joint,
                                                   min_crop_inside_ratio=0.1)

    # Train sampler and loader
    sequence_sample_info = {'num_train_frames': 3, 'num_test_frames': settings.test_sequence_length,
                            'max_train_gap': 30, 'allow_missing_target': True, 'min_fraction_valid_frames': 0.5,
                            'mode': 'Sequence'}

    dataset_train = sampler.KYSSampler([got10k_train, trackingnet_train, lasot_train],
                                       [0.3, 0.3, 0.25],
                                       samples_per_epoch=settings.batch_size * 150,
                                       sequence_sample_info=sequence_sample_info,
                                       processing=data_processing_train,
                                       sample_occluded_sequences=True)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                             num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    dataset_val = sampler.KYSSampler([got10k_val], [1], samples_per_epoch=1000,
                                     sequence_sample_info=sequence_sample_info, processing=data_processing_val,
                                     sample_occluded_sequences=True)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size,
                           num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # load base dimp
    dimp_weights_path = os.path.join(settings.env.pretrained_networks, 'dimp50.pth')
    base_net, _ = network_loading.load_network(checkpoint=dimp_weights_path)

    net = kysnet_models.kysnet_res50(optim_iter=3, cv_kernel_size=3, cv_max_displacement=9,
                                     cv_stride=1, init_gauss_sigma=output_sigma * settings.feature_sz,
                                     train_feature_extractor=False, train_iounet=False, detach_length=0, state_dim=8,
                                     representation_predictor_dims=(16,), conf_measure='entropy',
                                     dimp_thresh=0.05)

    # Move pre-trained dimp weights
    net.backbone_feature_extractor.load_state_dict(base_net.feature_extractor.state_dict())
    net.dimp_classifier.load_state_dict(base_net.classifier.state_dict())
    net.bb_regressor.load_state_dict(base_net.bb_regressor.state_dict())

    # To be safe
    for p in net.backbone_feature_extractor.parameters():
        p.requires_grad_(False)
    for p in net.dimp_classifier.parameters():
        p.requires_grad_(False)
    for p in net.bb_regressor.parameters():
        p.requires_grad_(False)

    objective = {'test_clf': ltr_losses.LBHingev2(threshold=settings.hinge_threshold, return_per_sequence=False),
                 'dimp_clf': ltr_losses.LBHingev2(threshold=settings.hinge_threshold, return_per_sequence=False),
                 'is_target': ltr_losses.IsTargetCellLoss(return_per_sequence=False),
                 'clf_acc': ltr_losses.TrackingClassificationAccuracy(threshold=0.25)}

    loss_weight = {'test_clf': 1.0*500, 'test_clf_orig': 50, 'is_target': 0.1*500, 'is_target_after_prop': 0.1*500}

    dimp_jitter_fn = DiMPScoreJittering(distractor_ratio=0.1, p_distractor=0.3, max_distractor_enhance_factor=1.3,
                                        min_distractor_enhance_factor=0.8)
    actor = actors.KYSActor(net=net, objective=objective, loss_weight=loss_weight,
                            dimp_jitter_fn=dimp_jitter_fn)

    optimizer = optim.Adam([{'params': actor.net.predictor.parameters(), 'lr': 1e-2}],
                           lr=1e-2)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(40, load_latest=True, fail_safe=True)
