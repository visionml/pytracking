import torch.nn as nn
import torch.optim as optim
import torchvision.transforms

from ltr.dataset import Lasot, TrackingNet, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
import ltr.models.bbreg.atom as atom_models
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as dltransforms


def run(settings):
    settings.description = 'Training ATOM.'
    settings.batch_size = 64
    settings.num_workers = 8
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.target_filter_sz = 4
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 0, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
    settings.output_sigma_factor = 1/4

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(11)))
    coco_train = MSCOCOSeq(settings.env.coco_dir)

    # Validation datasets
    trackingnet_val = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(11,12)))

    # Data transform
    transform_joint = dltransforms.ToGrayscale(probability=0.05)

    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    # The tracking pairs processing module
    proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}
    data_processing_train = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    data_processing_val = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    proposal_params=proposal_params,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    # Train sampler and loader
    dataset_train = sampler.ATOMSampler([lasot_train, trackingnet_train, coco_train], [1,1,1],
                                        samples_per_epoch=1000*settings.batch_size, max_gap=50,
                                        processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    dataset_val = sampler.ATOMSampler([trackingnet_val], [1], samples_per_epoch=500*settings.batch_size, max_gap=50,
                                      processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # Create network and actor
    net = atom_models.atom_resnet18(backbone_pretrained=True)
    objective = nn.MSELoss()
    actor = actors.AtomActor(net=net, objective=objective)

    # Optimizer
    optimizer = optim.Adam(actor.net.bb_regressor.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(40, load_latest=True, fail_safe=True)
