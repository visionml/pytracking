import os
import torch.optim as optim
from ltr.dataset import LasotCandidateMatching
from ltr.data import processing, sampler, LTRLoader
from ltr.models.target_candidate_matching import target_candidate_matching as tcm
import  ltr.models.loss.target_candidate_matching_loss  as tcm_loss
import ltr.actors.tracking as tcm_actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import ltr.admin.loading as network_loading


def run(settings):
    settings.description = 'KeepTrack: Trains the target candidate association network using the candidate matching dataset'
    settings.batch_size = 64
    settings.num_workers = 8
    settings.multi_gpu = False
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 6.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 22
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 5.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05

    # Train datasets
    lasot_dumped_train = LasotCandidateMatching(settings.env.lasot_dir,
                                                settings.env.lasot_candidate_matching_dataset_path, split='train-train')

    # Validation datasets
    lasot_dumped_val = LasotCandidateMatching(settings.env.lasot_dir,
                                              settings.env.lasot_candidate_matching_dataset_path, split='train-val')


    transform_train = tfm.Transform(tfm.ToTensor(),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    img_aug_transform_train = tfm.Transform(tfm.ToTensorAndJitter(normalize=True, brightness_jitter=0.5),
                                            tfm.RandomBlur(sigma=0.5, probability=0.75),
                                            tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    processing_train = processing.TargetCandiateMatchingProcessing(output_sz=settings.output_sz, train_transform=transform_train,
                                                                   num_target_candidates=5, img_aug_transform=img_aug_transform_train)

    processing_val = processing.TargetCandiateMatchingProcessing(output_sz=settings.output_sz, train_transform=transform_train,
                                                                 num_target_candidates=5, img_aug_transform=img_aug_transform_train)

    processing_val_real = processing.TargetCandiateMatchingProcessing(output_sz=settings.output_sz,
                                                                      train_transform=transform_train,
                                                                      real_target_candidates_only=True)

    # Train sampler and loader
    dataset_train = sampler.SequentialTargetCandidateMatchingSampler(lasot_dumped_train,
                                                                     samples_per_epoch=int(settings.batch_size*100),
                                                                     processing=processing_train, sup_modes=['self_sup','partial_sup'],
                                                                     frame_modes=['H', 'K', 'J'], p_frame_modes=[1., 0.5, 0.5],
                                                                     subseq_modes=['HH','HK','HG'], p_subseq_modes=[1.0, 0.1, 0.1])

    dataset_val = sampler.SequentialTargetCandidateMatchingSampler(lasot_dumped_val,
                                                                  samples_per_epoch=int(settings.batch_size * 10),
                                                                  processing=processing_val, sup_modes=['self_sup'],
                                                                  frame_modes=['G', 'H', 'J', 'K'],
                                                                  p_frame_modes=[0.2, 0.4, 0.2, 0.2])

    dataset_val_realHH = sampler.SequentialTargetCandidateMatchingSampler(lasot_dumped_val,
                                                                          samples_per_epoch=int(settings.batch_size * 5),
                                                                          processing=processing_val_real, sup_modes=['partial_sup'],
                                                                          subseq_modes=['HH'],
                                                                          p_subseq_modes=[1.])

    dataset_val_realHK = sampler.SequentialTargetCandidateMatchingSampler(lasot_dumped_val,
                                                                          samples_per_epoch=int(settings.batch_size * 2),
                                                                          processing=processing_val_real, sup_modes=['partial_sup'],
                                                                          subseq_modes=['HK'],
                                                                          p_subseq_modes=[1.])

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                             num_workers=settings.num_workers, shuffle=True, drop_last=True, stack_dim=1)

    loader_val = LTRLoader('val_self_sup', dataset_val, training=False, batch_size=settings.batch_size,
                           num_workers=settings.num_workers, shuffle=False, drop_last=True, stack_dim=1)

    loader_val_realHH = LTRLoader('val_HH', dataset_val_realHH, training=False, batch_size=1,
                                  num_workers=settings.num_workers, shuffle=False, drop_last=True, stack_dim=1)

    loader_val_realHK = LTRLoader('val_HK', dataset_val_realHK, training=False, batch_size=1,
                                  num_workers=settings.num_workers, shuffle=False, drop_last=True, stack_dim=1)

    loader = [loader_train, loader_val, loader_val_realHH, loader_val_realHK]

    net = tcm.target_candidate_matching_net_resnet50(backbone_pretrained=True,
                                                     frozen_backbone_layers=['conv1', 'bn1', 'layer1', 'layer2'])

    dimp_weights_path = os.path.join(settings.env.pretrained_networks, 'super_dimp_simple.pth.tar')
    base_net, _ = network_loading.load_network(checkpoint=dimp_weights_path, backbone_pretrained=False)

    net.load_state_dict(base_net.state_dict(), strict=False)

    for p in net.parameters():
        p.requires_grad_(True)

    for p in net.feature_extractor.parameters():
        p.requires_grad_(False)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {'target_candidate_matching': tcm_loss.TargetCandidateMatchingLoss()}

    actor = tcm_actors.TargetCandiateMatchingActor(net=net, objective=objective)

    # Optimizer
    optimizer = optim.Adam([{'params': actor.net.parameters(), 'lr': 1.0e-4}], lr=1.0e-4)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)

    trainer = LTRTrainer(actor, loader, optimizer, settings, lr_scheduler)

    trainer.train(15, load_latest=True, fail_safe=True)
