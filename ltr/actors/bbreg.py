from . import BaseActor


class AtomActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        iou_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        iou_pred = iou_pred.view(-1, iou_pred.shape[2])
        iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])

        # Compute loss
        loss = self.objective(iou_pred, iou_gt)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}

        return loss, stats


class AtomBBKLActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM with BBKL"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_density', and 'gt_density'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        bb_scores = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        bb_scores = bb_scores.view(-1, bb_scores.shape[2])
        proposal_density = data['proposal_density'].view(-1, data['proposal_density'].shape[2])
        gt_density = data['gt_density'].view(-1, data['gt_density'].shape[2])

        # Compute loss
        loss = self.objective(bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': loss.item()}

        return loss, stats
