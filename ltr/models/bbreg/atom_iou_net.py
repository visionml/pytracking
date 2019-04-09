import torch.nn as nn
import torch
from ltr.models.layers.blocks import LinearBlock
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


def valid_roi(roi: torch.Tensor, image_size: torch.Tensor):
    valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0]-1) and \
            all(roi[:, 4] <= image_size[1]-1)
    return valid


class AtomIoUNet(nn.Module):
    """ Network module for IoU prediction. Refer to the paper for an illustration of the architecture."""
    def __init__(self, input_dim=(128,256), pred_input_dim=(256,256), pred_inter_dim=(256,256)):
        super().__init__()
        # _r for reference, _t for test
        self.conv3_1r = conv(input_dim[0], 128, kernel_size=3, stride=1)
        self.conv3_1t = conv(input_dim[0], 256, kernel_size=3, stride=1)

        self.conv3_2t = conv(256, pred_input_dim[0], kernel_size=3, stride=1)

        self.prroi_pool3r = PrRoIPool2D(3, 3, 1/8)
        self.prroi_pool3t = PrRoIPool2D(5, 5, 1/8)

        self.fc3_1r = conv(128, 256, kernel_size=3, stride=1, padding=0)

        self.conv4_1r = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_1t = conv(input_dim[1], 256, kernel_size=3, stride=1)

        self.conv4_2t = conv(256, pred_input_dim[1], kernel_size=3, stride=1)

        self.prroi_pool4r = PrRoIPool2D(1, 1, 1/16)
        self.prroi_pool4t = PrRoIPool2D(3, 3, 1 / 16)

        self.fc34_3r = conv(256 + 256, pred_input_dim[0], kernel_size=1, stride=1, padding=0)
        self.fc34_4r = conv(256 + 256, pred_input_dim[1], kernel_size=1, stride=1, padding=0)

        self.fc3_rt = LinearBlock(pred_input_dim[0], pred_inter_dim[0], 5)
        self.fc4_rt = LinearBlock(pred_input_dim[1], pred_inter_dim[1], 3)

        self.iou_predictor = nn.Linear(pred_inter_dim[0]+pred_inter_dim[1], 1, bias=True)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat1, feat2, bb1, proposals2):
        assert feat1[0].dim() == 5, 'Expect 5  dimensional feat1'

        num_test_images = feat2[0].shape[0]
        batch_size = feat2[0].shape[1]

        # Extract first train sample
        feat1 = [f[0,...] for f in feat1]
        bb1 = bb1[0,...]

        # Get modulation vector
        filter = self.get_filter(feat1, bb1)

        feat2 = [f.view(batch_size * num_test_images, f.shape[2], f.shape[3], f.shape[4]) for f in feat2]
        iou_feat = self.get_iou_feat(feat2)

        filter = [f.view(1, batch_size, -1).repeat(num_test_images, 1, 1).view(batch_size*num_test_images, -1) for f in filter]

        proposals2 = proposals2.view(batch_size*num_test_images, -1, 4)
        pred_iou = self.predict_iou(filter, iou_feat, proposals2)
        return pred_iou.view(num_test_images, batch_size, -1)

    def predict_iou(self, filter, feat2, proposals):
        fc34_3_r, fc34_4_r = filter
        c3_t, c4_t = feat2

        batch_size = c3_t.size()[0]

        # Modulation
        c3_t_att = c3_t * fc34_3_r.view(batch_size, -1, 1, 1)
        c4_t_att = c4_t * fc34_4_r.view(batch_size, -1, 1, 1)

        # Add batch_index to rois
        batch_index = torch.Tensor([x for x in range(batch_size)]).view(batch_size, 1).to(c3_t.device)

        # Push the different rois for the same image along the batch dimension
        num_proposals_per_batch = proposals.shape[1]

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)

        # Add batch index
        roi2 = torch.cat((batch_index.view(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1),
                          proposals_xyxy), dim=2)
        roi2 = roi2.view(-1, 5).to(proposals_xyxy.device)

        roi3t = self.prroi_pool3t(c3_t_att, roi2)
        roi4t = self.prroi_pool4t(c4_t_att, roi2)

        fc3_rt = self.fc3_rt(roi3t)
        fc4_rt = self.fc4_rt(roi4t)

        fc34_rt_cat = torch.cat((fc3_rt, fc4_rt), dim=1)

        iou_pred = self.iou_predictor(fc34_rt_cat).view(batch_size, num_proposals_per_batch)

        return iou_pred

    def get_filter(self, feat1, bb1):
        feat3_r, feat4_r = feat1

        c3_r = self.conv3_1r(feat3_r)

        # Add batch_index to rois
        batch_size = bb1.size()[0]
        batch_index = torch.Tensor([x for x in range(batch_size)]).view(batch_size, 1).to(bb1.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb1 = bb1.clone()
        bb1[:, 2:4] = bb1[:, 0:2] + bb1[:, 2:4]
        roi1 = torch.cat((batch_index, bb1), dim=1)

        roi3r = self.prroi_pool3r(c3_r, roi1)

        c4_r = self.conv4_1r(feat4_r)
        roi4r = self.prroi_pool4r(c4_r, roi1)

        fc3_r = self.fc3_1r(roi3r)

        # Concatenate from block 3 and 4
        fc34_r = torch.cat((fc3_r, roi4r), dim=1)

        fc34_3_r = self.fc34_3r(fc34_r)
        fc34_4_r = self.fc34_4r(fc34_r)

        return fc34_3_r, fc34_4_r

    def get_iou_feat(self, feat2):
        feat3_t, feat4_t = feat2
        c3_t = self.conv3_2t(self.conv3_1t(feat3_t))
        c4_t = self.conv4_2t(self.conv4_1t(feat4_t))

        return c3_t, c4_t
