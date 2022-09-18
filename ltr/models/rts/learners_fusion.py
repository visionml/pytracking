import torch
import torch.nn as nn
import math


class LearnersFusion(nn.Module):
    """  """
    def __init__(self, fusion_type):
        super().__init__()
        self.fusion_type = fusion_type

        if self.fusion_type == 'concat':
            self.fusion_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, seg_learner_out, clf_learner_out):

        assert seg_learner_out.shape == clf_learner_out.shape
        assert seg_learner_out.shape[0] == 1

        if self.fusion_type == 'add':
            return seg_learner_out + clf_learner_out

        if self.fusion_type == 'concat':
            concat_output = torch.cat([seg_learner_out, clf_learner_out], dim=2)
            concat_output = concat_output.squeeze(0)
            concat_output = self.fusion_conv1(concat_output)
            concat_output = concat_output.unsqueeze(0)

            return concat_output

        print("Type of fusion not recognized")
        assert False



