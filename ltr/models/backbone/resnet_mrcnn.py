import math
import torch.nn as nn
import os
import torch
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
import ltr.admin.settings as ws_settings
from .base import Backbone


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride_1x1=1, stride_3x3=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride_1x1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride_3x3,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):
    """ ResNet network module. Allows extracting specific feature blocks."""
    def __init__(self, block, layers, output_layers, num_classes=1000, inplanes=64, dilation_factor=1, frozen_layers=()):
        self.inplanes = inplanes
        super(ResNet, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Change! stride condition
        stride = [int(l > 1) + 1 for l in range(1, 4)]
        # stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]
        self.layer1 = self._make_layer(block, inplanes, layers[0], stride=stride[0], dilation=max(dilation_factor//8, 1))
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=stride[1], dilation=max(dilation_factor//4, 1))
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=stride[2], dilation=max(dilation_factor//2, 1))
        self.layer4 = self._make_layer(block, inplanes*8, layers[3], stride=stride[2], dilation=dilation_factor)

        out_feature_strides = {'conv1': 4, 'layer1': 4, 'layer2': 4 * stride[0], 'layer3': 4 * stride[0] * stride[1],
                               'layer4': 4 * stride[0] * stride[1] * stride[2]}

        if isinstance(self.layer1[0], Bottleneck):
            base_num_channels = 4 * inplanes
            out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2,
                                    'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        else:
            raise Exception('block not supported')

        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        if layer is None:
            return self._out_feature_strides
        else:
            return self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        if layer is None:
            return self._out_feature_channels
        else:
            return self._out_feature_channels[layer]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, stride_in_1x1=True):
        downsample = None
        # Change! No condition for stride 1
        if self.inplanes != planes * block.expansion:
            down_stride = stride if dilation == 1 else 1
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=down_stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # Change! reset stride
        if dilation > 1:
            stride = 1

        # Change! Stride
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride_1x1, stride_3x3, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        x = self.layer4(x)

        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # if self._add_output_and_check('fc', x, outputs, output_layers):
        #     return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


def resnet50(output_layers=None, pretrained=False, weights_path=None, **kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck, [3, 4, 6, 3], output_layers, **kwargs)

    if pretrained:
        print('Pre-trained weights not available. Load it manually')

    return model


def resnet101(output_layers=None, pretrained=False, weights_path=None, **kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck, [3, 4, 23, 3], output_layers, **kwargs)

    if pretrained:
        print('Pre-trained weights not available. Load it manually')

    return model
