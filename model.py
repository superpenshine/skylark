# Network Model
import torch.nn as nn


def ConvBlock(fan_in, fan_out, stride=1, bias=False):
    # 3x3 convolution with padding
    return nn.Conv2d(fan_in, fan_out, kernel_size=3, stride=stride, bias=bias)


class ResUnit(nn.Module):
    expansion = 1

    def __init__(self, fan_in, fan_out, stride=1, downsample=None):
        super(ResUnit, self).__init__()
        self.conv1 = ConvBlock(fan_in, fan_out, stride=stride)
        # self.bn = nn.BatchNorm2d(fan_out)
        self.gn1 = nn.GroupNorm(1, fan_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvBlock(fan_out, fan_out)
        self.gn2 = nn.GroupNorm(1, fan_out)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        residual = residual[:,:,:out.size()[2],:out.size()[3]]

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''
    Simple ResNet:
    '''
    def __init__(self, n_class = 4):
        super(ResNet, self).__init__()
        self.fan_in = 64
        self.conv1 = ConvBlock(8, 64, bias=True)
        self.layer1 = self._make_layer(ResUnit, 128, 1)
        self.layer2 = self._make_layer(ResUnit, 256, 1)
        self.layer3 = self._make_layer(ResUnit, 512, 1)
        self.out_conv = ConvBlock(512 * ResUnit.expansion, 4, bias=True)

    def _make_layer(self, block, fan_out, blocks, stride=1, **kwargs):
        '''
        Construct major layers ResNetxx like
        '''
        downsample = None
        if stride != 1 or self.fan_in != fan_out * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.fan_in, fan_out * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(fan_out * block.expansion),
            )

        layers = []
        layers.append(block(self.fan_in, fan_out, stride, downsample, **kwargs))
        self.fan_in = fan_out * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.fan_in, fan_out))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.out_conv(x)

        return x