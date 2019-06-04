# Network Model
import torch.nn as nn
from util.padding import CircularPad2d

# def Padding(l, r, u, d):
#     '''
#     Custom padding, circular padding up/down, 
#     replication padding left/right
#     '''
#     self.ud_pad = CircularPad2d((0, 0, u, d))
#     self.lr_pad = nn.ReplicationPad2d((l, r, 0, 0))

#     def forward(self, x):
#         x = self.ud_pad(x)
#         x = self.lf_pad(x)


def ConvBlock(fan_in, fan_out, stride=1, bias=False):
    # 3x3 convolution with padding
    return nn.Conv2d(fan_in, fan_out, kernel_size=3, stride=stride, bias=bias)


class ResUnit(nn.Module):
    expansion = 1

    def __init__(self, fan_in, fan_out, stride=1, downsample=None):
        super(ResUnit, self).__init__()
        self.conv1 = ConvBlock(fan_in, fan_out, stride=stride)
        # self.bn = nn.BatchNorm2d(fan_out)
        self.gn1 = nn.GroupNorm(4, fan_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvBlock(fan_out, fan_out)
        self.gn2 = nn.GroupNorm(4, fan_out)
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
        # self.layer1 = self._make_layer1(64, 128)
        # self.layer2 = self._make_layer1(128, 256)
        # self.layer3 = self._make_layer1(256, 512)
        self.layer1 = self._make_layer2(ResUnit, 128, 1)
        self.layer2 = self._make_layer2(ResUnit, 256, 1)
        self.layer3 = self._make_layer2(ResUnit, 512, 1)
        self.out_conv = ConvBlock(512 * ResUnit.expansion, 4, bias=True)

    # def _make_layer1(self, fan_in, fan_out, **kwargs):
    #     '''
    #     Construct major layers with ResUnit
    #     '''
    #     layers = []
    #     layers.append(ResUnit(fan_in, fan_out, **kwargs))

    #     return nn.Sequential(*layers)

    def _make_layer2(self, block, fan_out, blocks, stride=1, **kwargs):
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


class ResNetxx(nn.Module):
    """docstring for ResNetxx"""
     # (W−F+2P)/S+1
    def __init__(self, block, layers, n_class = 4):
        super(ResNetxx, self).__init__()
        self.fan_in = 64

        self.ud_pad = CircularPad2d((0, 0, 3, 3))
        self.lr_pad = nn.ReplicationPad2d((3, 3, 0, 0))
        self.conv1 = nn.Conv2d(8, self.fan_in, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.fan_in)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512 * block.expansion, n_class)
        

    def _make_layer(self, block, fan_out, blocks, stride=1):
        '''
        Construct major layers ResNetxx like
        '''
        downsample = None
        if stride != 1 or self.fan_in != fan_out * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.fan_in, fan_out * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(fan_out * block.expansion),
                nn.GroupNorm(4, fan_out * block.expansion), 
            )

        layers = []
        layers.append(block(self.fan_in, fan_out, stride, downsample))
        self.fan_in = fan_out * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.fan_in, fan_out))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.ud_pad(x)
        x = self.lr_pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNet18(**kwargs):
    return ResNetxx(ResUnit, [2, 2, 2, 2], **kwargs)
