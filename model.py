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


def ConvBlock(fan_in, fan_out, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(fan_in, fan_out, kernel_size=3, stride=stride, bias=False)


class ResUnit(nn.Module):
    expansion = 1

    def __init__(self, fan_in, fan_out, stride=1, downsample=None):
        super(ResUnit, self).__init__()
        self.ud_pad = CircularPad2d((0, 0, 1, 1))
        self.lr_pad = nn.ReplicationPad2d((1, 1, 0, 0))

        self.conv1 = ConvBlock(fan_in, fan_out, stride=stride)
        self.bn1 = nn.BatchNorm2d(fan_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvBlock(fan_out, fan_out)
        self.bn2 = nn.BatchNorm2d(fan_out)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        x = self.ud_pad(x)
        x = self.lr_pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.ud_pad(x)
        x = self.lr_pad(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    """docstring for ResNet"""
     # (W−F+2P)/S+1
    def __init__(self, block, layers, n_class = 4):
        super(ResNet, self).__init__()
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
        Construct major layers ResNet like
        '''
        downsample = None
        if stride != 1 or self.fan_in != fan_out * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.fan_in, fan_out * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(fan_out * block.expansion),
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
    return ResNet(ResUnit, [2, 2, 2, 2], **kwargs)
