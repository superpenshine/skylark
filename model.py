# Network Model
import torch
import torch.nn as nn


def ConvBlock(fan_in, fan_out, stride=1, bias=False):
    '''
    3x3 convolution with padding
    '''
    return nn.Conv2d(fan_in, fan_out, kernel_size=3, stride=stride, bias=bias)


class ResUnit(nn.Module):

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
        tl = (int(0.5 * (residual.size()[2] - out.size()[2])), int(0.5 * (residual.size()[3] - out.size()[3])))
        br = (tl[0] + out.size()[2], tl[1] + out.size()[3])
        # import pdb
        # pdb.set_trace()
        residual = residual[:,:,tl[0]:br[0],tl[1]:br[1]]
        # residual = residual[:,:,2:3+out.size()[2],2:2+out.size()[3]]
        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''
    Simple ResNet:
    '''
    def __init__(self):
        super(ResNet, self).__init__()
        self.fan_in = 64
        self.conv1 = ConvBlock(8, 64, bias=True)
        self.layer1 = self._make_layer(ResUnit, 128, 1)
        self.layer2 = self._make_layer(ResUnit, 256, 1)
        self.layer3 = self._make_layer(ResUnit, 512, 1)
        self.out_conv = ConvBlock(512, 4, bias=True)


    def _make_layer(self, block, fan_out, blocks, stride=1, **kwargs):
        '''
        Construct a Res Unit
        '''
        downsample = None
        if stride != 1 or self.fan_in != fan_out:
            downsample = nn.Sequential(
                nn.Conv2d(self.fan_in, fan_out, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.fan_in, fan_out, stride, downsample, **kwargs))
        self.fan_in = fan_out

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


class DownBlock(nn.Module):
    '''
    Construct 2 conv blocks and 1 downsample
    '''
    def __init__(self, fan_in, fan_out):
        super(ResUnit, self).__init__()
        self.conv1 = ConvBlock(fan_in, fan_in)
        self.conv2 = ConvBlock(fan_in, fan_out)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        return x


class UpBlock(nn.Module):
    '''
    Construct 2 conv blocks and 1 downsample
    '''
    def __init__(self, fan_in, fan_out):
        super(ResUnit, self).__init__()
        self.upconv = nn.ConvTranspose2d(fan_in, fan_in, 2, stride=2)
        self.conv1 = ConvBlock(fan_in, fan_out)
        self.conv2 = ConvBlock(fan_out, fan_out)

    def forward(self, x):
        x = self.upconv(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class UNet(nn.Module):
    '''
    UNet
    '''
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = ConvBlock(8, 64)
        self.d_layer1 = _make_layers(DownBlock, 128)
        self.d_layer2 = _make_layers(DownBlock, 256)
        self.d_layer3 = _make_layers(DownBlock, 512)
        self.d_layer4 = _make_layers(DownBlock, 1024)
        self.bot1 = ConvBlock(1024, 1024)
        self.bot2 = ConvBlock(1024, 1024)
        self.u_layer1 = _make_layers(UpBlock, 512)
        self.u_layer2 = _make_layers(UpBlock, 256)
        self.u_layer3 = _make_layers(UpBlock, 128)
        self.u_layer4 = _make_layers(UpBlock, 64)
        self.out = ConvBlock(64, 4)


    def _make_layers(self, block, fan_out):
        '''
        Construct layers
        '''
        layers = []
        layers.append(block(self.fan_in, fan_out, stride, downsample, **kwargs))
        self.fan_in = fan_out

        return nn.Sequential(*layers)


    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_d1 = self.d_layer1(x_conv1)
        x_d2 = self.d_layer2(x_d1)
        x_d3 = self.d_layer3(x)
        x = self.d_layer4(x)
        residule1 = x_conv1
        residule2 = x_d1
        residule3 = x_d2
        residule4 = x_d3

        x = self.bot1(x)
        x = self.bot2(x)

        x = torch.cat((x, residule1[:,:,:x.size()[2],:x.size()[3]]), 1)
        x = self.u_layer1(x)
        x = torch.cat((x, residule2[:,:,:x.size()[2],:x.size()[3]]), 1)
        x = self.u_layer2(x)
        x = torch.cat((x, residule3[:,:,:x.size()[2],:x.size()[3]]), 1)
        x = self.u_layer3(x)
        x = torch.cat((x, residule4[:,:,:x.size()[2],:x.size()[3]]), 1)
        x = self.u_layer4(x)

        x = self.out(x)

        return x
