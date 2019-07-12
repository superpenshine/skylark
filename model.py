# Network Model
import torch
import torch.nn as nn


def crop_position(res_shp, main_shp):
    '''
    Given res_shp and main_shp, return tl and br
    tl, br: proper upper left & bottom right pixel position to crop
    '''
    tl = (int(0.5 * (res_shp[2] - main_shp[2])), int(0.5 * (res_shp[3] - main_shp[3])))
    br = (tl[0] + main_shp[2], tl[1] + main_shp[3])

    return tl, br

def ConvBlock(fan_in, fan_out, stride=1, bias=False):
    '''
    3x3 convolution with padding
    '''
    # return nn.Conv2d(fan_in, fan_out, kernel_size=3, stride=stride, bias=bias)
    return nn.Conv2d(fan_in, fan_out, kernel_size=(1, 3), stride=stride, bias=bias)


class ResUnit(nn.Module):

    def __init__(self, fan_in, fan_out, stride=1, downsample=None):
        super(ResUnit, self).__init__()
        self.conv1 = ConvBlock(fan_in, fan_out, stride=stride)
        # self.bn = nn.BatchNorm2d(fan_out)
        self.gn1 = nn.GroupNorm(32, fan_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = ConvBlock(fan_out, fan_out)
        self.gn2 = nn.GroupNorm(32, fan_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        tl = (int(0.5 * (residual.size()[2] - out.size()[2])), int(0.5 * (residual.size()[3] - out.size()[3])))
        br = (tl[0] + out.size()[2], tl[1] + out.size()[3])
        residual = residual[:,:,tl[0]:br[0],tl[1]:br[1]]
        out = out + residual

        return out


class ResNet(nn.Module):
    '''
    Simple ResNet:
    '''
    def __init__(self):
        super(ResNet, self).__init__()
        self.fan_in = 512
        # self.conv1 = ConvBlock(2, 64, bias=True)
        self.conv1 = ConvBlock(2*4*32, 512, bias=True)
        self.layer1 = self._make_layer(ResUnit, 512, 1)
        self.layer2 = self._make_layer(ResUnit, 512, 1)
        self.layer3 = self._make_layer(ResUnit, 512, 1)
        self.layer4 = self._make_layer(ResUnit, 512, 1)
        self.out_conv = ConvBlock(512, 1*4*32, bias=True)


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


class DoubleConv(nn.Module):
    '''
    Construct 2 conv blocks and 1 downsample
    '''
    def __init__(self, fan_in, fan_out):
        super(DoubleConv, self).__init__()
        self.conv1 = ConvBlock(fan_in, fan_out)
        self.conv2 = ConvBlock(fan_out, fan_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x


def UpConv(fan_in, fan_out, stride=2):
    return nn.ConvTranspose2d(fan_in, fan_out, 2, stride=stride)


def MaxPool(stride=2):
    return nn.MaxPool2d(2, stride=stride)


# class UpBlock(nn.Module):
#     '''
#     Construct 2 conv blocks and 1 downsample
#     '''
#     def __init__(self, fan_in, fan_out):
#         super(UpBlock, self).__init__()
#         self.upconv = nn.ConvTranspose2d(fan_in, fan_out, 2, stride=2)
#         self.conv1 = ConvBlock(fan_in, fan_out)
#         self.conv2 = ConvBlock(fan_out, fan_out)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.upconv(x)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)

#         return x


class UNet2(nn.Module):
    '''
    UNet
    '''
    def __init__(self):
        super(UNet2, self).__init__()
        self.fan_in = 8
        self.d_layer1 = self._make_layer(DoubleConv, 256)
        self.d_layer2 = self._make_layer(DoubleConv, 512)
        self.pool1 = MaxPool()
        self.pool2 = MaxPool()

        self.bot_layer = self._make_layer(DoubleConv, 1024)

        self.upconv1 = self._make_layer(UpConv, 512)
        self.u_layer1 = self._make_layer(DoubleConv, 512, cat=512)
        self.upconv2 = self._make_layer(UpConv, 256)
        self.u_layer2 = self._make_layer(DoubleConv, 256, cat=256)

        self.out = nn.Conv2d(256, 4, kernel_size=1, stride=1, bias=True)


    def _make_layer(self, block, fan_out, cat=0):
        '''
        Construct layers
        cat: Concatenate residual flow
        '''
        if cat != 0:
            self.fan_in += cat
        layer = block(self.fan_in, fan_out)
        self.fan_in = fan_out

        return layer


    def forward(self, x):
        x = self.d_layer1(x)
        residual1 = x

        x = self.pool1(x)
        x = self.d_layer2(x) 
        residual2 = x

        x = self.pool2(x)

        x = self.bot_layer(x)

        x = self.upconv1(x)
        tl, br = crop_position(residual2.size(), x.size())
        residual2 = residual2[:,:,tl[0]:br[0],tl[1]:br[1]]
        x = torch.cat((x, residual2[:,:,:x.size()[2],:x.size()[3]]), 1)
        x = self.u_layer1(x)

        x = self.upconv2(x)
        tl, br = crop_position(residual1.size(), x.size())
        residual1 = residual1[:,:,tl[0]:br[0],tl[1]:br[1]]
        x = torch.cat((x, residual1[:,:,:x.size()[2],:x.size()[3]]), 1)
        x = self.u_layer2(x)

        x = self.out(x)
        return x


class UNet(nn.Module):
    '''
    UNet
    '''
    def __init__(self):
        super(UNet, self).__init__()
        self.fan_in = 8
        self.d_layer1 = self._make_layer(DoubleConv, 64)
        self.d_layer2 = self._make_layer(DoubleConv, 128)
        self.d_layer3 = self._make_layer(DoubleConv, 256)
        self.d_layer4 = self._make_layer(DoubleConv, 512)
        self.pool = MaxPool()

        self.bot_layer = self._make_layer(DoubleConv, 1024)

        self.upconv1 = self._make_layer(UpConv, 512)
        self.u_layer1 = self._make_layer(DoubleConv, 512, cat=512)
        self.upconv2 = self._make_layer(UpConv, 256)
        self.u_layer2 = self._make_layer(DoubleConv, 256, cat=256)
        self.upconv3 = self._make_layer(UpConv, 128)
        self.u_layer3 = self._make_layer(DoubleConv, 128, cat=128)
        self.upconv4 = self._make_layer(UpConv, 64)
        self.u_layer4 = self._make_layer(DoubleConv, 64, cat=64)

        self.out = nn.Conv2d(64, 4, kernel_size=1, stride=1, bias=True)


    def _make_layer(self, block, fan_out, cat=0):
        '''
        Construct layers
        cat: Concatenate residual flow
        '''
        if cat != 0:
            self.fan_in += cat
        layer = block(self.fan_in, fan_out)
        self.fan_in = fan_out

        return layer


    def forward(self, x):
        x = self.d_layer1(x)
        residual1 = x

        x = self.pool(x)
        x = self.d_layer2(x) 
        residual2 = x

        x = self.pool(x)
        x = self.d_layer3(x)
        residual3 = x

        x = self.pool(x)
        x = self.d_layer4(x)
        residual4 = x

        x = self.pool(x)

        x = self.bot_layer(x)

        x = self.upconv1(x)
        tl, br = crop_position(residual4.size(), x.size())
        residual4 = residual4[:,:,tl[0]:br[0],tl[1]:br[1]]
        x = torch.cat((x, residual4[:,:,:x.size()[2],:x.size()[3]]), 1)
        x = self.u_layer1(x)

        x = self.upconv2(x)
        tl, br = crop_position(residual3.size(), x.size())
        residual3 = residual3[:,:,tl[0]:br[0],tl[1]:br[1]]
        x = torch.cat((x, residual3[:,:,:x.size()[2],:x.size()[3]]), 1)
        x = self.u_layer2(x)

        x = self.upconv3(x)
        tl, br = crop_position(residual2.size(), x.size())
        residual2 = residual2[:,:,tl[0]:br[0],tl[1]:br[1]]
        x = torch.cat((x, residual2[:,:,:x.size()[2],:x.size()[3]]), 1)
        x = self.u_layer3(x)

        x = self.upconv4(x)
        tl, br = crop_position(residual1.size(), x.size())
        residual1 = residual1[:,:,tl[0]:br[0],tl[1]:br[1]]
        x = torch.cat((x, residual1[:,:,:x.size()[2],:x.size()[3]]), 1)
        x = self.u_layer4(x)

        x = self.out(x)
        return x



