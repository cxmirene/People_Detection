import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, multiple, stride):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels*multiple,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn1 = nn.BatchNorm2d(in_channels*multiple)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels*multiple,
            out_channels=in_channels*multiple,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels*multiple,
        )
        self.bn2 = nn.BatchNorm2d(in_channels*multiple)
        self.conv3 = nn.Conv2d(
            in_channels=in_channels*multiple,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if  in_channels!=out_channels and stride==1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride==1:
            out = out + self.shortcut(x)
        return out


class MobileNetv2(nn.Module):
    def __init__(self,num_classes=1000):
        self.parameters = [
            (1,16,1,1),
            (6,24,2,2),
            (6,32,3,2),
            (6,64,4,2),
            (6,96,3,1),
            (6,160,3,2),
            (6,320,1,1)]
        super(MobileNetv2,self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.bottleneck = self.Make_Bottleneck()
        self.conv2 = nn.Conv2d(
            in_channels=320,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280,num_classes)

    def Make_Bottleneck(self):
        layers = []
        in_channels = 32
        for parameter in self.parameters:
            strides = [parameter[3]] + [1]*(parameter[2]-1)
            for stride in strides:
                layer = Bottleneck(in_channels, parameter[1], parameter[0], stride)
                layers.append(layer)
                in_channels = parameter[1]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = self.bottleneck(out)
        out = F.relu6(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
 
def test():
    net = MobileNetv2()
    print(net)
    x = torch.randn(2,3,32,32)    #个数、通道数、宽、高
    print(x)
    y = net(x)
    print(y.size())

# test()
