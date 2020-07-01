import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3., self.inplace) / 6.
        return out 


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., inplace=self.inplace) / 6.
        return out

class SqueezeBlock(nn.Module):
    def __init__(self, input_channel):
        super(SqueezeBlock, self).__init__()
        divide = 4
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel, input_channel // divide, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channel // divide),
            nn.ReLU6(inplace=True),
            nn.Conv2d(input_channel // divide, input_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channel),
            h_sigmoid()
        )

    def forward(self, x):
        out = x * self.dense(x)
        return out

class Bottleneck(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, input_channel, exp_size, output_channel, SE, NL, stride):
        super(Bottleneck, self).__init__()
        self.stride = stride
        if input_channel==output_channel:
            self.stride=1
        self.squeezeblock = nn.Sequential()
        if SE:
            self.squeezeblock = SqueezeBlock(output_channel)
        if NL == "RE":
            activation = nn.ReLU6
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(),
            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=self.stride, padding=kernel_size//2, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(),
            nn.Conv2d(exp_size, output_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channel)
        )

        self.shortcut = nn.Sequential()
        if stride == 1 and input_channel != output_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(output_channel),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.squeezeblock != None:
            out = self.squeezeblock(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV3(nn.Module):

    parameters_large = [
        [3, 16, 16, 16, False, "RE", 1],
        [3, 16, 64, 24, False, "RE", 2],
        [3, 24, 72, 24, False, "RE", 1],
        [5, 24, 72, 40, True, "RE", 2],
        [5, 40, 120,40, True, "RE", 1],
        [5, 40, 120, 40, True, "RE", 1],
        [3, 40, 240, 80, False, "HS", 2],
        [3, 80, 200, 80, False, "HS", 1],
        [3, 80, 184, 80, False, "HS", 1],
        [3, 80, 184, 80, False, "HS", 1],
        [3, 80, 480, 112, True, "HS", 1],
        [3, 112, 672, 112, True, "HS", 1],
        [5, 112, 672, 160, True, "HS", 1],
        [5, 160, 672, 160, True, "HS", 2],
        [5, 160, 960, 160, True, "HS", 1],
    ]

    parameters_small = [
        [3, 16, 16, 16, True, "RE", 2],
        [3, 16, 72, 24, False, "RE", 2],
        [3, 24, 88, 24, False, "RE", 1],
        [5, 24, 96, 40, True, "HS", 2],
        [5, 40, 240,40, True, "HS", 1],
        [5, 40, 240, 40, True, "HS", 1],
        [5, 40, 120, 48, True, "HS", 1],
        [5, 48, 144, 48, True, "HS", 1],
        [5, 48, 288, 96, True, "HS", 2],
        [5, 96, 576, 96, True, "HS", 1],
        [5, 96, 576, 96, True, "HS", 1],
    ]

    def __init__(self, moudle, n_class=1000, dropout_ratio=0.2,):
        super(MobileNetV3, self).__init__()
        self.moudle = moudle
        self.features = []

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = h_swish()
        self.features.append(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False))
        self.features.append(nn.BatchNorm2d(16))
        self.features.append(h_swish())

        self.bottlenect = self.Make_Bottleneck()

        if self.moudle == "Large":
            final_input = 160
            final_output = 960
        else:
            final_input = 96
            final_output = 576

        self.conv2 = nn.Conv2d(final_input, final_output, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(final_output)
        self.hs2 = h_swish()
        self.conv3 = nn.Conv2d(final_output, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(1280)
        self.hs3 = h_swish()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(1280, n_class),
        )
        # self.linear = nn.Linear(1280, n_class)
        # if self.moudle == "Small":
        self.features.append(nn.Conv2d(final_input, final_output, kernel_size=1, stride=2, padding=0, bias=False))
        # else:
            # self.features.append(nn.Conv2d(final_input, final_output, kernel_size=1, stride=1, padding=0, bias=False))
        self.features.append(nn.BatchNorm2d(final_output))
        self.features.append(h_swish())
        self.features.append(nn.Conv2d(final_output, 1280, kernel_size=1, stride=1, padding=0, bias=False))
        self.features.append(nn.BatchNorm2d(1280))
        self.features.append(h_swish())
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def Make_Bottleneck(self):
        layers = []
        if self.moudle == "Large":
            parameters = self.parameters_large[:]
        elif self.moudle == "Small":
            parameters = self.parameters_small[:]
        for p in parameters:
            kernel_size = p[0]
            input = p[1]
            exp = p[2]
            output = p[3]
            SE = p[4]
            NL = p[5]
            stride = p[6]
            layer = Bottleneck(kernel_size, input, exp, output, SE, NL, stride)
            layers.append(layer)
            self.features.append(layer)
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bottlenect(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = self.hs3(self.bn3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def test():
    net = MobileNetV3("Small").features
    print(net)
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())

# test()
