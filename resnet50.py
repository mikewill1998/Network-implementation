import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=1,
                    bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(in_channels=mid_channels,
                    out_channels=mid_channels*self.expansion,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels*self.expansion)
        self.relu = nn.ReLU()
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, input_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(in_channels=input_channels,
                            out_channels=64,
                            kernel_size=(7, 7),
                            stride=2,
                            padding=3,
                            bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], mid_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], mid_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], mid_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], mid_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn(self.conv(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, blocks_num, mid_channels, stride):
        downsample = None
        layers = []
        if stride != 1 or self.in_channels != mid_channels*4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                            out_channels=mid_channels*4,
                            kernel_size=(1, 1),
                            stride=stride,
                            bias=False),
                nn.BatchNorm2d(mid_channels*4)
            )
        layers.append(block(in_channels=self.in_channels,
                            mid_channels=mid_channels,
                            downsample=downsample,
                            stride=stride))
        self.in_channels = mid_channels*4

        for i in range(blocks_num-1):
            layers.append(block(in_channels=self.in_channels, mid_channels=mid_channels))

        return nn.Sequential(*layers)


x = torch.rand(3, 3, 224, 224)
model = ResNet(ResidualBlock, [3, 4, 6, 3], 3, 10)
y = model(x)
y