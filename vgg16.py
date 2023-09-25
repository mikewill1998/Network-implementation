import torch
import torch.nn as nn

VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

class VGGNets(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGGNets, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, structure):
        layers = []
        in_channels = self.in_channels

        for x in structure:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]
            else:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1),
                    nn.ReLU()
                ]
                in_channels = x

        return nn.Sequential(*layers)

model = VGGNets(in_channels=3, num_classes=10)
x = torch.randn(3, 3, 224, 224)
model(x)