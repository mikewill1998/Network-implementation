import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5),stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sig = nn.Sigmoid()

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(self.sig(self.conv1(x)))
        x = self.pool(self.sig(self.conv2(x)))
        x = self.sig(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc2(self.sig(self.fc1(x)))

        return x
    

x = torch.rand(1, 1, 32, 32)
model = LeNet()
model(x)