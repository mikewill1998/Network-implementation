import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv3d(in_channels[0], out_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv3d(in_channels[1], out_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.ag = AttentionGate(in_channels, out_channels)
        self.c1 = ConvBlock(in_channels[0]+out_channels, out_channels)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], 1)
        x = self.c1(x)
        return x

class AttentionUNet3D(nn.Module):
    def __init__(self, image_channels, num_classes):
        super().__init__()

        self.e1 = EncoderBlock(image_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)

        self.b1 = ConvBlock(256, 512)

        self.d1 = DecoderBlock([512, 256], 256)
        self.d2 = DecoderBlock([256, 128], 128)
        self.d3 = DecoderBlock([128, 64], 64)

        self.output = nn.Conv3d(64, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)
        return output