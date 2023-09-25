import torch
import torch.nn as nn


class BatchNormReLU(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        x = self.relu(self.bn(inputs))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        """Convolution layer"""
        self.b1 = BatchNormReLU(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=stride, padding=1)
        self.b2 = BatchNormReLU(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=(1, 1), stride=1, padding=0)

        """for skip connection"""
        self.s = nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=stride, padding=0)
    
    def forward(self, inputs):
        x = self.c2(self.b2(self.c1(self.b1(inputs))))
        s = self.s(inputs)
        skip = x + s
        return skip


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = ResidualBlock(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], 1)
        x = self.r(x)
        return x

class ResUNet2D(nn.Module):
    def __init__(self):
        super().__init__()

        """Encoder 1"""
        self.c11 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.br1 = BatchNormReLU(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)

        self.c13 = nn.Conv2d(3, 64, kernel_size=(1, 1), padding=0)

        """Encoder 2 and 3"""
        self.r2 = ResidualBlock(64, 128, stride=2)
        self.r3 = ResidualBlock(128, 256, stride=2)

        """Bottleneck(Bridge)"""
        self.r4 = ResidualBlock(256, 512, stride=2)

        """Decoder"""
        self.d1 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)

        """Output"""
        self.output = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0)
        self.sig = nn.Sigmoid()
    
    def forward(self, inputs):
        """Encoder 1"""
        x = self.c12(self.br1(self.c11(inputs)))
        s = self.c13(inputs)
        skip1 = x + s
        """Encoder 2 and 3"""
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)
        """Bottleneck"""
        b = self.r4(skip3)
        """Decoder"""
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)
        """Output"""
        output = self.sig(self.output(d3))

        return output