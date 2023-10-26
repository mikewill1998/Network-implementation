import torch
import torch.nn as nn
import math

# [expand_ratio, channels, repeats, stride, k_size]
base_model = [
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

# tuple of {phi_value, resolution, drop_rate}
# (alpha->for depth(depth_factor=alpha**phi), beta->for width(width_factor=beta**phi), gamma->for resolution)
phi_values = {
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5)
}

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride, padding, groups=1):
        super().__init__()
        self.cnn = nn.Conv2d(in_c, out_c, k_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_c, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_c, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class InvertedResBlock(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
        super().__init__()
        self.survival_prob=0.8
        self.use_residual = (in_c == out_c and stride==1)
        reduced_dim = int(in_c/reduction)
        hidden_dim = in_c * expand_ratio
        self.expand = (in_c != hidden_dim)
        
        if self.expand:
            self.expand_conv = ConvBlock(in_c, hidden_dim, k_size=3, stride=1, padding=1)
        
        self.conv = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, k_size=k_size, stride=stride, padding=padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob)

        return torch.div(x, self.survival_prob) * binary_tensor
    
    def forward(self, inputs):
        if self.expand:
            x = self.expand_conv(inputs)
        else:
            x = inputs
        
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super().__init__()

        width_factor, depth_factor, drop_rate = self.calculate_factors(version)
        last_channels = math.ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(last_channels, num_classes)
        )
    
    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, resolution, drop_rate = phi_values[version]
        depth_factor = alpha**phi
        width_factor = beta**phi

        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(width_factor * 32)
        features = [ConvBlock(3, channels, 3, stride=2, padding=1)]
        in_c = channels

        for expand_ratio, channels, repeats, stride, k_size in base_model:
            out_c = 4 * math.ceil(int(channels*width_factor)/4)
            layers_repeats = math.ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(InvertedResBlock(
                    in_c,
                    out_c,
                    expand_ratio=expand_ratio,
                    stride=stride if layer == 0 else 1,
                    k_size=k_size,
                    padding=k_size//2
                ))
                in_c = out_c
        
        features.append(ConvBlock(in_c, last_channels, k_size=1, stride=1, padding=0))

        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))

device = "cuda" if torch.cuda.is_available() else "cpu"
version = "b0"
phi, resolution, drop_rate = phi_values[version]
model = EfficientNet(version=version, num_classes=10).to(device)

x = torch.randn(4, 3, resolution, resolution).to(device)

model(x)