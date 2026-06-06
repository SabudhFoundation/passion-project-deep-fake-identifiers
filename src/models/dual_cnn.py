import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.gn1 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.gn2 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )

        self.act = nn.GELU()

        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.GroupNorm(
                    num_groups=8,
                    num_channels=out_channels
                )
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.act(out)

        return out


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(
                channels,
                channels // reduction
            ),
            nn.GELU(),
            nn.Linear(
                channels // reduction,
                channels
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(
            b,
            c,
            1,
            1
        )

        return x * y


class StreamNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            ResidualBlock(3, 16),
            SEBlock(16),
            nn.MaxPool2d(2),

            ResidualBlock(16, 32),
            SEBlock(32),
            nn.MaxPool2d(2),

            ResidualBlock(32, 64),
            SEBlock(64),
            nn.MaxPool2d(2),

            ResidualBlock(64, 128),
            SEBlock(128),

            nn.AdaptiveAvgPool2d((2, 2))
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        return x


class ImprovedDualStreamCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.spatial_net = StreamNet()

        self.freq_net = StreamNet()

        self.feature_dim = 128 * 2 * 2

        self.total_dim = self.feature_dim * 2

        self.fusion_attention = nn.Sequential(

            nn.Linear(
                self.total_dim,
                self.total_dim
            ),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(
                self.total_dim,
                self.total_dim
            ),

            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                self.total_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                64
            ),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, rgb, fft):

        spatial_feat = self.spatial_net(rgb)

        freq_feat = self.freq_net(fft)

        combined = torch.cat(
            (spatial_feat, freq_feat),
            dim=1
        )

        attention_weights = self.fusion_attention(
            combined
        )

        combined = combined * attention_weights

        output = self.classifier(combined)

        return output