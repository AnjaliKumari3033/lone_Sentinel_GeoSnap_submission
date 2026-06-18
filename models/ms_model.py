import torch
import pandas as pd
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),

            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, bias=False),
            nn.BatchNorm2d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()

        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class SpectralSE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(c, c // r),
            nn.ReLU(True),

            nn.Linear(c // r, c),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x).view(
            x.size(0),
            x.size(1),
            1,
            1
        )


class MultiSpectralCNN(nn.Module):
    def __init__(self, in_ch=16, nc=10):
        super().__init__()

        def cbr(i, o, s=1):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(True)
            )

        self.stem = nn.Sequential(
            cbr(in_ch, 64),
            cbr(64, 64)
        )

        self.se = SpectralSE(64)

        self.stage1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )

        self.stage2 = nn.Sequential(
            ResBlock(64, 128, 2),
            ResBlock(128, 128)
        )

        self.stage3 = nn.Sequential(
            ResBlock(128, 256, 2),
            ResBlock(256, 256)
        )

        self.stage4 = nn.Sequential(
            ResBlock(256, 512, 2),
            ResBlock(512, 512)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, nc)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity='relu'
                )

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):

        x = self.stem(x)
        x = self.se(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.head(x)

        return x