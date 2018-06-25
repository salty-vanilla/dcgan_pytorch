import torch.nn.functional as F
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim,
                 filters=64):
        super().__init__()
        self.input_dim = input_dim
        self.filters = filters

        self.fc_block = nn.Sequential(
            nn.Linear(input_dim, 4*4*4*filters),
            nn.BatchNorm1d(4*4*4*filters),
            nn.ReLU(inplace=True)
        )

        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(4*filters, 2*filters,
                               kernel_size=2,
                               stride=2),
            nn.BatchNorm2d(2*filters),
            nn.ReLU(True),
        )

        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose2d(2*filters, filters,
                               kernel_size=2,
                               stride=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
        )

        self.last_conv = nn.ConvTranspose2d(filters, 1,
                                            kernel_size=2,
                                            stride=2)

    def forward(self, x):
        x = self.fc_block(x)
        x = x.view(-1, 4*self.filters, 4, 4)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.last_conv(x)
        x = F.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, filters=64):
        super().__init__()
        self.filters = filters

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, filters,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(filters, 2*filters,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(2*filters),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(2*filters, 4*filters,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(4*filters),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(4*4*4*filters, 1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(-1, 4*self.filters*4*4)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x
