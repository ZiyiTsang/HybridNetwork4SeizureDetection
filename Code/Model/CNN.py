
import torch
from torch import nn

from torchvision.ops import sigmoid_focal_loss
import transformers
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC,BinaryAveragePrecision
from .CommonBlock import Classification_block
import lightning as L

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.encoder_name = "CNN"

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ConvNetBlock_large(nn.Module):
    def __init__(self):
        super(ConvNetBlock_large, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.mpv1 = nn.MaxPool2d(kernel_size=3, stride=(2, 1),padding=1)
        self.block1 = ResidualBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.mpv2 = nn.MaxPool2d(kernel_size=3, stride=(2, 1),padding=1)
        self.block2 = ResidualBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.mpv3 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.block3 = ResidualBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.mpv4 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.block4 = ResidualBlock(256, 512, kernel_size=3, stride=1, padding=1)
        self.mpv5 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.block5 = ResidualBlock(512, 768, kernel_size=3, stride=1, padding=1)
        self.mpv6 = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten()

        self.encoder_name = "CNN"

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mpv1(x)
        x = self.block1(x)
        x = self.mpv2(x)
        x = self.block2(x)
        x = self.mpv3(x)
        x = self.block3(x)
        x = self.mpv4(x)
        x = self.block4(x)
        x = self.mpv5(x)
        x = self.block5(x)
        x = self.mpv6(x)
        x = self.Flatten(x)
        return x

class ConvNetBlock_small(nn.Module):
    def __init__(self):
        super(ConvNetBlock_small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.mpv1 = nn.MaxPool2d(kernel_size=3, stride=(2, 1),padding=1)
        self.block1 = ResidualBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.mpv2 = nn.MaxPool2d(kernel_size=3, stride=(2, 1),padding=1)
        self.block2 = ResidualBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.mpv3 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.block3 = ResidualBlock(128, 192, kernel_size=3, stride=1, padding=1)
        self.mpv6 = nn.AdaptiveAvgPool2d((2, 2))
        self.Flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mpv1(x)
        x = self.block1(x)
        x = self.mpv2(x)
        x = self.block2(x)
        x = self.mpv3(x)
        x = self.block3(x)
        x = self.mpv6(x)
        x = self.Flatten(x)
        return x

