from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pdb
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        # self.conv1= nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ClassificationBranch(nn.Module):
    def __init__(self, inplanes, planes):
        super(ClassificationBranch, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.linear1 = nn.Linear(in_features=planes, out_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(in_features=64, out_features=32)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lineaer3 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.lineaer3(x)

        return x


class ResNetMod34(nn.Module):
    def __init__(self, block, block2, layers, num_classes=21):
        super().__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.classificationBranch = ClassificationBranch(inplanes=64, planes=128)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 1, stride=2)

        self.layer32 = self._make_layer(block, 256, layers[2])
        self.layer42 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(512 * block.expansion, 3)

        self.inplanes = 256
        self.layer31 = self._make_layer(block2, 128, layers[4], stride=2)
        self.layer41 = self._make_layer(block2, 256, layers[5], stride=2)
        self.layer51 = self._make_layer(block2, 512, layers[6], stride=2)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block2.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        x = self.layer1(x)  # 56x56
        out1 = self.classificationBranch(x)
        x = self.layer2(x)  # 28x28
        x = self.layer3(x)  # 14x14

        x1 = self.layer31(x)
        x1 = self.layer41(x1)  # 7x7
        x1 = self.layer51(x1)
        x1 = self.avgpool1(x1)  # 1x1
        x1 = torch.flatten(x1, 1)  # remove 1 X 1 grid and make vector of tensor shape
        x1 = self.fc1(x1)

        x2 = self.layer32(x)
        x2 = self.layer42(x2)
        x2 = self.avgpool2(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc2(x2)

        return out1, x1, x2


class ResNetMod(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        super().__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.classificationBranch = ClassificationBranch(inplanes=256, planes=128)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 1, stride=2)

        self.layer31 = self._make_layer(block, 256, layers[2])
        self.layer41 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.inplanes = 1024
        self.layer32 = self._make_layer(block, 256, layers[2])
        self.layer42 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(512 * block.expansion, 3)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        x = self.layer1(x)  # 56x56
        out1 = self.classificationBranch(x)
        x = self.layer2(x)  # 28x28
        x = self.layer3(x)  # 14x14

        x1 = self.layer31(x)
        x1 = self.layer41(x1)  # 7x7
        x1 = self.avgpool1(x1)  # 1x1
        x1 = torch.flatten(x1, 1)  # remove 1 X 1 grid and make vector of tensor shape
        x1 = self.fc1(x1)

        x2 = self.layer32(x)
        x2 = self.layer42(x2)
        x2 = self.avgpool2(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc2(x2)

        return out1, x1, x2


'''def initializeWeights():
    modelmod34.conv1.load_state_dict(torch.load('resnet34conv1.pth'))
    modelmod34.layer1.load_state_dict(torch.load('resnet34layer1.pth'))
    modelmod34.layer2.load_state_dict(torch.load('resnet34layer2.pth'))
    modelmod34.layer42.load_state_dict(torch.load('resnet34layer4.pth'))
    modelmod34.layer31.load_state_dict(torch.load('resnet50layer2.pth'))
    modelmod34.layer41.load_state_dict(torch.load('resnet50layer3.pth'))
    modelmod34.layer51.load_state_dict(torch.load('resnet50layer4.pth'))


def resnet34():
    layers = [3, 4, 6, 3]
    model = ResNet(BasicBlock, layers)
    return model


def resnet50():
    layers = [3, 4, 6, 3]
    model = ResNet(Bottleneck, layers)
    return model'''


def resnetmod50():
    layers = [3, 4, 5, 3]
    model = ResNetMod(Bottleneck, layers)
    return model


def resnetmod34():
    layers = [3, 4, 5, 3, 4, 6, 3]
    model = ResNetMod34(BasicBlock, Bottleneck, layers)
    return model
