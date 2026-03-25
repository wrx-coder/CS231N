import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                # CIFAR10 分辨率更小，所以把第一层卷积改成 3x3 / stride 1 更合适。
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                # 去掉原始分类头和最大池化，保留卷积特征提取器。
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head: 训练对比损失时主要作用在投影空间，而不是直接在 backbone feature 上。
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        # feature 是 backbone 的表示，常用于下游检索 / kNN 评估。
        feature = torch.flatten(x, start_dim=1)
        # out 是 projection head 的输出，用于计算 SimCLR 对比损失。
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
