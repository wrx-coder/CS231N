import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10


def compute_train_transform(seed=123456):
    """
    This function returns a composition of data augmentations to a single training image.
    """
    random.seed(seed)
    torch.random.manual_seed(seed)

    # SimCLR 的关键不只是模型，而是“强数据增强”制造出足够难的正样本对。
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

    train_transform = transforms.Compose(
        [
            # 随机裁剪 + 翻转 + 颜色扰动 + 灰度化，共同构成论文里的两视图增强策略。
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    return train_transform


def compute_test_transform():
    # 测试阶段不做强增强，只做标准化，保持输入稳定。
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    return test_transform


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset."""

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            # 对同一张原图独立调用两次 transform，得到 SimCLR 需要的正样本对 (x_i, x_j)。
            x_i = self.transform(img)
            x_j = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target
