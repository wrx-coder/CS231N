import pandas as pd
import torch
import torch.optim as optim
from thop import clever_format, profile
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from .contrastive_loss import *


def train(
    model,
    data_loader,
    train_optimizer,
    epoch,
    epochs,
    batch_size=32,
    temperature=0.5,
    device="cuda",
):
    """Trains the model defined in ./model.py with one epoch."""
    model.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_pair in train_bar:
        x_i, x_j, target = data_pair
        x_i, x_j = x_i.to(device), x_j.to(device)

        # 两个增强视图分别过同一个编码器，得到用于对比学习的表示。
        _, out_left = model(x_i)
        _, out_right = model(x_j)
        loss = simclr_loss_vectorized(out_left, out_right, temperature, device=x_i.device)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # 这里按 batch 累加，便于最后算整个 epoch 的平均损失。
        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(
            "Train Epoch: [{}/{}] Loss: {:.4f}".format(epoch, epochs, total_loss / total_num)
        )

    return total_loss / total_num


def train_val(model, data_loader, train_optimizer, epoch, epochs, device="cuda"):
    # train_optimizer 为 None 时就走验证流程，否则走监督训练流程。
    is_train = train_optimizer is not None
    model.train() if is_train else model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            # prediction 保存每个样本从高到低的类别排序，用来算 top-1 / top-5。
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description(
                "{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                    "Train" if is_train else "Test",
                    epoch,
                    epochs,
                    total_loss / total_num,
                    total_correct_1 / total_num * 100,
                    total_correct_5 / total_num * 100,
                )
            )

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def test(model, memory_data_loader, test_data_loader, epoch, epochs, c, temperature=0.5, k=200, device="cuda"):
    model.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        for data, _, target in tqdm(memory_data_loader, desc="Feature extracting"):
            # 先给 memory set 建立一个特征库，后面测试样本会和它做 kNN 检索。
            feature, out = model(data.to(device))
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature, out = model(data)

            total_num += data.size(0)
            # sim_matrix 的每一行表示一个测试样本和所有 memory 特征的相似度。
            sim_matrix = torch.mm(feature, feature_bank)
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # one_hot_label 把 k 个邻居的标签展开成 one-hot，后面按相似度加权投票。
            one_hot_label = torch.zeros(data.size(0) * k, c, device=device)
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            pred_scores = torch.sum(
                one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1),
                dim=1,
            )

            # 分数最高的类别就是 kNN 分类结果。
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description(
                "Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%".format(
                    epoch,
                    epochs,
                    total_top1 / total_num * 100,
                    total_top5 / total_num * 100,
                )
            )

    return total_top1 / total_num * 100, total_top5 / total_num * 100
