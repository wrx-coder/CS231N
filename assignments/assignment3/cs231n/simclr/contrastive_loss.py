import numpy as np
import torch


def sim(z_i, z_j):
    """Normalized dot product between two vectors."""
    # 归一化点积就是 cosine similarity。
    norm_dot_product = torch.dot(z_i, z_j) / (
        torch.linalg.norm(z_i) * torch.linalg.norm(z_j)
    )
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version)."""
    N = out_left.shape[0]
    # 把两种增强视图拼起来，形成长度 2N 的特征列表。
    out = torch.cat([out_left, out_right], dim=0)

    total_loss = 0
    for k in range(N):
        # z_k 和 z_{k+N} 是同一张原图经过两次增强得到的正样本对。
        z_k, z_k_N = out[k], out[k + N]

        # 分子只放正样本对的相似度；分母放除了自己之外的所有候选。
        numerator_1 = torch.exp(sim(z_k, z_k_N) / tau)
        denom_1 = 0
        for i in range(2 * N):
            if i != k:
                denom_1 += torch.exp(sim(z_k, out[i]) / tau)
        loss_1 = -torch.log(numerator_1 / denom_1)

        numerator_2 = torch.exp(sim(z_k_N, z_k) / tau)
        denom_2 = 0
        for i in range(2 * N):
            if i != k + N:
                denom_2 += torch.exp(sim(z_k_N, out[i]) / tau)
        loss_2 = -torch.log(numerator_2 / denom_2)

        total_loss += loss_1 + loss_2

    # 每个原样本会贡献两个方向的 loss，所以最后除以 2N。
    total_loss = total_loss / (2 * N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs."""
    # 按行计算每一对正样本的 cosine similarity，输出形状是 (N, 1)。
    pos_pairs = torch.sum(out_left * out_right, dim=1, keepdim=True) / (
        torch.linalg.norm(out_left, dim=1, keepdim=True)
        * torch.linalg.norm(out_right, dim=1, keepdim=True)
    )
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products."""
    # 先把每个表示归一化，再通过矩阵乘法一次性得到所有两两相似度。
    out_norm = out / torch.linalg.norm(out, dim=1, keepdim=True)
    sim_matrix = out_norm @ out_norm.t()
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device="cuda"):
    """Compute the contrastive loss L over a batch (vectorized version)."""
    N = out_left.shape[0]
    out = torch.cat([out_left, out_right], dim=0)
    sim_matrix = compute_sim_matrix(out)

    # 指数化后的相似度会进入 InfoNCE 的分子和分母。
    exponential = torch.exp(sim_matrix / tau)
    device = out.device if hasattr(out, "device") else device
    # mask 去掉对角线上的“自己和自己”，因为它不是合法负样本。
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).bool()
    exponential = exponential.masked_select(mask).view(2 * N, -1)
    denom = exponential.sum(dim=1, keepdim=True)

    # 正样本对位于 sim_matrix 的第 +N 和 -N 条对角线上。
    positive_pairs = torch.cat(
        [torch.diag(sim_matrix, N), torch.diag(sim_matrix, -N)], dim=0
    ).view(2 * N, 1)
    numerator = torch.exp(positive_pairs / tau)
    loss = (-torch.log(numerator / denom)).mean()
    return loss


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
