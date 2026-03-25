import math

import torch
import torch.nn as nn
from torch.nn import functional as F

"""
This file defines layer types that are commonly used for transformers.
"""


class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """

    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        # 这里默认用 sin/cos 成对编码，因此 embedding 维度必须是偶数。
        assert embed_dim % 2 == 0

        # pe 的形状是 (1, max_len, D)，第 0 维保留给 batch 方向广播。
        pe = torch.zeros(1, max_len, embed_dim)
        # position: 每个 token 的绝对位置下标，形状 (max_len, 1)。
        position = torch.arange(max_len, dtype=pe.dtype).unsqueeze(1)
        # div_term 控制不同维度上的频率，让不同通道编码不同尺度的位置变化。
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=pe.dtype)
            * (-math.log(10000.0) / embed_dim)
        )
        # 偶数维放 sin，奇数维放 cos，这样同一个位置会得到一对相位不同的编码。
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # register_buffer 表示这不是可学习参数，但会跟随模型搬到 cpu/gpu 并参与保存。
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        _, S, _ = x.shape
        # 只截取当前序列长度 S 对应的位置编码，然后和 token embedding 逐元素相加。
        output = x + self.pe[:, :S, :].to(dtype=x.dtype)
        # dropout 放在 embedding + position 之后，和原始 Transformer 一致。
        output = self.dropout(output)
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # 多头注意力要求总维度能被 head 数整除，这样每个 head 才有相同 head_dim。
        assert embed_dim % num_heads == 0

        # 分别线性映射出 Q / K / V，最后再做一次输出投影。
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        # head_dim: 每个注意力头内部工作的子空间维度。
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E)
        """
        N, S, E = query.shape
        _, T, _ = value.shape

        # 先把总 embedding 投影成 Q/K/V，再 reshape 成多头格式 (N, n_head, seq_len, head_dim)。
        q = self.query(query).view(N, S, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(key).view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(value).view(N, T, self.n_head, self.head_dim).transpose(1, 2)

        # 注意力打分 = QK^T / sqrt(head_dim)，缩放是为了避免维度大时 softmax 过尖。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            # attn_mask 的 1 表示允许看，0 表示禁止看；这里转成 bool 后做 masked_fill。
            mask = attn_mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # softmax 后得到“每个 query 位置对所有 key 位置”的注意力分布。
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        # 用注意力权重对 V 做加权求和，得到每个 head 的输出。
        output = torch.matmul(attn, v)
        # 把多头重新拼回原始 embedding 维度，再过线性层混合各 head 的信息。
        output = output.transpose(1, 2).contiguous().view(N, S, E)
        output = self.proj(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        # Transformer block 内部的 MLP：先升维，再非线性，再降回 embed_dim。
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        # x: (N, S, D)，这里对每个 token 独立地应用同一套两层感知机。
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, to be used with TransformerDecoder.
    """

    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Decoder block 有三段：masked self-attn -> cross-attn -> FFN。
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_cross = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        # 第一段 masked self-attention：caption 里当前位置只能看自己和前面词。
        shortcut = tgt
        tgt = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = self.dropout_self(tgt)
        # 残差连接 + LayerNorm 是 Transformer block 的标准结构。
        tgt = tgt + shortcut
        tgt = self.norm_self(tgt)

        # 第二段 cross-attention：让文本 token 去“查询”图像特征 memory。
        shortcut = tgt
        tgt = self.cross_attn(query=tgt, key=memory, value=memory)
        tgt = self.dropout_cross(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_cross(tgt)

        # 第三段 FFN：在每个 token 内部再做一次非线性特征变换。
        shortcut = tgt
        tgt = self.ffn(tgt)
        tgt = self.dropout_ffn(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_ffn(tgt)

        return tgt


class PatchEmbedding(nn.Module):
    """
    A layer that splits an image into patches and projects each patch to an embedding vector.
    Used as the input layer of a Vision Transformer (ViT).
    """

    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=128):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # ViT 会把图像切成规则 patch，因此边长必须能整除 patch_size。
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        # num_patches: 一张图总共有多少个 patch；patch_dim: 一个 patch 拉平后的长度。
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        # proj 把“像素块向量”映射成 Transformer 使用的 token embedding。
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        N, C, H, W = x.shape
        # 这里固定输入分辨率，避免 patch 切分后 token 数不一致。
        assert H == self.img_size and W == self.img_size, (
            f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
        )

        P = self.patch_size
        # 先把图像按 patch 网格重排，再把每个 patch 拉平成一行。
        patches = x.reshape(N, C, H // P, P, W // P, P)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(N, self.num_patches, self.patch_dim)
        # 输出形状是 (N, num_patches, embed_dim)，可以直接当作 token 序列送进 Transformer。
        out = self.proj(patches)
        return out


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, to be used with TransformerEncoder.
    """

    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Encoder block 只有 self-attention 和 FFN，没有 cross-attention。
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # self-attention 负责在整段输入序列内部做信息交互。
        shortcut = src
        src = self.self_attn(query=src, key=src, value=src, attn_mask=src_mask)
        src = self.dropout_self(src)
        src = src + shortcut
        src = self.norm_self(src)

        # FFN 负责在每个 token 内部做更强的非线性特征提炼。
        shortcut = src
        src = self.ffn(src)
        src = self.dropout_ffn(src)
        src = src + shortcut
        src = self.norm_ffn(src)

        return src
