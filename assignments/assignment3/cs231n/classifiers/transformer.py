import copy
import numpy as np

import torch
import torch.nn as nn

from ..transformer_layers import *


class CaptioningTransformer(nn.Module):
    """
    A CaptioningTransformer produces captions from image features using a
    Transformer decoder.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim,
        wordvec_dim,
        num_heads=4,
        num_layers=2,
        max_length=50,
    ):
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        # 特殊 token 的索引后面会频繁用到：padding、句子起点、句子终点。
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # 把图像特征投影到和词向量同维度，方便后续 cross-attention 对齐。
        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        # embedding 把词 id 变成稠密向量；padding_idx 会让 <NULL> 的梯度处理更自然。
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)

        # decoder_layer 是一个“模板层”，后面会复制 num_layers 份堆叠起来。
        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        # 输出头把每个时间步的隐藏状态映射成整个词表上的打分。
        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        # 统一初始化方式，尽量和课程作业里给出的 Transformer 设置保持一致。
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        N, T = captions.shape
        # scores 最终形状 (N, T, V)，V 是词表大小。
        scores = torch.empty((N, T, self.vocab_size), device=captions.device)

        # caption_embeddings: 每个单词先转成 embedding，再叠加位置编码。
        caption_embeddings = self.embedding(captions)
        caption_embeddings = self.positional_encoding(caption_embeddings)

        # memory: 把图像特征投影成 decoder 可以 cross-attend 的“视觉 token”。
        memory = self.visual_projection(features).unsqueeze(1)
        # 下三角 mask 保证第 t 个词预测时看不到未来词，避免训练时信息泄露。
        tgt_mask = torch.tril(torch.ones(T, T, device=captions.device, dtype=torch.bool))

        # decoder_out 保留每个时间步的上下文表示。
        decoder_out = self.transformer(caption_embeddings, memory, tgt_mask=tgt_mask)
        scores = self.output(decoder_out)
        return scores

    def sample(self, features, max_length=30):
        with torch.no_grad():
            device = self.visual_projection.weight.device
            features = torch.as_tensor(features, dtype=torch.float32, device=device)
            N = features.shape[0]

            # captions 保存最终采样结果；初始先全部填 <NULL>。
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # partial_caption 是当前已经生成出来的前缀，初始只有 <START>。
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(device).unsqueeze(1)

            for t in range(max_length):
                # 每一步都把“当前前缀”重新送进 decoder，取最后一个位置的词表分数。
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]
                # 这里用贪心解码：直接选概率最大的词，而不是采样。
                word = torch.argmax(output_logits, axis=1)
                captions[:, t] = word.cpu().numpy()
                # 把新预测出的词接到前缀后面，供下一轮继续生成。
                partial_caption = torch.cat([partial_caption, word.unsqueeze(1)], dim=1)

            return captions


def clones(module, N):
    # deep copy 的作用是让每一层拥有独立参数，而不是共享同一份权重。
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt
        # 顺序通过多层 decoder block，让文本逐层整合历史词和图像信息。
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_mask=None):
        output = src
        # Encoder 是纯 self-attention 堆叠，常用于 ViT 或文本编码器。
        for mod in self.layers:
            output = mod(output, src_mask=src_mask)
        return output


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.
    """

    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=128,
        num_layers=6,
        num_heads=4,
        dim_feedforward=256,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        # 先把图像切成 patch token，再送入 Transformer Encoder。
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        # x -> patch tokens，形状从 (N, C, H, W) 变成 (N, num_patches, D)。
        x = self.patch_embed(x)
        x = self.positional_encoding(x)
        # 编码器让所有 patch token 彼此交换信息。
        x = self.transformer(x)
        # 这里用 mean pooling 聚合整张图的信息，而不是额外引入 [CLS] token。
        x = torch.mean(x, dim=1)
        logits = self.head(x)
        return logits
