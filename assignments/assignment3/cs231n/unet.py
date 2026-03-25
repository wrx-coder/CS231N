import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


def exists(x):
    # 小工具函数：统一判断一个对象是不是 None。
    return x is not None


def default(val, d):
    # val 存在就用 val，否则退回默认值 d。
    if exists(val):
        return val
    return d() if callable(d) else d


def Upsample(dim, dim_out=None):
    # 上采样先放大分辨率，再用卷积整理通道信息。
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # 下采样用 stride=2 的卷积同时完成降分辨率和通道调整。
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=2, stride=2)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # 这里沿通道维做归一化，保持每个空间位置的特征尺度更稳定。
        return F.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # 和 Transformer 类似，把时间步 t 映射成不同频率的 sin/cos 编码。
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.GELU()

    def forward(self, x, scale_shift=None):
        # 基础卷积块：Conv -> Norm -> 可选条件调制 -> GELU。
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            # scale_shift 来自时间/条件嵌入，用于 FiLM 风格的条件注入。
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, context_dim):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.context_dim = context_dim

        # 把 context 映射成两份参数：一份控制缩放，一份控制平移。
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(context_dim, dim_out * 2))
            if exists(context_dim)
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, context=None):
        scale_shift = None
        if exists(self.mlp) and exists(context):
            # context 先过 MLP，再 reshape 成 (N, C, 1, 1)，方便广播到特征图上。
            context = self.mlp(context)
            context = context[:, :, None, None]
            scale_shift = context.chunk(2, dim=1)

        # 两个 block 组成一个 ResNet 单元，中间带 dropout，最后加残差支路。
        h = self.block1(x, scale_shift=scale_shift)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        condition_dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        uncond_prob=0.2,
    ):
        super().__init__()

        # init_conv 先把输入图像映射到基础通道数 dim。
        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.channels = channels

        # dims 定义 U-Net 每个分辨率层级上的通道数。
        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        in_out_ups = [(b, a) for a, b in reversed(in_out)]

        context_dim = dim * 4
        # time_mlp 把扩散时间步 t 编码成 context 向量。
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        self.condition_dim = condition_dim
        # condition_mlp 把文本/条件 embedding 投影到和 time context 同维度。
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # uncond_prob 用于 classifier-free guidance 训练时的条件 dropout。
        self.uncond_prob = uncond_prob
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for dim_in, dim_out in in_out:
            # 每个下采样层级做两次残差块，再降一半分辨率。
            down_block = nn.ModuleList(
                [
                    ResnetBlock(dim_in, dim_in, context_dim),
                    ResnetBlock(dim_in, dim_in, context_dim),
                    Downsample(dim_in, dim_out),
                ]
            )
            self.downs.append(down_block)

        mid_dim = dims[-1]
        # bottleneck 位于 U-Net 最底部，分辨率最低、感受野最大。
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)

        for dim_in, dim_out in in_out_ups:
            # 上采样先提升分辨率，再和 skip connection 拼接后做两次残差块。
            up_block = nn.ModuleList(
                [
                    Upsample(dim_in, dim_out),
                    ResnetBlock(dim_out * 2, dim_out, context_dim),
                    ResnetBlock(dim_out * 2, dim_out, context_dim),
                ]
            )
            self.ups.append(up_block)

        # final_conv 把最后的特征图映射回图像通道数，输出噪声图或 x_0 估计。
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def cfg_forward(self, x, time, model_kwargs={}):
        # classifier-free guidance: 同时跑有条件和无条件两次前向，再线性组合。
        cfg_scale = model_kwargs.pop("cfg_scale")
        print("Classifier-free guidance scale:", cfg_scale)

        cond_kwargs = copy.deepcopy(model_kwargs)
        uncond_kwargs = copy.deepcopy(model_kwargs)
        uncond_kwargs["text_emb"] = None

        cond = self.forward(x, time, cond_kwargs)
        uncond = self.forward(x, time, uncond_kwargs)
        # 公式可理解为：在无条件预测的基础上，沿着“条件带来的变化方向”再推远一点。
        x = (cfg_scale + 1) * cond - cfg_scale * uncond
        return x

    def forward(self, x, time, model_kwargs={}):
        if "cfg_scale" in model_kwargs:
            return self.cfg_forward(x, time, model_kwargs)

        # 先得到时间条件向量，后面每个 ResNet block 都会用到它。
        context = self.time_mlp(time)

        cond_emb = model_kwargs["text_emb"]
        if cond_emb is None:
            # 无条件分支时，用全 0 条件向量代替。
            cond_emb = torch.zeros(x.shape[0], self.condition_dim, device=x.device)
        if self.training:
            # 训练阶段随机丢掉一部分条件，给 classifier-free guidance 留出无条件能力。
            mask = (torch.rand(cond_emb.shape[0], device=cond_emb.device) > self.uncond_prob).float()
            cond_emb = cond_emb * mask[:, None]
        # 最终 context = 时间信息 + 文本条件信息。
        context = context + self.condition_mlp(cond_emb)

        x = self.init_conv(x)

        skips = []
        for block1, block2, downsample in self.downs:
            # 每个层级保存两次中间特征，供 decoder 端做 skip connection。
            x = block1(x, context)
            skips.append(x)
            x = block2(x, context)
            skips.append(x)
            x = downsample(x)

        # 中间 bottleneck 持续用同一份 context 调制特征。
        x = self.mid_block1(x, context)
        x = self.mid_block2(x, context)

        for upsample, block1, block2 in self.ups:
            x = upsample(x)
            # decoder 把当前特征和 encoder 保存的高分辨率细节拼接起来恢复图像结构。
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, context)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block2(x, context)

        x = self.final_conv(x)
        return x
