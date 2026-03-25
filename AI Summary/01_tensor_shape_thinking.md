# 01. 张量维度思维

## 这份笔记想解决什么问题

很多人学深度学习时，真正卡住的不是公式，而是：

- 这个张量到底是什么？
- 这一维表示 batch、channel、time 还是 token？
- 为什么这里要 `reshape`，那里要 `permute`？
- 为什么代码一跑就报 shape mismatch？

CS231n 三份作业其实一直在训练同一个核心能力：`先看清维度，再理解计算`。

你可以把“张量维度思维”理解成一种读代码的母语。只要这一层足够稳定，很多复杂模型就会突然变得清楚。

## 一、为什么 shape 是第一优先级

在神经网络里，很多公式表面上看起来很像，真正决定含义的是维度。

例如：

\[
XW
\]

这个乘法可以表示很多完全不同的事情：

- 在 MLP 里，它可能是样本特征乘分类权重
- 在词嵌入里，它可能是 token 特征乘投影矩阵
- 在 attention 里，它可能是 token 表示乘 \(W_Q\)、\(W_K\)、\(W_V\)
- 在 diffusion 里，它可能是某种条件投影或时间嵌入投影

所以读代码时，不能只看运算符，要先问：

- 输入是什么对象？
- 每一维对应什么语义？
- 运算后哪个维度被混合了，哪个维度被保留了？

## 二、最常见的维度语义

在三份作业里，最常见的是下面几类维度：

| 记号 | 常见含义 | 典型出现位置 |
| --- | --- | --- |
| \(N\) | batch size | 所有任务 |
| \(D\) | 特征维度 | 线性层、MLP、embedding |
| \(C\) | 类别数 / 通道数 | 分类、卷积 |
| \(H, W\) | 空间高宽 | 图像、卷积、U-Net |
| \(T\) | 序列长度 / 时间步 | RNN、Transformer、diffusion 时间索引 |
| \(V\) | 词表大小 | captioning、language head |
| \(F\) | 卷积核个数 | CNN |
| \(P\) | patch size | ViT |
| \(h\) | attention head 数 | Transformer |
| \(d_h\) | 单个 head 的维度 | Multi-head attention |

你读代码时，应该习惯在脑中把一个张量立即翻译成：

- 它是什么对象组成的集合？
- 最外层是不是 batch？
- 内部是不是时间、空间或 token 维？
- 最后一维是不是 feature/channel/embedding？

## 三、assignment1 到 assignment3 的 shape 进化

### 1. assignment1：二维世界

assignment1 的大多数核心张量都很规整：

\[
X \in \mathbb{R}^{N \times D}, \quad
W \in \mathbb{R}^{D \times C}, \quad
scores \in \mathbb{R}^{N \times C}
\]

这一阶段最重要的训练是：

- 先理解 batch 维 \(N\)
- 再理解“样本特征维” \(D\)
- 最后理解输出类别维 \(C\)

这是最基本的“矩阵视角”。

### 2. assignment2：三维和四维世界

进入 CNN 和 RNN 后，shape 不再只是二维。

卷积里：

\[
X \in \mathbb{R}^{N \times C \times H \times W}
\]

RNN 里：

\[
X \in \mathbb{R}^{N \times T \times D}
\]

这时你要学会问：

- 如果这是图像，哪个维度是空间？哪个是通道？
- 如果这是序列，哪个维度是时间？哪个是 feature？

### 3. assignment3：多视角与多语义世界

Transformer、CLIP、SimCLR、Diffusion 会把同一个张量系统分成更多语义层次：

- batch 维
- token 维
- head 维
- per-head feature 维
- 条件维
- 时间步维
- patch 网格维

例如 multi-head attention：

\[
Q, K, V \in \mathbb{R}^{N \times T \times D}
\]

拆 head 后：

\[
Q, K, V \in \mathbb{R}^{N \times h \times T \times d_h}
\]

光看数字没有意义，必须清楚：

- \(h\) 是并行注意力子空间数
- \(T\) 是 token 数
- \(d_h\) 是每个 head 内部的子特征维

## 四、读任何一段模型代码时的 shape 问法

建议把下面这套问法养成习惯。

### 第一步：先问输入是什么

不要先看公式，先看函数签名。

例如：

```python
def forward(self, x):
    ...
```

这时你要马上找：

- `x` 的 shape 是多少？
- 每一维是什么意思？
- 这个函数有没有默认假设，例如图像输入默认是 `NCHW`？

### 第二步：标出每一层“改变了什么维”

读一层层代码时，重点不是把每行都背下来，而是看：

- 哪一层改变了最后一维？
- 哪一层改变了空间分辨率？
- 哪一层增加了 token 数？
- 哪一层只是重排维度，没有改变信息内容？

例如：

```python
x = x.permute(0, 2, 3, 1)
```

这行通常只是重排视角，不是新的数学变换。

### 第三步：区分“语义变换”和“布局变换”

有些操作真正改变表示：

- `Linear`
- `Conv`
- `Attention`
- `MLP`

有些操作只是为了后续计算方便：

- `reshape`
- `view`
- `permute`
- `transpose`
- `flatten`
- `unsqueeze`
- `squeeze`

这两类操作在脑中要分开。

## 五、作业里最值得记住的 shape 模板

### 1. 全连接层

\[
X \in \mathbb{R}^{N \times D}, \quad
W \in \mathbb{R}^{D \times M}, \quad
out \in \mathbb{R}^{N \times M}
\]

代码模板：

```python
out = x.reshape(N, -1) @ W + b
```

读法：

- `reshape(N, -1)`：把非 batch 维全展平
- `@ W`：沿着最后那个特征维做线性映射
- 输出最后一维变成新特征维 `M`

### 2. 卷积层

\[
X \in \mathbb{R}^{N \times C \times H \times W}
\]

\[
W \in \mathbb{R}^{F \times C \times HH \times WW}
\]

\[
out \in \mathbb{R}^{N \times F \times H' \times W'}
\]

读法：

- batch \(N\) 不变
- 输入通道 \(C\) 被卷积核读入
- 输出通道变成卷积核个数 \(F\)
- 空间大小是否变化由 stride 和 pad 决定

### 3. RNN

\[
X \in \mathbb{R}^{N \times T \times D}
\]

\[
h \in \mathbb{R}^{N \times T \times H}
\]

读法：

- 同一 batch 中每个样本是一段长度为 \(T\) 的序列
- 每个时刻有 \(D\) 维输入
- 每个时刻输出 \(H\) 维隐藏状态

### 4. Transformer attention

输入：

\[
X \in \mathbb{R}^{N \times T \times D}
\]

投影后：

\[
Q, K, V \in \mathbb{R}^{N \times T \times D}
\]

拆 head：

\[
Q, K, V \in \mathbb{R}^{N \times h \times T \times d_h}
\]

注意力分数：

\[
A \in \mathbb{R}^{N \times h \times T \times T}
\]

输出：

\[
out \in \mathbb{R}^{N \times T \times D}
\]

你应该能一眼看出：

- 最后两个 \(T\) 表示“query token 对 key token 的关系”
- 注意力矩阵不是“特征图”，而是“token 间关系图”

### 5. SimCLR

\[
out\_left, out\_right \in \mathbb{R}^{N \times D}
\]

拼接后：

\[
out \in \mathbb{R}^{2N \times D}
\]

相似度矩阵：

\[
sim \in \mathbb{R}^{2N \times 2N}
\]

这是非常经典的读法训练：

- 行和列都代表样本
- 对角线是自己和自己，不参与正负样本比较

### 6. Diffusion

图像：

\[
x_0, x_t \in \mathbb{R}^{N \times C \times H \times W}
\]

时间步：

\[
t \in \mathbb{Z}^{N}
\]

条件嵌入：

\[
cond \in \mathbb{R}^{N \times D}
\]

这里最容易错的是：`t` 虽然只是一维索引，但会通过系数提取函数 broadcast 到图像张量的 shape。

## 六、reshape / permute / transpose 到底在干什么

### 1. `reshape` / `view`

作用：

- 重新解释内存布局
- 不改变元素总数
- 常用于把多个维度合并或拆开

典型场景：

```python
x = x.reshape(N, -1)
```

含义：

- 保留 batch 维
- 其余所有维展平成一个特征维

### 2. `permute`

作用：

- 交换维度顺序
- 改变“看待张量”的角度

典型场景：

```python
x = x.permute(0, 2, 3, 1)
```

含义：

- 从 `NCHW` 变成 `NHWC`

### 3. `transpose`

在二维时常等同于矩阵转置；在高维时是交换某两个轴。

Transformer 里很常见：

```python
k.transpose(-2, -1)
```

含义：

- 把最后两个维度交换
- 用于把 `(..., T, d_h)` 变成 `(..., d_h, T)`，从而计算 \(QK^\top\)

## 七、如何把 shape 读成“流程图”

当你面对复杂网络时，不要一上来读每一行实现。先写一张 shape 流程图。

例如图像分类网络可以写成：

\[
(N, C, H, W)
\to
(N, F, H', W')
\to
(N, F, H'' , W'')
\to
(N, D)
\to
(N, C_{cls})
\]

例如 captioning Transformer 可以写成：

\[
features: (N, D_{img})
\to
memory: (N, 1, D)
\]

\[
captions: (N, T)
\to
embeddings: (N, T, D)
\to
decoder\ output: (N, T, D)
\to
logits: (N, T, V)
\]

只要这张图画出来，整个模型就已经理解了一半。

## 八、shape 错误的排查模板

如果代码报错，建议按这个顺序查：

### 1. 打印每个关键节点的 shape

```python
print("x:", x.shape)
print("q:", q.shape)
print("attn:", attn.shape)
```

### 2. 检查最后一维是否和线性层输入匹配

很多错误都来自：

- `Linear(in_features, out_features)` 的 `in_features` 设错
- `flatten` 后维度和预期不一致

### 3. 检查 mask 的维度能否 broadcast

attention 和 loss 中的 mask 很常出问题。

### 4. 检查 `permute` 后是否需要 `contiguous()`

在 PyTorch 里，有些 `view` 操作要求内存连续。

### 5. 检查“你以为保留的维度”是否其实被求和掉了

例如：

```python
x.sum(dim=1)
```

如果你没加 `keepdim=True`，这个维就真的没了。

## 九、真正成熟的 shape 思维是什么样

成熟的 shape 思维，不是“记住几个 API”，而是形成以下直觉：

- 一个张量不是一堆数字，而是一种组织结构
- 维度顺序本身就是语义
- 线性层大多作用在最后一个特征维
- 卷积主要作用在局部空间邻域
- attention 主要建立 token 与 token 的关系
- `reshape` 常常是“换解释方式”，不是“换数学意义”

当你能做到下面这件事时，说明你真的进步了：

> 还没看具体实现，只看函数输入输出和几行 shape 变化，就能大致猜出这段代码在做什么。

## 十、建议你怎么继续训练这项能力

### 方法一：每读一个函数，先手写 shape 注释

例如：

```python
# x: (N, T, D)
# qkv: (N, T, 3D)
# q, k, v: (N, h, T, d_h)
```

### 方法二：强迫自己把每个模型画成箭头图

不用画得很漂亮，只要明确：

- 输入 shape
- 中间 shape
- 输出 shape

### 方法三：反着练

看到输出 shape，倒推输入和中间结构。

例如：

- 为什么 attention score 是 `(N, h, T, T)`？
- 为什么 diffusion 里 `extract(...)` 之后要 reshape 成 `(N, 1, 1, 1)`？

### 方法四：把 shape 当成最重要的调试日志

一旦模型跑不通，先查 shape，不要先怀疑“数学错了”。

## 最后一句总结

在 CS231n 这三份作业里，shape 不是辅助信息，它本身就是模型结构的一部分。

真正的进步不是“我记住了很多层的名字”，而是“我看到一个张量时，知道它是按什么逻辑组织起来的”。这项能力会陪你走很久。
