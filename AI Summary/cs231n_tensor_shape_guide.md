# CS231n Tensor Shape Guide

这份文档专门总结 `/Users/wrx/Desktop/learn` 这三份作业里最常见的张量维度变化。

它不强调算法推导，而强调另一件更实用的事：

> 看到一段代码时，你能不能立刻判断每个变量的 shape，知道它为什么这样变，以及哪里最容易写错。

如果前两份文档分别是：

- 方法总结
- 代码语言总结

那么这一份就是：

- 张量维度变化总表

---

## 1. 读 shape 的总原则

先记四句话：

1. 先看输入 shape，再看这层做了什么
2. `reshape/view` 通常不改数据，只改“看法”
3. `transpose/permute` 改的是维度顺序
4. `cat` 是沿旧维度拼接，`stack` 是新建一个维度

最常用的符号：

- \(N\): batch size
- \(D\): 特征维度
- \(H\): hidden dimension
- \(C\): 类别数或通道数，具体看上下文
- \(T\): 序列长度 / 时间步数
- \(V\): 词表大小
- \(F\): 卷积核个数
- \(P\): patch size
- \(E\): embedding dim
- \(h\): attention head 数
- \(d_h\): 每个 head 的维度

---

## 2. 最基础的 shape 模式

### 2.1 表格型数据

最常见：

$$
X \in \mathbb{R}^{N \times D}
$$

意思：

- 第 0 维是样本数
- 第 1 维是每个样本的特征长度

例子：

- 线性分类器输入
- 全连接层输入
- BatchNorm / LayerNorm 输入

---

### 2.2 图像数据

NumPy 在部分数据读入阶段常见：

$$
(N, H, W, C)
$$

PyTorch / 卷积网络最常见：

$$
(N, C, H, W)
$$

区别：

- `NHWC`: 通道在最后
- `NCHW`: 通道在前，PyTorch 卷积默认用这个

常见转换：

```python
X = X.transpose(0, 3, 1, 2)
```

从 `NHWC -> NCHW`

---

### 2.3 序列数据

词嵌入后常见：

$$
(N, T, D)
$$

含义：

- \(N\): batch 内有多少条序列
- \(T\): 每条序列长度
- \(D\): 每个位置的特征维度

例子：

- RNN 输入
- Transformer token embedding
- ViT patch token

---

## 3. NumPy 基础 shape 变化

---

### 3.1 `reshape(N, -1)`

最常见写法：

```python
x_row = x.reshape(x.shape[0], -1)
```

如果：

$$
x \in \mathbb{R}^{N \times d_1 \times d_2 \times \cdots \times d_k}
$$

那么：

$$
x\_row \in \mathbb{R}^{N \times D},\qquad D=d_1d_2\cdots d_k
$$

用途：

- 全连接层前把输入压平

例子：

```python
x.shape = (64, 3, 32, 32)
x_row.shape = (64, 3072)
```

最容易错的点：

- 忘了保留 batch 维
- 把 `(N, ...)` 错写成 `(-1,)`

---

### 3.2 `transpose`

例子：

```python
X = X.transpose(0, 2, 3, 1)
```

如果原来：

$$
(N, C, H, W)
$$

那么新 shape：

$$
(N, H, W, C)
$$

记法：

- 新第 0 维来自旧第 0 维
- 新第 1 维来自旧第 2 维
- 新第 2 维来自旧第 3 维
- 新第 3 维来自旧第 1 维

---

### 3.3 `sum/mean/max` 的 shape

若：

$$
x \in \mathbb{R}^{N \times D}
$$

则：

```python
np.sum(x, axis=0).shape == (D,)
np.sum(x, axis=1).shape == (N,)
np.sum(x, axis=1, keepdims=True).shape == (N, 1)
```

这在 softmax 和归一化里极其常见。

---

### 3.4 广播

如果：

$$
x \in \mathbb{R}^{N \times D},\quad b \in \mathbb{R}^{D}
$$

那么：

```python
x + b
```

的输出是：

$$
\mathbb{R}^{N \times D}
$$

因为 `b` 会自动广播到每一行。

更一般地：

- `(N, D)` 和 `(D,)` 可广播
- `(N, D)` 和 `(N, 1)` 可广播
- `(N, C, H, W)` 和 `(N, 1, 1, 1)` 可广播

---

## 4. 全连接网络中的 shape

源码参考：

- `/Users/wrx/Desktop/learn/assignment1/cs231n/layers.py`
- `/Users/wrx/Desktop/learn/assignment1/cs231n/classifiers/fc_net.py`
- `/Users/wrx/Desktop/learn/assignment2/cs231n/classifiers/fc_net.py`

---

### 4.1 `affine_forward`

输入：

$$
x \in \mathbb{R}^{N \times d_1 \times \cdots \times d_k}
$$

权重：

$$
w \in \mathbb{R}^{D \times M}
$$

偏置：

$$
b \in \mathbb{R}^{M}
$$

其中：

$$
D=d_1d_2\cdots d_k
$$

过程：

1. 压平输入
2. 做矩阵乘法
3. 加偏置

输出：

$$
out \in \mathbb{R}^{N \times M}
$$

例子：

```python
x.shape = (64, 3, 32, 32)
w.shape = (3072, 100)
b.shape = (100,)
out.shape = (64, 100)
```

---

### 4.2 `affine_backward`

上游梯度：

$$
dout \in \mathbb{R}^{N \times M}
$$

输出梯度：

$$
dx \in \mathbb{R}^{N \times d_1 \times \cdots \times d_k}
$$

$$
dw \in \mathbb{R}^{D \times M}
$$

$$
db \in \mathbb{R}^{M}
$$

记忆法：

- `db` 对 batch 维求和
- `dw` 是输入转置乘上游梯度
- `dx` 是上游梯度乘 `w^T`，再 reshape 回原形状

---

### 4.3 `relu_forward / relu_backward`

输入输出 shape 完全不变。

如果：

$$
x \in \mathbb{R}^{N \times D}
$$

则：

$$
out, dx \in \mathbb{R}^{N \times D}
$$

如果输入是四维图像：

$$
x \in \mathbb{R}^{N \times C \times H \times W}
$$

则输出也完全同 shape。

ReLU 是“逐元素”操作，所以不改维度。

---

### 4.4 TwoLayerNet 整体 shape

如果输入是 CIFAR10 图像压平后：

$$
X \in \mathbb{R}^{N \times 3072}
$$

设隐藏层维度 \(H=100\)，类别数 \(C=10\)。

参数：

$$
W_1 \in \mathbb{R}^{3072 \times 100},\qquad b_1 \in \mathbb{R}^{100}
$$

$$
W_2 \in \mathbb{R}^{100 \times 10},\qquad b_2 \in \mathbb{R}^{10}
$$

前向：

$$
X:(N,3072) \to hidden:(N,100) \to scores:(N,10)
$$

---

### 4.5 FullyConnectedNet 多层 shape

如果层维度链是：

$$
[3072, 100, 100, 10]
$$

那么：

- `W1`: `(3072, 100)`
- `b1`: `(100,)`
- `W2`: `(100, 100)`
- `b2`: `(100,)`
- `W3`: `(100, 10)`
- `b3`: `(10,)`

前向流：

$$
(N,3072)\to(N,100)\to(N,100)\to(N,10)
$$

这类题最建议先在纸上把每一层维度链写出来。

---

## 5. 归一化层的 shape

源码参考：

- `/Users/wrx/Desktop/learn/assignment2/cs231n/layers.py`

---

### 5.1 BatchNorm

输入：

$$
x \in \mathbb{R}^{N \times D}
$$

参数：

$$
\gamma,\beta \in \mathbb{R}^{D}
$$

中间量：

$$
\mu \in \mathbb{R}^{D},\quad
\sigma^2 \in \mathbb{R}^{D},\quad
\hat x \in \mathbb{R}^{N \times D}
$$

输出：

$$
out \in \mathbb{R}^{N \times D}
$$

为什么均值和方差是 `(D,)`？

因为 BatchNorm 对“每个特征维”统计 batch 内的均值方差。

---

### 5.2 LayerNorm

输入仍然是：

$$
x \in \mathbb{R}^{N \times D}
$$

但均值和方差是：

$$
\mu \in \mathbb{R}^{N \times 1},\quad
\sigma^2 \in \mathbb{R}^{N \times 1}
$$

因为 LayerNorm 对“每个样本内部”统计均值和方差。

输出仍然：

$$
out \in \mathbb{R}^{N \times D}
$$

---

### 5.3 Spatial BatchNorm

卷积输入：

$$
x \in \mathbb{R}^{N \times C \times H \times W}
$$

作业里的做法是先 reshape：

```python
x_reshape = x.transpose(0, 2, 3, 1).reshape(-1, C)
```

shape 变化：

$$
(N,C,H,W) \to (N,H,W,C) \to (NHW, C)
$$

然后调用普通 batchnorm。

最后再 reshape 回去：

$$
(NHW,C) \to (N,H,W,C) \to (N,C,H,W)
$$

这是一个经典技巧：

> 把卷积特征图问题转成“很多个样本，每个样本有 C 个通道特征”的普通 BN 问题。

---

### 5.4 GroupNorm

输入：

$$
x \in \mathbb{R}^{N \times C \times H \times W}
$$

若分成 \(G\) 组，先 reshape：

$$
(N,C,H,W)\to(N,G,C/G,H,W)\to(NG,(C/G)\cdot H\cdot W)
$$

你可以把它理解成：

- 把每个样本的通道拆成 \(G\) 组
- 每组内部单独做归一化

---

## 6. Dropout 的 shape

输入：

$$
x \in \mathbb{R}^{\text{任意形状}}
$$

mask：

$$
mask \in \mathbb{R}^{\text{同形状}}
$$

输出：

$$
out \in \mathbb{R}^{\text{同形状}}
$$

Dropout 是逐元素筛掉部分神经元，因此不会改 shape。

---

## 7. 卷积网络中的 shape

源码参考：

- `/Users/wrx/Desktop/learn/assignment2/cs231n/layers.py`
- `/Users/wrx/Desktop/learn/assignment2/cs231n/classifiers/cnn.py`

---

### 7.1 卷积层输入输出

输入：

$$
x \in \mathbb{R}^{N \times C \times H \times W}
$$

卷积核：

$$
w \in \mathbb{R}^{F \times C \times HH \times WW}
$$

偏置：

$$
b \in \mathbb{R}^{F}
$$

输出：

$$
out \in \mathbb{R}^{N \times F \times H' \times W'}
$$

其中：

$$
H' = 1 + \frac{H + 2pad - HH}{stride}
$$

$$
W' = 1 + \frac{W + 2pad - WW}{stride}
$$

---

### 7.2 池化层输入输出

输入：

$$
x \in \mathbb{R}^{N \times C \times H \times W}
$$

输出：

$$
out \in \mathbb{R}^{N \times C \times H' \times W'}
$$

池化不改通道数，只改空间分辨率。

例如 2x2 stride 2 池化：

$$
H' = H/2,\qquad W' = W/2
$$

---

### 7.3 `conv_relu_pool` 整体 shape

假设输入：

$$
(N, 3, 32, 32)
$$

卷积核个数 \(F=32\)，卷积保持尺寸不变，再做 2x2 池化。

那么：

1. 卷积后

$$
(N, 32, 32, 32)
$$

2. ReLU 后

$$
(N, 32, 32, 32)
$$

3. 池化后

$$
(N, 32, 16, 16)
$$

4. 若接全连接，要先压平成

$$
(N, 32 \cdot 16 \cdot 16)
$$

---

### 7.4 ThreeLayerConvNet 整体 shape

结构：

$$
\text{conv} \to \text{relu} \to \text{pool} \to \text{affine} \to \text{relu} \to \text{affine}
$$

假设：

- 输入 `(N, 3, 32, 32)`
- 卷积核数 `32`
- 隐藏层维度 `100`
- 类别数 `10`

那么：

1. 输入：

$$
(N,3,32,32)
$$

2. 卷积后：

$$
(N,32,32,32)
$$

3. 池化后：

$$
(N,32,16,16)
$$

4. 展平后：

$$
(N,8192)
$$

5. 全连接隐藏层：

$$
(N,100)
$$

6. 输出分数：

$$
(N,10)
$$

---

## 8. RNN / Captioning 中的 shape

源码参考：

- `/Users/wrx/Desktop/learn/assignment2/cs231n/rnn_layers_pytorch.py`
- `/Users/wrx/Desktop/learn/assignment2/cs231n/classifiers/rnn_pytorch.py`

---

### 8.1 词 id 到 embedding

输入词 id：

$$
x \in \mathbb{Z}^{N \times T}
$$

词表矩阵：

$$
W_{\text{embed}} \in \mathbb{R}^{V \times D}
$$

查表后：

$$
out \in \mathbb{R}^{N \times T \times D}
$$

例子：

```python
x.shape = (64, 16)
W_embed.shape = (1000, 128)
out.shape = (64, 16, 128)
```

---

### 8.2 单步 RNN

输入：

$$
x_t \in \mathbb{R}^{N \times D}
$$

前一隐藏状态：

$$
h_{t-1} \in \mathbb{R}^{N \times H}
$$

参数：

$$
W_x \in \mathbb{R}^{D \times H},\quad
W_h \in \mathbb{R}^{H \times H},\quad
b \in \mathbb{R}^{H}
$$

输出：

$$
h_t \in \mathbb{R}^{N \times H}
$$

---

### 8.3 整段 RNN

输入：

$$
x \in \mathbb{R}^{N \times T \times D}
$$

初始隐藏状态：

$$
h_0 \in \mathbb{R}^{N \times H}
$$

输出 hidden 序列：

$$
h \in \mathbb{R}^{N \times T \times H}
$$

代码里常写：

```python
h = []
for t in range(T):
    next_h = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
    h.append(next_h)
h = torch.stack(h, dim=1)
```

这里 `torch.stack(h, dim=1)` 是关键：

- 列表里每个元素 shape 是 `(N, H)`
- 堆起来后变成 `(N, T, H)`

---

### 8.4 图像描述整体 shape

输入图像特征：

$$
features \in \mathbb{R}^{N \times D_{\text{img}}}
$$

投影到初始 hidden：

$$
h_0 \in \mathbb{R}^{N \times H}
$$

caption 输入：

$$
captions\_in \in \mathbb{Z}^{N \times T}
$$

embedding 后：

$$
wordvecs \in \mathbb{R}^{N \times T \times W}
$$

RNN hidden：

$$
h \in \mathbb{R}^{N \times T \times H}
$$

词表分数：

$$
scores \in \mathbb{R}^{N \times T \times V}
$$

mask：

$$
mask \in \{0,1\}^{N \times T}
$$

---

### 8.5 `temporal_affine_forward`

输入：

$$
x \in \mathbb{R}^{N \times T \times D}
$$

参数：

$$
w \in \mathbb{R}^{D \times M},\quad b \in \mathbb{R}^{M}
$$

输出：

$$
out \in \mathbb{R}^{N \times T \times M}
$$

做法是：

$$
(N,T,D)\to(NT,D)\to(NT,M)\to(N,T,M)
$$

---

## 9. Transformer 中的 shape

源码参考：

- `/Users/wrx/Desktop/learn/assignment3/cs231n/transformer_layers.py`
- `/Users/wrx/Desktop/learn/assignment3/cs231n/classifiers/transformer.py`

---

### 9.1 位置编码

输入：

$$
x \in \mathbb{R}^{N \times S \times D}
$$

位置编码缓冲区：

$$
pe \in \mathbb{R}^{1 \times max\_len \times D}
$$

取前 `S` 个位置后：

$$
pe[:, :S, :] \in \mathbb{R}^{1 \times S \times D}
$$

广播加到 `x` 上：

$$
out \in \mathbb{R}^{N \times S \times D}
$$

---

### 9.2 Multi-Head Attention

输入：

$$
query \in \mathbb{R}^{N \times S \times E}
$$

$$
key,value \in \mathbb{R}^{N \times T \times E}
$$

线性映射后先不改变最后一维总长度：

$$
Q \in \mathbb{R}^{N \times S \times E}
$$

然后切 head：

$$
Q \to (N, S, h, d_h) \to (N, h, S, d_h)
$$

所以：

$$
q \in \mathbb{R}^{N \times h \times S \times d_h}
$$

同理：

$$
k,v \in \mathbb{R}^{N \times h \times T \times d_h}
$$

注意力分数：

$$
scores = qk^\top \in \mathbb{R}^{N \times h \times S \times T}
$$

softmax 后权重 shape 不变：

$$
attn \in \mathbb{R}^{N \times h \times S \times T}
$$

乘上 `v` 后：

$$
output \in \mathbb{R}^{N \times h \times S \times d_h}
$$

再拼回去：

$$
(N,h,S,d_h) \to (N,S,h,d_h) \to (N,S,E)
$$

---

### 9.3 Transformer Decoder Layer

输入：

$$
tgt \in \mathbb{R}^{N \times S \times D}
$$

视觉 memory：

$$
memory \in \mathbb{R}^{N \times T \times D}
$$

在作业 captioning 里，通常：

$$
T=1
$$

因为图像特征被当成一个视觉 token。

自注意力输出：

$$
(N,S,D)
$$

cross attention 输出：

$$
(N,S,D)
$$

FFN 输出：

$$
(N,S,D)
$$

所以 decoder layer 整体不改变 shape，只改变内容。

---

### 9.4 Captioning Transformer 整体 shape

图像特征：

$$
features \in \mathbb{R}^{N \times D_{\text{img}}}
$$

视觉投影：

$$
memory = visual\_projection(features) \in \mathbb{R}^{N \times D}
$$

插入 sequence 维：

$$
memory \in \mathbb{R}^{N \times 1 \times D}
$$

caption token：

$$
captions \in \mathbb{Z}^{N \times T}
$$

embedding 后：

$$
caption\_embeddings \in \mathbb{R}^{N \times T \times D}
$$

decoder 输出：

$$
decoder\_out \in \mathbb{R}^{N \times T \times D}
$$

词表打分：

$$
scores \in \mathbb{R}^{N \times T \times V}
$$

---

## 10. Vision Transformer 中的 shape

源码参考：

- `/Users/wrx/Desktop/learn/assignment3/cs231n/transformer_layers.py`
- `/Users/wrx/Desktop/learn/assignment3/cs231n/classifiers/transformer.py`

---

### 10.1 PatchEmbedding

输入图像：

$$
x \in \mathbb{R}^{N \times C \times H \times W}
$$

令 patch 大小为 \(P\)。

先 reshape：

$$
(N,C,H,W)\to(N,C,H/P,P,W/P,P)
$$

再 permute：

$$
(N,H/P,W/P,C,P,P)
$$

最后每个 patch 拉平：

$$
patches \in \mathbb{R}^{N \times num\_patches \times (P^2C)}
$$

其中：

$$
num\_patches = (H/P)(W/P)
$$

线性投影后：

$$
out \in \mathbb{R}^{N \times num\_patches \times D}
$$

---

### 10.2 ViT 整体 shape

输入：

$$
x \in \mathbb{R}^{N \times C \times H \times W}
$$

Patch embedding：

$$
(N, num\_patches, D)
$$

位置编码后：

$$
(N, num\_patches, D)
$$

经过 encoder：

$$
(N, num\_patches, D)
$$

做 mean pooling：

$$
(N, D)
$$

线性分类头：

$$
logits \in \mathbb{R}^{N \times C_{\text{class}}}
$$

---

## 11. SimCLR 中的 shape

源码参考：

- `/Users/wrx/Desktop/learn/assignment3/cs231n/simclr/model.py`
- `/Users/wrx/Desktop/learn/assignment3/cs231n/simclr/contrastive_loss.py`

---

### 11.1 双视图输入

同一张图做两次增强：

$$
x_i, x_j \in \mathbb{R}^{N \times 3 \times 32 \times 32}
$$

---

### 11.2 编码器和投影头

模型输出：

```python
return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
```

所以：

- `feature`: backbone 表征
- `out`: projection head 输出

若 backbone 最后是 ResNet50 全局池化结果，则：

$$
feature \in \mathbb{R}^{N \times 2048}
$$

若投影维度是 `128`，则：

$$
out \in \mathbb{R}^{N \times 128}
$$

---

### 11.3 相似度矩阵

左右两支输出拼接：

$$
out = [out\_left; out\_right] \in \mathbb{R}^{2N \times D}
$$

归一化后两两点积：

$$
sim\_matrix \in \mathbb{R}^{2N \times 2N}
$$

它的第 \((i,j)\) 个元素表示：

- 第 `i` 个样本表示
- 和第 `j` 个样本表示

的 cosine similarity。

正样本对在两条偏移 \(N\) 的对角线上。

---

## 12. CLIP 和 DINO 中的 shape

源码参考：

- `/Users/wrx/Desktop/learn/assignment3/cs231n/clip_dino.py`

---

### 12.1 CLIP 文本和图像特征

文本特征：

$$
text\_features \in \mathbb{R}^{N_t \times D}
$$

图像特征：

$$
image\_features \in \mathbb{R}^{N_i \times D}
$$

归一化后相似度矩阵：

$$
similarity = text\_features \cdot image\_features^\top \in \mathbb{R}^{N_t \times N_i}
$$

如果要做 zero-shot 分类：

- 每一列表示一张图像对所有类名的相似度

---

### 12.2 DINO patch token

若 DINO 输出：

$$
tok \in \mathbb{R}^{1 \times (1 + num\_patches) \times D}
$$

其中第一个 token 是 CLS token。

取 patch token：

```python
tok[0, 1:]
```

shape：

$$
(num\_patches, D)
$$

如果把多帧拼起来：

$$
res \in \mathbb{R}^{num\_frames \times num\_patches \times D}
$$

---

### 12.3 分割 mask 对齐

原始 mask 经过 resize 到 patch 网格：

$$
(H, W) \to (60, 60)
$$

再 flatten：

$$
(60,60)\to(3600,)
$$

这就和 patch token 数量对齐了。

---

## 13. Diffusion 中的 shape

源码参考：

- `/Users/wrx/Desktop/learn/assignment3/cs231n/gaussian_diffusion.py`

---

### 13.1 图像和时间步

图像：

$$
x_0, x_t \in \mathbb{R}^{N \times C \times H \times W}
$$

时间步：

$$
t \in \mathbb{Z}^{N}
$$

这里的 `t` 不是一个标量，而是 batch 内每张图自己的时间步。

---

### 13.2 `extract(a, t, x_shape)`

这是 diffusion 里最关键的 shape 工具之一。

如果：

- `a` 是长度为 `T_total` 的一维系数表
- `t` 是 `(N,)`

那么：

```python
out = a.gather(-1, t)
out = out.reshape(b, *((1,) * (len(x_shape) - 1)))
```

会把 shape 变成：

$$
(N,1,1,1)
$$

这样才能和图像：

$$
(N,C,H,W)
$$

正确广播。

这是 diffusion 最值得记住的 shape 技巧。

---

### 13.3 `q_sample`

输入：

$$
x_0 \in \mathbb{R}^{N \times C \times H \times W}
$$

$$
t \in \mathbb{Z}^{N}
$$

$$
noise \in \mathbb{R}^{N \times C \times H \times W}
$$

提取系数后：

$$
\sqrt{\bar\alpha_t} \in \mathbb{R}^{N \times 1 \times 1 \times 1}
$$

$$
\sqrt{1-\bar\alpha_t} \in \mathbb{R}^{N \times 1 \times 1 \times 1}
$$

最终广播得到：

$$
x_t \in \mathbb{R}^{N \times C \times H \times W}
$$

---

### 13.4 `sample` 循环 shape

初始化：

$$
img \in \mathbb{R}^{N \times C \times H \times W}
$$

每次 `p_sample` 输入输出 shape 都一样：

$$
x_t \to x_{t-1}
$$

shape 始终：

$$
(N,C,H,W)
$$

所以 diffusion 采样的难点不在维度变化，而在：

- 系数广播
- 时间步索引
- 数值稳定

---

## 14. U-Net 中的 shape

源码参考：

- `/Users/wrx/Desktop/learn/assignment3/cs231n/unet.py`

---

### 14.1 时间 embedding

输入：

$$
time \in \mathbb{R}^{N}
$$

经过 `SinusoidalPosEmb(dim)`：

$$
(N, dim)
$$

再过 `time_mlp`：

$$
context \in \mathbb{R}^{N \times 4dim}
$$

文本条件经过 `condition_mlp` 后也是：

$$
(N, 4dim)
$$

两者相加后仍然：

$$
(N, 4dim)
$$

---

### 14.2 ResnetBlock 中的 `scale_shift`

MLP 输出：

$$
context \to \mathbb{R}^{N \times 2C}
$$

然后：

```python
context = context[:, :, None, None]
scale_shift = context.chunk(2, dim=1)
```

shape 变化：

$$
(N,2C)\to(N,2C,1,1)
$$

再拆成：

$$
scale,shift \in \mathbb{R}^{N \times C \times 1 \times 1}
$$

这样就能广播到卷积特征图：

$$
x \in \mathbb{R}^{N \times C \times H \times W}
$$

---

### 14.3 Downsample / Upsample

若输入：

$$
x \in \mathbb{R}^{N \times C \times H \times W}
$$

下采样后通常：

$$
(N,C',H/2,W/2)
$$

上采样后通常：

$$
(N,C',2H,2W)
$$

---

### 14.4 Skip connection 拼接

在 decoder 中：

```python
x = torch.cat([x, skips.pop()], dim=1)
```

表示沿通道维拼接。

如果：

- 当前 `x`: `(N, C1, H, W)`
- skip: `(N, C2, H, W)`

那么拼完：

$$
(N, C1 + C2, H, W)
$$

这就是为什么后面的 `ResnetBlock` 输入通道数通常写成 `dim_out * 2`。

---

## 15. 一眼判断 `cat`、`stack`、`reshape`、`permute` 的区别

这是最容易混的地方，单独列出来。

### 15.1 `reshape`

只改视角，不改元素总数。

例子：

$$
(N, C, H, W) \to (N, CHW)
$$

---

### 15.2 `permute` / `transpose`

只改维度顺序，不改元素总数。

例子：

$$
(N, C, H, W) \to (N, H, W, C)
$$

---

### 15.3 `cat`

沿已有维度拼接。

例子：

$$
(N,D) + (N,D) \xrightarrow{dim=0} (2N,D)
$$

或者：

$$
(N,C_1,H,W) + (N,C_2,H,W) \xrightarrow{dim=1} (N,C_1+C_2,H,W)
$$

---

### 15.4 `stack`

新建一个维度。

例子：

若列表中有 `T` 个 `(N,H)`：

$$
[(N,H), (N,H), \dots] \xrightarrow{stack(dim=1)} (N,T,H)
$$

---

## 16. 最容易出 shape bug 的地方

这部分是专门给调试用的。

### 16.1 全连接层前忘记压平

错误：

- 直接把 `(N,C,H,W)` 送进 `(D,M)` 的线性层

修正：

- 先 `reshape(N, -1)`

### 16.2 `axis` 用错

尤其在：

- softmax
- batchnorm
- layernorm

### 16.3 `cat` 维度选错

尤其在 U-Net：

- skip connection 必须沿通道维拼接，不是 batch 维

### 16.4 `permute` 后直接 `view`

有时需要：

```python
x = x.permute(...).contiguous().view(...)
```

### 16.5 mask shape 不匹配

Transformer 中：

- score 通常是 `(N,h,S,T)`
- mask 可能原始是 `(S,T)`

所以要扩维：

```python
mask.unsqueeze(0).unsqueeze(0)
```

### 16.6 diffusion 系数没 reshape 成 `(N,1,1,1)`

如果直接拿 `(N,)` 去乘 `(N,C,H,W)`，广播逻辑通常不对。

---

## 17. 读代码时的 shape 检查模板

以后你看到一段代码，可以直接套这个模板：

### 17.1 先写输入

```text
x: ?
w: ?
b: ?
```

### 17.2 每一行后面都写 shape

例如：

```python
x_row = x.reshape(N, -1)      # (N, D)
out = x_row @ w + b           # (N, M)
```

### 17.3 再问这一行属于哪一类

每一行代码通常只属于以下四类之一：

1. 改 shape
2. 做数学运算
3. 做索引/查表
4. 组织流程

### 17.4 最后再问数学含义

例如：

- 这是 affine
- 这是 softmax
- 这是 attention score
- 这是 diffusion broadcast 系数

这样会非常稳。

---

## 18. 最后的总结

如果把这份文档压成最核心的几句话，就是：

1. 你读深度学习代码时，第一反应应该是 shape，不是语法
2. 大多数“看不懂的复杂代码”，本质上只是 `reshape / transpose / 广播 / 索引`
3. 真正难的往往不是公式，而是“这个公式在 batch 和多维张量上怎么落地”
4. 只要你开始习惯每一行代码都在脑中标 shape，复杂模型会突然变得顺很多

最值得你长期记住的一个习惯：

> 先写 shape，再写代码；先看 shape，再看代码。

