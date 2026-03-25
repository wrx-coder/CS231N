# 05. 从数学公式到代码的翻译能力

## 为什么这是一项核心能力

CS231n 最有价值的一层训练，不只是“会写几个模型”，而是让你不断做一件事：

> 把数学公式翻译成可以运行、可以训练、可以调试的代码。

很多人学机器学习时，会出现一种断层：

- 公式看懂了，但不会写
- 代码能抄，但不知道对应哪条数学关系

如果你把“公式到代码”的转换链打通，之后学论文、复现模型、读源码都会快很多。

## 一、翻译时最先要识别的 5 个元素

看到一条公式，不要立刻敲代码。先拆成五件事。

## 1. 对象是什么

例如：

- 向量？
- 矩阵？
- batch 张量？
- token 序列？
- 图像特征图？

如果对象识别错了，后面的代码几乎一定会错。

## 2. 参数是什么

哪些是可学习参数，哪些是输入，哪些是中间量？

例如在 softmax 分类器里：

\[
scores = XW + b
\]

这里：

- \(X\) 是输入
- \(W, b\) 是参数
- `scores` 是中间输出

## 3. 运算类型是什么

常见运算包括：

- 矩阵乘
- 按元素乘
- 求和
- 均值
- 归一化
- 索引采样
- 拼接
- mask

翻译代码时，这一步特别重要。

## 4. 运算发生在哪个轴上

例如：

\[
\sum_{j=1}^{C} e^{s_{ij}}
\]

这说明：

- 固定样本 \(i\)
- 沿类别维 \(j\) 求和

代码里就应该是：

```python
torch.sum(torch.exp(scores), dim=1)
```

或如果类别维在最后：

```python
torch.sum(torch.exp(scores), dim=-1)
```

## 5. 输出语义是什么

你最后得到的是：

- 标量 loss？
- 每个样本一个 loss？
- 每个位置一个表示？
- 每个 token 的 logits？

输出语义不清楚，代码很容易 shape 对了但意义错了。

## 二、公式翻译成代码的标准流程

推荐你以后统一按下面这 6 步走。

### 第一步：写出张量维度

例如 softmax 分类：

\[
X \in \mathbb{R}^{N \times D}, \quad
W \in \mathbb{R}^{D \times C}, \quad
scores \in \mathbb{R}^{N \times C}
\]

没有 shape 的公式，很难直接安全落地。

### 第二步：识别广播关系

例如：

\[
out = xW + b
\]

这里 `b` 常常是 `shape=(C,)`，代码里依赖广播加到 batch 上。

### 第三步：把求和、均值和索引落到具体 `dim`

例如 temporal softmax loss 中，要沿词表维做 softmax，而不是沿 batch 维。

### 第四步：处理数值稳定性

很多公式的“教科书形式”不能直接照抄到代码里。

需要补上：

- `subtract max`
- `logsumexp`
- `eps`
- `clamp`
- mask

### 第五步：决定缓存哪些中间量

如果你要手写 backward，就必须提前决定保存什么。

例如 affine 层常缓存：

- 原始输入 `x`
- 权重 `w`
- 偏置 `b`

### 第六步：写一个最小可验证版本

不要一开始就写成超级复杂版本。先写最小版，再做矢量化和优化。

## 三、assignment1 中最经典的公式翻译例子

## 1. Softmax loss

公式：

\[
p_{ic} = \frac{e^{s_{ic}}}{\sum_{j=1}^{C} e^{s_{ij}}}
\]

\[
L = -\frac{1}{N}\sum_{i=1}^{N} \log p_{i,y_i}
\]

翻译思路：

- `scores` 是 `(N, C)`
- 先沿类别维做稳定 softmax
- 再用标签索引取出正确类概率
- 再求平均

代码模板：

```python
scores = X @ W
scores = scores - scores.max(axis=1, keepdims=True)
exp_scores = np.exp(scores)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
loss = -np.log(probs[np.arange(N), y]).mean()
```

这里的关键翻译点：

- \(\sum_j\) 对应 `axis=1`
- \(p_{i,y_i}\) 对应高级索引 `probs[np.arange(N), y]`

## 2. ReLU

公式：

\[
\mathrm{ReLU}(x) = \max(0, x)
\]

代码：

```python
out = np.maximum(0, x)
```

反向：

\[
\frac{\partial \mathrm{ReLU}}{\partial x}
=
\mathbf{1}(x > 0)
\]

代码：

```python
dx = dout * (x > 0)
```

这个例子说明：

- 很多公式最终就是布尔 mask + 按元素乘

## 四、assignment2 中更复杂的翻译例子

## 1. BatchNorm

公式：

\[
\mu = \frac{1}{N}\sum_i x_i
\]

\[
\sigma^2 = \frac{1}{N}\sum_i (x_i - \mu)^2
\]

\[
\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

\[
y = \gamma \hat x + \beta
\]

翻译重点：

- `mean` 和 `var` 沿 batch 维算
- `gamma` 和 `beta` 按特征维 broadcast

代码模板：

```python
mean = x.mean(axis=0)
var = x.var(axis=0)
x_hat = (x - mean) / np.sqrt(var + eps)
out = gamma * x_hat + beta
```

## 2. Temporal softmax loss

序列任务中，logits 通常是：

\[
scores \in \mathbb{R}^{N \times T \times V}
\]

如果你直接照二维 softmax 写，就会很乱。

典型做法是先 reshape：

```python
scores_flat = scores.reshape(N * T, V)
y_flat = y.reshape(N * T)
mask_flat = mask.reshape(N * T)
```

这一步的思想非常重要：

> 先把复杂张量变成熟悉的标准形，再复用已有公式。

## 五、assignment3 中最关键的翻译例子

## 1. Attention

公式：

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]

\[
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
\]

\[
out = AV
\]

翻译重点：

- `QK^T` 的“转置”通常只转最后两个维度
- softmax 沿 key 维做
- 若是 multi-head，要先拆 head 再合并回来

PyTorch 模板：

```python
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)

q = q.view(N, T, h, d_h).permute(0, 2, 1, 3)
k = k.view(N, T, h, d_h).permute(0, 2, 1, 3)
v = v.view(N, T, h, d_h).permute(0, 2, 1, 3)

scores = q @ k.transpose(-2, -1) / math.sqrt(d_h)
attn = torch.softmax(scores, dim=-1)
out = attn @ v
```

这个例子教会你：

- 数学里的 \(T\) 可以变成代码里的 `transpose(-2, -1)`
- “沿哪个维 softmax”比公式本身更重要

## 2. Diffusion 的前向加噪

公式：

\[
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
\]

翻译重点：

- \(t\) 是每个样本一个时间索引
- 系数表是长度为 `T_total` 的数组
- 要按 batch 索引抽取后 reshape 成可 broadcast 的形状

代码模板：

```python
coef1 = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
coef2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
x_t = coef1 * x_start + coef2 * noise
```

这里真正重要的翻译点不是乘法本身，而是：

- “按索引取对应时间系数”
- “把标量系数扩展成整张图可乘的 shape”

## 六、从公式到代码时，最容易犯的 8 种错误

### 1. 把矩阵乘写成按元素乘

例如把 \(XW\) 误写成 `X * W`。

### 2. 求和维度写错

例如应该沿类别维求和，却沿 batch 维求了。

### 3. 忘记保留维度

例如：

```python
sum = x.sum(dim=1)
```

后续如果需要 broadcast，可能其实应该写：

```python
sum = x.sum(dim=1, keepdim=True)
```

### 4. 索引语义错

例如 `y` 是 label index，不是 one-hot。

### 5. 训练态和测试态混淆

比如 BN、dropout、autoregressive decode。

### 6. 忘记数值稳定处理

softmax、log、归一化都容易出问题。

### 7. 理论上是标量，代码里却成了向量

有时候你以为自己在算总 loss，其实得到的是 per-sample loss。

### 8. 手写 backward 时缓存不全

这在 NumPy 作业里特别常见。

## 七、建议你以后固定使用的一套翻译模板

看到公式时，建议先写成下面这种笔记：

```text
1. 输入对象：
   x: (N, D)
   W: (D, C)

2. 运算：
   scores = x @ W + b
   softmax over class dim
   select correct-class prob by index y

3. 输出：
   loss: scalar
   dscores: (N, C)

4. 数值稳定：
   subtract row max before exp

5. backward 需要缓存：
   x, W, probs, y
```

这会极大降低“公式懂了但代码写不出来”的情况。

## 八、真正高级的“翻译能力”是什么

成熟之后，你会发现自己不再只是机械对应公式，而是开始问：

- 这条公式在代码里最自然的表示是什么？
- 有没有更稳定的等价写法？
- 有没有更容易向量化的形式？
- 哪些中间量应该保留，哪些可以即时重算？

也就是说，真正的翻译不是“逐字照搬”，而是“在数学意义不变的前提下，用计算机更友好的形式实现它”。

## 最后一句总结

从公式到代码的能力，是连接“会做题”和“会研究、会实现”的桥梁。

你以后学任何新模型时，都可以反复提醒自己：

> 先识别对象，写清 shape，找出运算轴，处理数值稳定，再把公式翻译成一段最小可验证代码。
