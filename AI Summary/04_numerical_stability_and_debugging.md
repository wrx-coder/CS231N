# 04. 数值稳定性与调试能力

## 为什么这项能力值得单独写一篇

很多人学深度学习时，会把注意力都放在：

- 算法名字
- 网络结构
- 论文公式

但真正一上手写代码，就会发现另一条隐藏主线：

- 为什么 loss 变成了 `nan`？
- 为什么 accuracy 完全不涨？
- 为什么梯度一开始就爆掉？
- 为什么同样的公式，代码结果和预期差很多？

CS231n 的真正训练之一，就是让你在实践里碰到这些问题，再慢慢学会稳定地排查它们。

数值稳定性和调试能力，往往决定你能不能把一个“理论上对的模型”真正跑起来。

## 一、常见异常现象与它们通常意味着什么

### 1. loss 直接变成 `nan`

通常意味着：

- 除零
- 对非常小的数取对数
- exponent 溢出
- 梯度爆炸
- 非法的归一化或方差计算

### 2. loss 很大但不下降

通常意味着：

- 学习率太大或太小
- 梯度方向错了
- 训练模式和测试模式混了
- label、mask、正负样本定义错了

### 3. accuracy 卡在随机水平

通常意味着：

- 数据标签对应错位
- logits 维度错
- backward 有 bug
- 输出头或 loss 目标不匹配

### 4. 一开始正常，过几轮后崩掉

通常意味着：

- 学习率过大
- running statistics 漂移
- 梯度逐渐爆炸
- 混合精度或 dtype 问题

## 二、assignment1 里最经典的数值稳定点

## 1. Softmax overflow

Softmax:

\[
p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}
\]

如果某个 \(s_i\) 很大，\(e^{s_i}\) 会溢出。

标准做法是减去最大值：

\[
p_i = \frac{e^{s_i - \max(s)}}{\sum_j e^{s_j - \max(s)}}
\]

这一步不会改变结果，因为分子分母都乘了同一个常数。

这是一种你以后会不断遇到的思路：

> 通过等价变形，把计算移到数值更安全的区间。

## 2. log(0) 问题

交叉熵里有：

\[
-\log p_{y}
\]

如果 \(p_y\) 极小甚至被数值下溢成 0，就会出问题。

工程上常见处理：

- 先做稳定 softmax / log-softmax
- 或在计算中加很小的 \(\epsilon\)

## 3. 数值梯度检查

assignment1 很重要的一点是让你看到：

\[
\frac{f(x+h)-f(x-h)}{2h}
\]

可以作为梯度近似来验证手写 backward。

这教会你一个特别重要的调试原则：

> 不要直接相信自己写的梯度，先验证。

## 三、assignment2 中更复杂的稳定性问题

## 1. BatchNorm / LayerNorm

BatchNorm 的归一化形式：

\[
\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

这里的 \(\epsilon\) 不是装饰，它是数值稳定必需品。

如果方差很小，没有 \(\epsilon\)，分母就可能接近 0。

你还要注意：

- train 和 test 模式不同
- running mean / var 更新是否合理
- 统计轴是否写对

很多 BN 错误不是数学公式错，而是轴和模式错。

## 2. Dropout 的训练 / 测试一致性

inverted dropout 的好处是：

- 训练时直接缩放
- 测试时不需要额外处理

如果这里缩放忘了，或者 train/test 混了，网络虽然能跑，但结果会非常怪。

## 3. 卷积和池化里的边界与索引

卷积最容易出的问题包括：

- pad 后索引越界
- 输出高宽公式算错
- backward 时覆盖和累加逻辑写错

这些不一定会马上报错，但会让梯度 subtly wrong。

## 四、assignment3 里的现代稳定性问题

## 1. Attention 的 scale 和 mask

缩放点积注意力：

\[
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
\]

为什么要除以 \(\sqrt{d_k}\)？

因为当维度大时，点积的方差会变大，不缩放的话 softmax 会非常尖锐，梯度变差。

mask 也特别关键：

- causal mask 方向不能反
- padding mask 维度要能 broadcast
- 有些实现用 `-inf`，有些用极小负数

mask 出错通常不是直接崩，而是模型“学得很怪”。

## 2. SimCLR 的相似度与温度

InfoNCE 常见形式：

\[
\ell = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\mathrm{sim}(z_i, z_k)/\tau)}
\]

这里有几个常见稳定点：

- 特征要先 normalize
- 温度 \(\tau\) 不宜乱设
- 对角线和正样本索引要排对

如果正负样本掩码错了，loss 数值可能看起来正常，但语义完全错。

## 3. Diffusion 的系数提取和 broadcast

扩散模型特别容易在系数 shape 上踩坑。

例如：

\[
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
\]

其中 \(\sqrt{\bar\alpha_t}\) 对每个样本只对应一个标量时间系数，但要 broadcast 到整张图上。

所以常见实现会有类似：

```python
coef = extract(self.sqrt_alphas_cumprod, t, x.shape)
```

如果这个提取与 reshape 错了，公式看起来没问题，结果会完全偏掉。

## 五、你应该形成的一套调试顺序

遇到问题时，建议按下面顺序查，不要乱撞。

## 1. 先查 shape

这是第一优先级。

很多数值异常其实来自 shape 错误，导致广播逻辑悄悄改变了数学意义。

## 2. 再查数值范围

打印：

- `min`
- `max`
- `mean`
- `std`
- 是否存在 `nan` / `inf`

例如：

```python
print(x.min().item(), x.max().item(), x.mean().item(), x.std().item())
```

## 3. 再查极小样本是否能过拟合

这是很强的实验诊断法。

如果一个小 batch 都过拟合不了，说明：

- 代码有 bug
- loss 定义错
- 优化器配置不对
- 模型根本没拿到正确监督信号

## 4. 再查梯度

看：

- 梯度是不是全 0
- 梯度是不是极大
- 不同层梯度量级是否离谱

## 5. 再查模式切换

确认：

- `model.train()`
- `model.eval()`
- dropout / norm / sampling 逻辑是否切换正确

## 6. 最后再怀疑“算法思想”

很多时候问题并不在方法本身，而是实现细节和训练配置。

## 六、很值得长期记住的工程模板

### 模板一：稳定 softmax

```python
scores = scores - scores.max(dim=-1, keepdim=True).values
log_probs = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
```

### 模板二：归一化时加 epsilon

```python
x_hat = (x - mean) / torch.sqrt(var + eps)
```

### 模板三：attention mask 用极小值屏蔽

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
attn = torch.softmax(scores, dim=-1)
```

### 模板四：训练前做小样本过拟合测试

```python
small_loader = take_few_batches(loader)
train_until_overfit(small_loader)
```

### 模板五：关键节点加断言

```python
assert x.shape[-1] == expected_dim
assert torch.isfinite(loss).all()
```

## 七、从“会排错”到“会设计不容易错的代码”

成熟的工程习惯，不只是出问题后会修，还包括提前降低出错概率。

例如：

- 给变量起清楚名字
- 明确区分 logits / probs / embeddings
- 在关键函数入口写 shape 注释
- 把 mask 逻辑封装好，不要散落在各处
- 把 train/test 行为差异写清楚

很多“神秘 bug”本质上都是可读性差带来的。

## 八、你可以给自己准备的一张调试检查表

以后只要训练出问题，就按这张表过一遍：

| 检查项 | 关键问题 |
| --- | --- |
| 数据 | 输入范围对吗？标签对吗？增强过头了吗？ |
| shape | 每个关键张量维度对吗？ |
| 数值 | 有没有 NaN / Inf？激活和梯度范围正常吗？ |
| loss | 目标定义、mask、正负样本、索引都对吗？ |
| 模式 | train/eval 是否切换正确？ |
| 优化 | 学习率、weight decay、调度是否合理？ |
| 最小实验 | 能否在一个小 batch 上过拟合？ |

## 最后一句总结

数值稳定性和调试能力，不是附加技能，而是深度学习落地能力的核心部分。

真正成熟的学习者，不是“从来不写错代码”，而是遇到问题时知道：

- 先查什么
- 怎么定位
- 哪些地方最容易出错
- 如何把错误缩小到一个可理解的小范围内
