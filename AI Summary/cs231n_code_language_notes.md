# CS231n Code Language Notes

这份文档不讲算法本身，而是专门总结你在 `/Users/wrx/Desktop/learn` 这三份大作业里反复看到的代码语言规则和常用写法。

目标不是做一份“Python 教科书”，而是做一份和你现在代码水平最贴近的“作业代码语言速查表”。

重点覆盖：

1. Python 基础语法和代码组织
2. NumPy 的数组、广播、索引、矩阵运算
3. PyTorch 的 Tensor、Module、Autograd、训练循环
4. Matplotlib / PIL / OpenCV / DataLoader 等常见工具
5. 这些写法在 CS231n 作业里分别起什么作用

---

## 1. 怎么使用这份文档

建议把这份文档当成以下几种场景的工具：

- 看到一行代码能猜到它的 shape 在怎么变
- 看见 `axis=0`、`keepdims=True`、`permute`、`view` 不再发懵
- 看见 `nn.Module`、`optimizer.step()`、`torch.no_grad()` 知道它们在训练流程里的位置
- 看见 `zip`、`enumerate`、`f-string`、字典推导式等写法时，能直接读过去

一句话版本：

> 这份文档是“读懂作业代码”的语言层底座。

---

## 2. Python 基础语法

先记一个最重要的原则：

> 你在 CS231n 里看到的 Python，不是为了写花哨语法，而是为了更清楚地表达数据流、模块结构和训练流程。

---

### 2.1 变量、对象、引用

Python 里变量更像“名字绑定到对象”，不是 C/C++ 那种“变量盒子”直觉。

例如：

```python
a = [1, 2, 3]
b = a
b.append(4)
```

这时 `a` 和 `b` 指向的是同一个列表，所以 `a` 也会变。

这件事在作业里非常重要，因为：

- `config` 字典会被原地更新
- `w -= lr * dw` 会直接改原数组
- `model_kwargs` 可能在函数里被修改

所以你会在代码里看到：

```python
copy.deepcopy(module)
copy.deepcopy(model_kwargs)
```

意思是：

- 我要一份真正独立的副本
- 后面的修改不能影响原对象

---

### 2.2 函数定义

最普通的函数：

```python
def affine_forward(x, w, b):
    out = x.reshape(x.shape[0], -1) @ w + b
    return out
```

几个常见点：

- `def` 定义函数
- `return` 返回结果
- 参数按位置传递

带默认参数：

```python
def predict(self, X, k=1, num_loops=0):
    ...
```

意思是：

- 如果用户不传 `k`，默认就是 `1`
- 如果用户不传 `num_loops`，默认就是 `0`

这在作业里很常见，比如：

- `dropout=0.1`
- `batch_size=100`
- `device="cuda"`

---

### 2.3 `*args` 和 `**kwargs`

作业里更常见的是 `**kwargs`。

例如：

```python
def __init__(self, model, data, **kwargs):
    self.learning_rate = kwargs.pop("learning_rate", 0.001)
```

意思是：

- 把额外命名参数统统打包进一个字典 `kwargs`
- 再从里面取需要的配置项

这让构造函数更灵活。

你可以把它理解成：

```python
Solver(model, data, learning_rate=1e-3, batch_size=64)
```

等价于：

```python
kwargs = {
    "learning_rate": 1e-3,
    "batch_size": 64,
}
```

`pop(key, default)` 的意思是：

- 取出这个 key 的值
- 如果没有，就给默认值
- 同时把这个 key 从字典中删除

---

### 2.4 类和 `self`

CS231n 的核心模型几乎都是类。

例如：

```python
class TwoLayerNet(object):
    def __init__(self, input_dim, hidden_dim, num_classes):
        self.params = {}

    def loss(self, X, y=None):
        ...
```

几个关键点：

- `class` 定义类
- `__init__` 是构造函数
- `self` 表示“当前对象本身”

所以：

```python
self.params["W1"] = ...
```

意思是：

- 给这个网络对象存一个成员变量 `params`
- 后面别的函数也能访问它

如果你看到：

```python
self.model = model
self.data = data
```

就是把传进来的对象保存到当前实例中。

---

### 2.5 面向对象里最常见的三类东西

在这三份作业里，类大致分成三种：

1. 模型类

- `TwoLayerNet`
- `FullyConnectedNet`
- `ThreeLayerConvNet`
- `CaptioningRNN`
- `CaptioningTransformer`
- `VisionTransformer`
- `Unet`
- `GaussianDiffusion`

2. 训练器 / 求解器类

- `Solver`
- `CaptioningSolverPytorch`
- `CaptioningSolverTransformer`
- `Trainer`

3. 数据类

- `EmojiDataset`
- `CIFAR10Pair`
- `DavisDataset`

你可以把它们的职责记成：

- 模型类：定义 `forward / loss / sample`
- 训练器类：定义 `train / _step`
- 数据类：定义 `__len__ / __getitem__`

---

### 2.6 `if / elif / else`

作业里最常见的逻辑分支写法：

```python
if mode == "train":
    ...
elif mode == "test":
    ...
else:
    raise ValueError(...)
```

常见用途：

- 区分 train / test 模式
- 区分 `rnn` / `lstm`
- 区分 `batchnorm` / `layernorm` / `None`
- 区分 `pred_noise` / `pred_x_start`

要点：

- `if` 第一个分支
- `elif` 中间分支
- `else` 兜底分支

---

### 2.7 `for` 循环

最常见：

```python
for i in range(num_test):
    ...
```

`range(n)` 表示：

- 从 `0` 到 `n-1`

倒序循环：

```python
for i in reversed(range(1, self.num_layers)):
    ...
```

表示从后往前遍历。

这在 backward 和 diffusion 采样里特别常见。

时间步循环：

```python
for t in range(T):
    ...
```

采样倒序：

```python
for t in reversed(range(self.num_timesteps)):
    ...
```

---

### 2.8 `enumerate`

当你既要元素，又要下标时：

```python
for i, img_file in enumerate(img_files):
    ...
```

这比手动写 `for i in range(len(img_files))` 更自然。

作业里常见用途：

- 遍历类别
- 遍历图片路径
- 遍历样本列表

---

### 2.9 `zip`

当你要并排遍历多个列表时：

```python
for feature_fn, feature_dim in zip(feature_fns, feature_dims):
    ...
```

意思是：

- 第一个 `feature_fn` 配第一个 `feature_dim`
- 第二个配第二个

在你的作业里常出现在：

- 特征函数和对应维度一起遍历
- 图像和标签一起显示
- 文本和图像一起处理

---

### 2.10 列表推导式和字典推导式

列表推导式：

```python
[line.strip() for line in f]
```

字典推导式：

```python
{i: w for w, i in word_to_idx.items()}
```

这两种写法在作业里极常见，因为它们适合做“轻量级批量变换”。

理解方式：

- 列表推导式就是“把 for 循环写成一行”
- 字典推导式就是“批量构造映射”

---

### 2.11 切片

切片是读代码的重灾区。

```python
captions[:, :-1]
captions[:, 1:]
```

意思：

- `:` 表示这一维全取
- `:-1` 表示从头取到倒数第一个之前
- `1:` 表示从第 1 个位置取到最后

对 captioning 来说：

```python
captions_in = captions[:, :-1]
captions_out = captions[:, 1:]
```

刚好形成“输入前缀”和“右移一格的监督目标”。

再比如：

```python
self.pe[:, :S, :]
```

表示：

- 取 batch 维全部
- 只取前 `S` 个位置编码
- embedding 维全部保留

---

### 2.12 断言 `assert`

作业里经常用 `assert` 做早期检查：

```python
assert embed_dim % num_heads == 0
assert H == self.img_size and W == self.img_size
```

作用：

- 提前发现输入不合法
- 防止 shape 错了以后往后面传，难调试

你可以把它理解成：

> 如果这个条件不成立，立刻停下来报错。

---

### 2.13 `pass`

`pass` 的意思是“这里先占一个空位，什么都不做”。

在课程 starter code 里很常见：

```python
def batchnorm_backward(...):
    pass
```

意思不是“这个函数没意义”，而是：

> 这里留给你实现。

---

### 2.14 `lambda`

你在作业里看到过类似：

```python
register_buffer = lambda name, val: self.register_buffer(name, val.float())
```

`lambda` 是匿名函数，适合写很短的函数。

等价于：

```python
def register_buffer(name, val):
    self.register_buffer(name, val.float())
```

如果逻辑复杂，不建议用 `lambda`。

---

### 2.15 `with`

上下文管理器写法：

```python
with torch.no_grad():
    ...
```

或者：

```python
with open(filename, "rb") as f:
    data = f.read()
```

理解成：

- 进入某种特殊状态
- 执行块内代码
- 自动收尾

最常见两类：

1. 文件读写
2. PyTorch 关闭梯度

---

### 2.16 f-string

作业里有很多：

```python
f"W{i}"
f"model-{milestone}.pt"
f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
```

这是 Python 最常用的字符串插值方式。

规则：

- 在字符串前加 `f`
- 用 `{}` 把变量嵌进去

---

## 3. Python 容器和常见模式

---

### 3.1 列表 `list`

常见用法：

```python
xs = []
xs.append(X)
Xtr = np.concatenate(xs)
```

或者：

```python
h = []
for t in range(T):
    h.append(next_h)
h = torch.stack(h, dim=1)
```

作业里列表常用来：

- 暂存每层 cache
- 暂存每个时间步 hidden state
- 暂存所有 batch 的结果

---

### 3.2 字典 `dict`

字典是作业里最重要的 Python 容器之一。

例如：

```python
self.params["W1"] = ...
self.params["b1"] = ...
```

或者：

```python
bn_param["running_mean"] = running_mean
```

或者：

```python
model_kwargs = {"text_emb": text_emb, "text": text}
```

你可以把字典看成“带名字的盒子集合”。

常见用途：

- 参数表 `params`
- 配置表 `config`
- 条件输入 `model_kwargs`
- 数据集返回结果

---

### 3.3 `setdefault`

优化器里很常见：

```python
config.setdefault("learning_rate", 1e-2)
```

意思是：

- 如果这个 key 已经存在，就保持原值
- 如果不存在，就插入默认值

这特别适合写“优化器状态初始化”。

---

### 3.4 `get`

```python
v = config.get("velocity", np.zeros_like(w))
```

意思是：

- 如果有 `velocity`，就取它
- 没有就返回默认值

这是写“可选配置项”时最常见的方式。

---

### 3.5 `pop`

```python
self.learning_rate = kwargs.pop("learning_rate", 0.001)
```

作用：

- 把这个 key 的值取出来
- 同时把它从字典里删除

这在读取构造参数时非常常见。

---

## 4. NumPy 速查

如果说 assignment1 和 assignment2 前半部分在训练什么语言能力，那就是：

> 你能不能把数学公式翻译成 NumPy 数组操作。

---

### 4.1 `ndarray` 和 `shape`

NumPy 的核心对象是数组。

例如：

```python
X.shape
```

如果返回：

```python
(500, 3072)
```

意思是：

- 500 个样本
- 每个样本 3072 维

作业里最常见的 shape：

- 图像平铺后：`(N, D)`
- 图像卷积输入：`(N, C, H, W)`
- 序列：`(N, T, D)`
- 分类分数：`(N, C)`

你读代码时，第一件事永远是先读 `shape`。

---

### 4.2 `reshape`

`reshape` 只是“换视角”，不改变元素总数。

例如：

```python
x_row = x.reshape(N, -1)
```

意思是：

- 保留第一维是 `N`
- 剩下所有维自动压平

这在全连接层里非常常见，因为 affine 层输入必须是二维：

$$
(N, d_1, \dots, d_k) \to (N, D)
$$

反向时再 reshape 回去：

```python
dx = dx_row.reshape(x.shape)
```

---

### 4.3 `transpose`

`transpose` 用来调换维度顺序。

例如：

```python
X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
```

意思是：

- 原始数据是 `(N, C, H, W)`
- 转成 `(N, H, W, C)`

又比如：

```python
X_train = X_train.transpose(0, 3, 1, 2)
```

把 `(N, H, W, C)` 变回 `(N, C, H, W)`。

记忆方法：

- `transpose(0, 3, 1, 2)` 表示“新的第 0 维来自旧的第 0 维，新的第 1 维来自旧的第 3 维...”

---

### 4.4 `axis`

这是 NumPy 最关键也最容易混的概念之一。

先记：

- `axis=0`: 沿着“样本方向”压缩
- `axis=1`: 沿着“列或特征方向”压缩

例如：

```python
np.sum(x, axis=0)
```

表示：

- 对每一列求和

而：

```python
np.sum(x, axis=1)
```

表示：

- 对每一行求和

在作业里最典型的例子：

BatchNorm：

```python
sample_mean = np.mean(x, axis=0)
```

因为 BN 是“对每个特征维，在 batch 里算均值方差”。

LayerNorm：

```python
sample_mean = np.mean(x, axis=1, keepdims=True)
```

因为 LN 是“对每个样本内部的特征维算均值方差”。

---

### 4.5 `keepdims=True`

例如：

```python
np.sum(x, axis=1, keepdims=True)
```

如果不用 `keepdims=True`，输出可能是 `(N,)`。

用了之后，输出是 `(N, 1)`。

它的意义不是“更漂亮”，而是：

> 让后面的广播更自然。

例如 softmax：

```python
scores -= np.max(scores, axis=1, keepdims=True)
```

因为每一行都要减去“这一行自己的最大值”，所以最大值必须保持成列向量 `(N,1)`。

---

### 4.6 广播 `broadcasting`

广播是 NumPy 和 PyTorch 里最重要的隐式规则之一。

例子：

```python
scores = X @ W + b
```

如果：

- `X @ W` 的形状是 `(N, C)`
- `b` 的形状是 `(C,)`

那么 `b` 会自动广播成 `(N, C)`。

再比如：

```python
x_hat = (x - mean) / np.sqrt(var + eps)
```

如果：

- `x` 是 `(N, D)`
- `mean` 是 `(D,)`

那么 `mean` 会沿 batch 维自动复制成 `(N, D)`。

广播规则直觉版：

- 从右往左对齐维度
- 维度相同可以广播
- 某一维是 1 也可以广播

---

### 4.7 花式索引

这是读作业代码时必须熟悉的点。

#### 取正确类别概率

```python
correct_class_probs = probs[np.arange(N), y]
```

意思是：

- 第 0 行取 `y[0]` 列
- 第 1 行取 `y[1]` 列
- ...

这在 softmax 损失里是经典写法。

#### 修改正确类别梯度

```python
dscores[np.arange(N), y] -= 1
```

意思是：

- 每个样本在自己的正确类别位置减 1

#### 词嵌入查表

```python
out = W[x]
```

如果：

- `W` 是 `(V, D)`
- `x` 是 `(N, T)`

那么输出就是 `(N, T, D)`。

这其实就是“按索引查表”。

---

### 4.8 布尔掩码

例如：

```python
mask = captions_out != self._null
```

或者：

```python
dx = dout * (x > 0)
```

这里：

- `captions_out != self._null` 产生布尔数组
- `x > 0` 也会产生布尔数组

布尔数组常用于：

- 选出有效元素
- 实现 ReLU backward
- 屏蔽 padding
- 构造 dropout mask

---

### 4.9 `zeros_like` / `ones_like`

```python
dW = np.zeros_like(W)
```

意思是：

- 造一个和 `W` 形状完全一样的全零数组

这在写梯度时非常常见，因为梯度 shape 一定和参数 shape 一致。

---

### 4.10 `concatenate` / `hstack` / `stack`

#### `concatenate`

```python
Xtr = np.concatenate(xs, axis=0)
```

表示沿某个已有维度拼接。

#### `hstack`

```python
feat = np.hstack([hog, color])
```

常用于“把多个一维特征拼成一条长特征”。

#### `stack`

`stack` 和 `concatenate` 不同，它会新建一个维度。

这点在 PyTorch 里也极其重要。

---

### 4.11 `np.random.choice`

这是 mini-batch 采样的经典写法：

```python
batch_idx = np.random.choice(num_train, batch_size, replace=True)
```

意思是：

- 从 `0 ... num_train-1` 随机选 `batch_size` 个索引

`replace=True` 表示有放回采样。

作业里常用它来：

- 采样 mini-batch
- 随机抽训练子集
- 随机抽一个样本图片或文本

---

### 4.12 `np.pad`

卷积里最常见：

```python
x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode="constant")
```

如果 `x` 是 `(N, C, H, W)`，这表示：

- batch 维不补
- channel 维不补
- 高度维两边各补 `pad`
- 宽度维两边各补 `pad`

---

### 4.13 `np.maximum`

ReLU 的本质就是：

```python
out = np.maximum(0, x)
```

逐元素比较，取较大值。

---

### 4.14 数值稳定性的 NumPy 写法

Softmax：

```python
scores -= np.max(scores, axis=1, keepdims=True)
```

相对误差：

```python
np.max(np.abs(x - y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))
```

这些都说明：

> NumPy 代码不只是算公式，还要保护数值稳定。

---

## 5. 从 NumPy 公式到矩阵代码

这部分是最值得反复记的。

---

### 5.1 全连接层

数学：

$$
out = xW + b
$$

NumPy：

```python
x_row = x.reshape(N, -1)
out = x_row.dot(w) + b
```

反向：

```python
db = np.sum(dout, axis=0)
dw = x_row.T.dot(dout)
dx = dout.dot(w.T).reshape(x.shape)
```

---

### 5.2 Softmax

数学：

$$
p = \mathrm{softmax}(scores)
$$

NumPy：

```python
shifted = scores - np.max(scores, axis=1, keepdims=True)
probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
```

---

### 5.3 kNN 距离矩阵

数学：

$$
\|x-y\|^2 = \|x\|^2 + \|y\|^2 - 2x^\top y
$$

NumPy：

```python
X_sq = np.sum(X ** 2, axis=1, keepdims=True)
train_sq = np.sum(X_train ** 2, axis=1, keepdims=True).T
cross = X.dot(X_train.T)
dists = np.sqrt(X_sq + train_sq - 2 * cross)
```

这是“把循环公式翻译成向量化矩阵代码”的代表例子。

---

## 6. PyTorch 速查

从 assignment2 后半部分开始，你的代码进入 PyTorch 世界。

PyTorch 最核心的对象是 `Tensor`。

你可以把它暂时理解成：

> 支持 GPU、支持自动求导的 NumPy 数组。

---

### 6.1 Tensor 和 NumPy 的关系

常见转换：

```python
torch.from_numpy(x)
torch.tensor(x)
torch.as_tensor(x)
```

区别直觉版：

- `torch.from_numpy(x)`: 从 NumPy 共享数据创建 Tensor
- `torch.tensor(x)`: 通常会复制，最稳妥
- `torch.as_tensor(x)`: 尽量不复制，比较灵活

从 PyTorch 回到 NumPy：

```python
t.detach().cpu().numpy()
```

拆开理解：

- `detach()`: 从计算图里断开
- `cpu()`: 移到 CPU
- `numpy()`: 转成 NumPy

---

### 6.2 dtype

作业里常见：

```python
dtype=torch.float32
dtype=torch.long
```

几个重要 dtype：

- `torch.float32`: 浮点数，网络参数和输入最常用
- `torch.float64`: 更精确，数值检查时可能用
- `torch.long`: 整数索引，embedding 和分类标签常用
- `torch.bool`: mask 常用

记住：

- `Embedding` 的输入 token id 必须是整数类型
- `CrossEntropyLoss` 的标签也必须是整数类别索引

---

### 6.3 device 和 `.to(...)`

最常见：

```python
x = x.to(device)
model = model.to(device)
```

作用：

- 把 Tensor 或模型搬到 CPU / GPU

也可以同时指定 dtype：

```python
features = torch.as_tensor(features, dtype=torch.float32, device=device)
```

---

### 6.4 `nn.Module`

PyTorch 模型的标准写法：

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(x)
```

你在作业里会看到很多：

- `class PositionalEncoding(nn.Module)`
- `class MultiHeadAttention(nn.Module)`
- `class Unet(nn.Module)`
- `class CaptioningTransformer(nn.Module)`

记住：

- `__init__` 里定义层
- `forward` 里写数据流
- 调 `model(x)` 时，本质上就是执行 `forward`

---

### 6.5 常见层

这三份作业里最常见的 PyTorch 层：

```python
nn.Linear(...)
nn.Embedding(...)
nn.Conv2d(...)
nn.Dropout(...)
nn.LayerNorm(...)
nn.Sequential(...)
nn.GELU()
nn.ReLU()
```

理解方式：

- `nn.Linear(in_dim, out_dim)`: 全连接层
- `nn.Embedding(V, D)`: 词表查表层
- `nn.Conv2d(in_ch, out_ch, kernel_size)`: 卷积层
- `nn.Dropout(p)`: dropout
- `nn.LayerNorm(dim)`: layer normalization
- `nn.Sequential(a, b, c)`: 把多个层按顺序串起来

---

### 6.6 `forward` 不是手动调用的唯一入口

你虽然定义的是：

```python
def forward(self, x):
    ...
```

但实际调用一般写：

```python
out = model(x)
```

不是：

```python
out = model.forward(x)
```

因为 `model(x)` 会自动触发 PyTorch 的模块机制。

---

### 6.7 Autograd：自动求导

PyTorch 最大的好处之一，就是不用你手写 backward。

训练循环：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

意义分别是：

- `zero_grad()`: 清掉上一步残留梯度
- `backward()`: 自动反向传播
- `step()`: 更新参数

这是整个 PyTorch 训练系统最核心的三行。

---

### 6.8 `torch.no_grad()` 和 `@torch.no_grad()`

推理时常见：

```python
with torch.no_grad():
    output = model(x)
```

或者：

```python
@torch.no_grad()
def sample(...):
    ...
```

作用：

- 不构建计算图
- 节省显存
- 推理更快

适用场景：

- 验证
- 采样
- CLIP 编码图库
- 生成 caption

---

### 6.9 `model.train()` 和 `model.eval()`

这也是必须记住的两行。

```python
model.train()
model.eval()
```

它们会影响：

- Dropout
- BatchNorm

`train()`:

- dropout 生效
- batchnorm 用当前 batch 统计量

`eval()`:

- dropout 关闭
- batchnorm 用 running statistics

---

### 6.10 `state_dict()`

PyTorch 保存参数的标准方式：

```python
model.state_dict()
optimizer.state_dict()
```

然后保存：

```python
torch.save({
    "model": model.state_dict(),
    "opt": optimizer.state_dict(),
}, path)
```

这在 diffusion trainer 里就有体现。

---

### 6.11 `register_buffer`

Transformer 和 diffusion 里很重要：

```python
self.register_buffer("pe", pe)
```

或者：

```python
self.register_buffer("betas", betas.float())
```

缓冲区的意思是：

- 这不是可训练参数
- 但它属于模型状态的一部分
- 会跟着 `.to(device)` 一起搬运
- 会被 `state_dict()` 保存

典型用途：

- 位置编码
- diffusion 的系数表

---

## 7. PyTorch 张量操作

这部分是 PyTorch 版的 shape 操作速查。

---

### 7.1 `view` 和 `reshape`

例如：

```python
q = q.view(N, S, n_head, head_dim)
```

或者：

```python
x_flat = x.reshape(N * T, V)
```

一般直觉：

- `view` 更像“改变视图”
- `reshape` 更通用

如果前面做过 `transpose/permute`，有时必须：

```python
output = output.transpose(1, 2).contiguous().view(N, S, E)
```

为什么要 `contiguous()`？

因为某些维度重排后内存不连续，`view` 需要连续内存。

---

### 7.2 `transpose` 和 `permute`

PyTorch 里：

```python
x.transpose(1, 2)
```

表示交换两个维度。

而：

```python
x.permute(0, 2, 4, 1, 3, 5)
```

表示按任意顺序重排所有维度。

在 Transformer 和 ViT 里很常见。

例如 patch 切分：

```python
patches = x.reshape(N, C, H // P, P, W // P, P)
patches = patches.permute(0, 2, 4, 1, 3, 5)
```

---

### 7.3 `unsqueeze` 和 `squeeze`

`unsqueeze(dim)` 表示插入一个长度为 1 的维度。

例如：

```python
memory = visual_projection(features).unsqueeze(1)
```

如果原来是 `(N, D)`，现在变成 `(N, 1, D)`。

这在 Transformer 里表示“插入一个 sequence 维”。

---

### 7.4 `torch.stack`

如果你有一个列表，里面每个 Tensor shape 一样：

```python
h = [h1, h2, h3]
torch.stack(h, dim=1)
```

会新建一个维度。

例子：

- 每个 `h_t` 是 `(N, H)`
- `torch.stack(h, dim=1)` 后是 `(N, T, H)`

这在 RNN 中最典型。

---

### 7.5 `torch.cat`

`cat` 是沿已有维度拼接：

```python
out = torch.cat([out_left, out_right], dim=0)
```

在 SimCLR 中：

- `out_left`: `(N, D)`
- `out_right`: `(N, D)`
- 拼完后：`(2N, D)`

在 U-Net 中：

```python
x = torch.cat([x, skips.pop()], dim=1)
```

表示沿通道维拼接 skip connection。

---

### 7.6 `torch.argmax`

分类最常见：

```python
pred = torch.argmax(logits, dim=1)
```

表示在类别维度上取最大值对应的索引。

在 captioning 里：

```python
word = torch.argmax(output_logits, axis=1)
```

表示选出当前时间步概率最高的词。

---

### 7.7 `torch.mean` / `torch.sum`

用法和 NumPy 类似。

例如：

```python
x = torch.mean(x, dim=1)
```

ViT 里表示对 patch token 做 mean pooling。

---

### 7.8 布尔 mask

Transformer 里：

```python
tgt_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
```

这里涉及几个关键点：

- `torch.tril(...)`: 下三角矩阵
- `dtype=torch.bool`: 布尔 mask
- `~mask`: 取反
- `masked_fill`: 把不合法位置填成 `-inf`

为什么填 `-inf`？

因为 softmax 后这些位置的概率就会变成 0。

---

### 7.9 `torch.randn` / `randn_like` / `randint`

扩散模型中最常见：

```python
noise = torch.randn_like(x_start)
t = torch.randint(0, nts, (b,), device=x_start.device).long()
```

作用：

- `randn_like(x)`: 和 `x` 同 shape 的高斯噪声
- `randint(low, high, shape)`: 随机整数

---

## 8. PyTorch 训练循环模板

这是你以后最值得背下来的模板之一。

---

### 8.1 标准训练模板

```python
model.train()

for x, y in data_loader:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
```

---

### 8.2 验证模板

```python
model.eval()

with torch.no_grad():
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
```

---

### 8.3 生成 / 采样模板

```python
model.eval()

with torch.no_grad():
    sample = model.sample(...)
```

---

## 9. `Dataset` 和 `DataLoader`

---

### 9.1 `Dataset`

如果一个类继承了 `Dataset`，最关键的是两个函数：

```python
def __len__(self):
    return len(self.data)

def __getitem__(self, idx):
    return sample
```

这意味着：

- `__len__`: 数据集长度
- `__getitem__`: 第 `idx` 个样本怎么取

作业里例子：

- `CIFAR10Pair`
- `EmojiDataset`

---

### 9.2 `DataLoader`

```python
loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

作用：

- 自动按 batch 取数据
- 自动打乱
- 自动迭代

训练时常见：

```python
for data, target in loader:
    ...
```

---

### 9.3 `default_collate`

在 `EmojiDataset.random_model_kwargs` 里你会看到：

```python
imgs, model_kwargs = torch.utils.data.default_collate(samples)
```

作用：

- 把一堆单样本拼成 batch

如果每个样本是：

```python
(img, {"text_emb": emb, "text": text})
```

它会自动帮你把图像和字典字段批量拼起来。

---

## 10. Matplotlib 速查

作业 notebook 里非常常见。

---

### 10.1 最常见的画图方式

画曲线：

```python
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training loss")
plt.show()
```

画图像：

```python
plt.imshow(img)
plt.axis("off")
plt.title("Sample")
plt.show()
```

---

### 10.2 `figure` 和 `subplots`

```python
plt.figure(figsize=(6, 6))
```

表示新建一个画布。

更常用的是：

```python
fig, axes = plt.subplots(1, nh, figsize=(3 * nh, 3))
```

意思是：

- 创建一个图
- 再创建多个子图坐标轴

---

### 10.3 图像显示时的 shape

Matplotlib 显示彩色图像通常要求：

$$
(H, W, C)
$$

而 PyTorch 图像常是：

$$
(C, H, W)
$$

所以 notebook 里经常会看到：

```python
plt.imshow(np.transpose(npimg, (1, 2, 0)))
```

或者：

```python
grid_img = grid_img.permute(1, 2, 0).cpu().numpy()
```

这就是把通道维挪到最后。

---

### 10.4 关闭坐标轴

```python
plt.axis("off")
```

这是图像展示里非常常见的写法，因为看图片本身比看坐标更重要。

---

## 11. PIL、OpenCV、TensorFlow Datasets、CLIP、tqdm

这些不是“语言本身”，但在作业里出现很多，值得单独记一下。

---

### 11.1 PIL：`Image.fromarray`

常见：

```python
img = Image.fromarray(img_array)
```

作用：

- 把 NumPy 图像数组转成 PIL Image
- 方便接 torchvision transform

---

### 11.2 OpenCV：`cv2.resize`

常见：

```python
m = cv2.resize(m, (60, 60), cv2.INTER_NEAREST)
```

这里注意：

- OpenCV 的尺寸顺序是 `(W, H)`，不是 `(H, W)`
- 分割 mask 用 `INTER_NEAREST`
- 连续图像可以用双线性之类插值

---

### 11.3 `tfds.load`

在 DINO 分割里：

```python
tfds.load("davis/480p", split="validation", as_supervised=False)
```

说明：

- 这是 TensorFlow Datasets
- 用来读取数据集

---

### 11.4 `clip.load` 和 `clip.tokenize`

CLIP 里：

```python
model, preprocess = clip.load("ViT-B/32", device=device)
tokens = clip.tokenize(texts)
```

记住：

- `load` 返回模型和预处理函数
- `tokenize` 把字符串变成 token id

---

### 11.5 `tqdm`

训练进度条：

```python
for data in tqdm(loader):
    ...
```

它不会影响算法，只是让进度更可视化。

---

## 12. 代码阅读时最常见的 shape 变换

这部分最适合你以后反复看。

---

### 12.1 全连接层

输入：

$$
(N, d_1, d_2, ..., d_k)
$$

先压平：

$$
(N, D)
$$

再线性：

$$
(N, M)
$$

---

### 12.2 卷积层

图像：

$$
(N, C, H, W)
$$

卷积输出：

$$
(N, F, H', W')
$$

池化后：

$$
(N, F, H'/2, W'/2)
$$

---

### 12.3 RNN

词 id：

$$
(N, T)
$$

词嵌入后：

$$
(N, T, W)
$$

RNN hidden：

$$
(N, T, H)
$$

词表分数：

$$
(N, T, V)
$$

---

### 12.4 Transformer

token embedding：

$$
(N, T, D)
$$

多头切分：

$$
(N, h, T, d_h)
$$

拼回：

$$
(N, T, D)
$$

---

### 12.5 ViT patchify

输入图像：

$$
(N, C, H, W)
$$

切 patch 后：

$$
(N, num\_patches, P^2 C)
$$

投影后：

$$
(N, num\_patches, D)
$$

---

### 12.6 Diffusion

图像：

$$
(N, C, H, W)
$$

时间步：

$$
(N,)
$$

扩散系数经 `extract(...)` 后：

$$
(N, 1, 1, 1)
$$

这样才能广播到整张图像。

---

## 13. 你在作业里最该重点记住的“语言习惯”

这部分是浓缩版。

### 13.1 先看 shape，再看公式

很多代码一眼看不懂，不是因为公式难，而是因为不知道张量在几维。

### 13.2 看 `axis` 和 `dim`

NumPy 用 `axis`
PyTorch 用 `dim`

本质一样，都是：

> 沿哪个维度做操作

### 13.3 看 `reshape / transpose / permute`

很多复杂代码不是在“算新东西”，只是在“换维度视角”。

### 13.4 看 mask 的语义

尤其是：

- caption 的 padding mask
- Transformer 的 causal mask
- dropout mask

### 13.5 看 train/test 切换

很多 bug 来自：

- 训练和推理模式没切
- `no_grad` 没加
- BN / Dropout 行为不一致

### 13.6 看有没有原地修改

例如：

```python
w -= lr * dw
config["cache"] *= rho
```

这些都会直接改原对象。

### 13.7 看是拼接 `cat/concatenate` 还是新建维度 `stack`

这是 shape bug 的高发点。

### 13.8 看是否在“查表”

例如：

- `W[x]`
- `probs[np.arange(N), y]`

这类花式索引常常比公式还重要。

---

## 14. 最后给你的学习建议

如果你现在对 Python / NumPy / PyTorch 还不够熟，最有效的方式不是单独背语法，而是按下面顺序练：

1. 先把每个函数输入输出的 `shape` 写出来
2. 再把里面每一行代码的 `shape` 写出来
3. 再问“这行代码是算新值，还是换维度，还是做索引”
4. 最后再问“这对应数学公式的哪一部分”

可以把代码里的每一行都分成下面四类：

- 取数据
- 换 shape
- 做数学运算
- 记缓存 / 组织流程

一旦你开始这样读代码，复杂模型会突然变得清楚很多。

最后压缩成一句话：

> 你现在最需要练的，不是背更多库函数，而是把“shape、索引、广播、数据流”这四件事练到下意识。

