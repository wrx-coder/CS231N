# 03. 常见模型块的积木化理解

## 为什么这一层理解特别重要

很多模型第一次看会觉得名字很吓人：

- ResNet
- U-Net
- Transformer
- CLIP
- ViT
- DDPM

但如果你真正做过 CS231n 的三份作业，就会发现一个很关键的事实：

> 新模型通常不是从零发明一切，而是把一组熟悉的基本模块重新组合起来。

也就是说，现代模型的阅读方式，不应该是“背模型名”，而应该是“拆积木”。

这份笔记就是把这些积木整理出来，帮助你以后看新模型时先拆模块，再理解整体。

## 一、什么叫“模型块”

模型块是一个比“单层”稍大、但比“完整模型”更小的功能单元。

它通常满足：

- 有明确输入输出
- 有稳定功能
- 可以重复堆叠
- 在很多不同模型中复用

例如：

- `Linear`
- `Conv`
- `Norm`
- `Activation`
- `Pooling`
- `Embedding`
- `Attention`
- `Residual Block`
- `MLP Block`
- `Projection Head`

一旦你用模块视角去看问题，大模型会容易很多。

## 二、三份作业里最核心的 10 种模型块

## 1. Linear / Affine Block

数学形式：

\[
out = xW + b
\]

功能：

- 改变特征维度
- 混合特征
- 做分类头或投影头

典型出现位置：

- Softmax 分类器
- 两层网络
- RNN 中输入投影和输出投影
- Transformer 中 Q/K/V 投影、输出投影、MLP
- diffusion 中条件投影

你应该形成的直觉：

- 它几乎总是作用在“最后一个特征维”
- 它不关心空间和时间本身，只关心每个位置上的特征变换

## 2. Activation Block

最常见是 ReLU：

\[
\mathrm{ReLU}(x) = \max(0, x)
\]

功能：

- 引入非线性
- 让网络能表达复杂函数

assignment1 和 assignment2 里你已经看到：

- 没有非线性，多层线性叠加仍然等价于一层线性

所以 `Linear + ReLU` 是最基本的积木组合。

## 3. Normalization Block

包括：

- BatchNorm
- LayerNorm

功能：

- 稳定中间表示分布
- 改善优化

在 assignment2 和 assignment3 里，归一化已经是基础设施级别的模块。

你要学会区分：

- BN 更依赖 batch 统计
- LN 更适合序列和 Transformer

## 4. Convolution Block

数学直觉：

- 局部感受野
- 权重共享
- 空间结构保留

典型结构是：

\[
\text{Conv} \to \text{Norm?} \to \text{ReLU}
\]

在 CNN、U-Net 中极其重要。

你要把它理解成：

- 在空间邻域上做局部模式提取
- 比全连接更有图像归纳偏置

## 5. Pooling / Downsample Block

功能：

- 减少空间分辨率
- 增大感受野
- 保留更粗粒度语义

assignment2 的 max-pooling 和 assignment3 的 U-Net 下采样，属于同一类思路。

## 6. Embedding Block

功能：

- 把离散对象映射到连续向量空间

典型对象：

- 词 token
- 类别标签
- 时间步
- patch

公式：

\[
e_i = W_{\text{embed}}[i]
\]

你在 captioning、Transformer、diffusion 时间嵌入里都见过它。

## 7. Attention Block

这是 assignment3 最重要的新积木之一。

公式：

\[
\mathrm{Attn}(Q, K, V)
=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

功能：

- 让一个位置动态读取其他位置的信息
- 建立 token 之间的关系图

你要形成的高层直觉：

- 卷积更像固定局部模板
- attention 更像数据依赖的动态路由

## 8. Residual Block

形式：

\[
y = x + F(x)
\]

功能：

- 保留原信息通路
- 稳定深层训练
- 让模块学习“增量修正”

Transformer block 和 ResNet / U-Net block 的思想都离不开 residual 连接。

## 9. MLP / Feed-Forward Block

Transformer 中的 FFN，本质还是位置独立的 MLP：

\[
\mathrm{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2
\]

它通常和 attention 配合：

- attention 负责跨位置交互
- FFN 负责每个位置内的特征变换

## 10. Projection Head / Output Head

功能：

- 把 backbone 表示映射到任务空间

例如：

- 分类头：映射到类别 logits
- SimCLR projection head：映射到对比空间
- CLIP 编码器头：映射到对齐空间
- diffusion 输出头：映射到噪声预测

这一层提醒你：

> backbone 学表示，head 决定任务接口。

## 三、如何把整个模型拆成模块图

## 1. MLP

\[
\text{Linear} \to \text{ReLU} \to \text{Linear}
\]

这是最基础模块链。

## 2. CNN

\[
\text{Conv} \to \text{ReLU} \to \text{Pool}
\to
\text{Linear} \to \text{ReLU} \to \text{Linear}
\]

## 3. RNN Captioning

\[
\text{Image Feature Projection}
\to
\text{Word Embedding}
\to
\text{RNN Cell}
\to
\text{Temporal Affine}
\]

## 4. Transformer Decoder

\[
\text{Token Embedding}
\to
\text{Positional Encoding}
\to
\left[
\text{Masked Self-Attn}
\to
\text{Cross-Attn}
\to
\text{FFN}
\right] \times L
\to
\text{Vocab Head}
\]

## 5. ViT

\[
\text{Patch Embedding}
\to
\text{Positional Encoding}
\to
\left[
\text{Self-Attn}
\to
\text{FFN}
\right] \times L
\to
\text{Classifier Head}
\]

## 6. DDPM U-Net

\[
\text{Input Conv}
\to
\text{Down Blocks}
\to
\text{Middle Blocks}
\to
\text{Up Blocks}
\to
\text{Output Conv}
\]

其中每一层 block 又会融合：

- 时间嵌入
- 条件嵌入
- 卷积块
- 残差连接

## 四、你应该如何看待“新模型”

以后当你遇到一个陌生模型名时，可以先做下面这件事：

### 第一步：它的输入是什么类型

- 图像？
- 序列？
- 图文对？
- 噪声图像？

### 第二步：主干积木是什么

- Conv 主导？
- Attention 主导？
- MLP 主导？
- 混合型？

### 第三步：它用哪些辅助模块稳定训练

- residual
- normalization
- dropout
- skip connection
- positional encoding
- time embedding

### 第四步：它的 head 和 loss 是什么

一旦这四步写出来，大多数模型已经不再陌生。

## 五、CS231n 的一个巨大价值：你已经见过这些积木的原型

这点很重要，因为很多人看现代模型会误以为它们完全脱离基础课。

其实不是。

### assignment1 给你的原型

- affine
- ReLU
- softmax loss
- 参数字典
- forward/backward 分层结构

### assignment2 给你的原型

- conv
- batch norm / layer norm
- dropout
- 序列模型
- 模块组合

### assignment3 给你的原型

- attention
- positional encoding
- projection head
- cross-modal alignment
- diffusion denoising block

所以现代模型看似复杂，其实只是把这些原型堆得更深、接得更巧。

## 六、怎么把“模块意识”练成熟

### 1. 每读一个模型，都先列模块清单

例如：

- 输入编码模块
- 主干模块
- 融合模块
- 输出头
- loss 模块

### 2. 用一句话概括每个模块的功能

例如：

- `Embedding`：把离散符号变成连续向量
- `Attention`：让 token 间动态交换信息
- `Residual`：保留主通路，学习增量修正
- `Projection Head`：把通用表示映射到具体任务空间

### 3. 学会分辨“主干模块”和“训练辅助模块”

主干模块：

- conv
- attention
- MLP

训练辅助模块：

- normalization
- dropout
- residual
- skip connection

这种区分很有用，因为它能帮助你知道模型真正的计算核心是什么。

## 七、从“背名字”转向“看结构”

成熟的阅读方式应该是：

- 不被模型名字带着走
- 看到结构就知道它为什么能工作
- 看到模块组合就能猜出它适合什么任务

比如：

- 如果模型主要靠卷积，多半更强调局部空间结构
- 如果模型主要靠 attention，多半更强调全局关系建模
- 如果模型加 projection head，多半训练目标和最终评估空间不完全一样
- 如果模型有 encoder + decoder，多半是条件生成或映射任务

## 八、给自己留的一套“拆模型模板”

以后读任何模型，都可以先填这张表：

| 维度 | 要回答的问题 |
| --- | --- |
| 输入编码 | 原始输入如何转成可计算表示？ |
| 主干块 | 主要计算靠哪些模块完成？ |
| 信息交互 | 是局部交互、递归交互还是全局注意力？ |
| 辅助稳定 | 用了哪些 norm / residual / dropout？ |
| 输出头 | 最终要输出什么对象？ |
| loss | 训练目标如何定义？ |

## 最后一句总结

如果你把模型看成“积木系统”而不是“神秘大黑盒”，很多复杂方法都会突然变得平易近人。

真正重要的不是记住多少模型名，而是看到一个新架构时，能立刻拆出：

- 它用了哪些熟悉模块
- 这些模块各自做什么
- 它们为什么要这样组合
