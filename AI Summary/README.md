# CS231n 高阶学习方向总索引

这组笔记不是再去重复某一道题怎么写，而是从三份作业里往上抽象，提炼出更长期有价值的学习能力。

如果说前面的三份总文档更像：

- `cs231n_study_summary.md`：按算法和方法总结
- `cs231n_code_language_notes.md`：按 Python / NumPy / PyTorch 语法与写法总结
- `cs231n_tensor_shape_guide.md`：按张量维度与形状变化总结

那么这一组文档更像“站在整个课程之上再回头看”，重点回答下面这些问题：

- 为什么这些作业会这样安排？
- 这些作业真正训练的是哪些可迁移的能力？
- 以后学新模型、新论文、新框架时，哪些能力还能继续复用？

## 阅读顺序建议

如果你想按“最有迁移性”的顺序读，建议：

1. `01_tensor_shape_thinking.md`
2. `02_unified_training_framework.md`
3. `04_numerical_stability_and_debugging.md`
4. `05_formula_to_code_translation.md`
5. `03_model_blocks_as_modules.md`
6. `06_representation_learning_mainline.md`
7. `07_generative_vs_discriminative.md`
8. `08_experiment_design_and_research_habits.md`
9. `09_code_abstraction_and_system_design.md`
10. `10_bridge_from_cs231n_to_modern_foundation_models.md`

## 文档清单

### 1. 张量维度思维

文件：[01_tensor_shape_thinking.md](./01_tensor_shape_thinking.md)

关键词：

- `shape`
- `reshape / view / permute`
- `batch / channel / time / token / head`
- “先看维度，再看公式”

### 2. 统一训练框架思维

文件：[02_unified_training_framework.md](./02_unified_training_framework.md)

关键词：

- `data -> model -> loss -> backward -> update -> eval`
- `Solver / Trainer`
- 训练态与测试态

### 3. 常见模型块的积木化理解

文件：[03_model_blocks_as_modules.md](./03_model_blocks_as_modules.md)

关键词：

- `Linear / Conv / Norm / Activation / Attention / Residual`
- “新模型 = 旧模块的新组合”

### 4. 数值稳定性与调试能力

文件：[04_numerical_stability_and_debugging.md](./04_numerical_stability_and_debugging.md)

关键词：

- `softmax overflow`
- `NaN / Inf`
- 梯度爆炸 / 消失
- 调试顺序

### 5. 从数学公式到代码的翻译能力

文件：[05_formula_to_code_translation.md](./05_formula_to_code_translation.md)

关键词：

- 数学对象识别
- 轴与求和维度
- 广播、mask、索引
- 公式落地成矩阵代码

### 6. 表示学习的主线

文件：[06_representation_learning_mainline.md](./06_representation_learning_mainline.md)

关键词：

- supervised representation
- contrastive learning
- multimodal alignment
- self-supervised features

### 7. 生成式模型与判别式模型的统一视角

文件：[07_generative_vs_discriminative.md](./07_generative_vs_discriminative.md)

关键词：

- \(p(y \mid x)\)
- \(p(x)\)
- \(p(x \mid c)\)
- 分类、描述、检索、生成

### 8. 实验设计与科研习惯

文件：[08_experiment_design_and_research_habits.md](./08_experiment_design_and_research_habits.md)

关键词：

- baseline
- ablation
- error analysis
- reproducibility

### 9. 代码抽象能力与系统设计

文件：[09_code_abstraction_and_system_design.md](./09_code_abstraction_and_system_design.md)

关键词：

- 接口设计
- 模块边界
- 参数管理
- 可维护性

### 10. 从 CS231n 作业到现代基础模型的桥梁

文件：[10_bridge_from_cs231n_to_modern_foundation_models.md](./10_bridge_from_cs231n_to_modern_foundation_models.md)

关键词：

- CNN -> Transformer
- SimCLR / CLIP / DINO / Diffusion
- foundation model
- 迁移学习路径

## 怎么使用这组文档

有三种很实用的用法：

### 用法一：复盘式学习

每看完一份作业，就回到这组文档里对应的一两篇，问自己：

- 这道题训练的到底是哪一层能力？
- 如果以后换一个模型，我还能迁移哪部分能力？
- 这部分知识在大模型时代有没有过时？

### 用法二：读代码前热身

在读一份新代码前，先看：

- 看不懂 shape，就读 `01`
- 看不懂训练流程，就读 `02`
- 看不懂模块组成，就读 `03`
- 看不懂公式如何落地，就读 `05`
- 跑着跑着炸了，就读 `04`

### 用法三：构建自己的“元知识”

课程里的每个模型都只是载体。真正值得长期保存的，是下面这些“元知识”：

- 怎么看 shape
- 怎么拆模块
- 怎么从公式写代码
- 怎么做实验
- 怎么调试
- 怎么抽象出可复用框架

这组文档的目的，就是帮助你把这些能力从“做过作业”升级成“以后能反复复用”。

## 一句总总结

这 10 个方向合起来，其实是在回答一个问题：

> 三份 CS231n 作业，除了让我会做题，到底把我训练成了什么样的人？

如果你把这 10 份文档读透了，你收获的不只是“会写某个算法”，而是“以后再学新模型时，我知道该从哪里下手”。
