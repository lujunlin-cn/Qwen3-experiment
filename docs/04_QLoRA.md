# QLoRA 指导：面向 RTX 4090 24GB 的参数高效训练

版本：v0.1 \| 对应 Notebook：`03_SFT` 内的 QLoRA 分支

## 1. 定位

QLoRA 不是独立训练算法，而是 SFT / DPO / PPO / GRPO
中可使用的低显存参数更新方式。首版只在 SFT Notebook 中完成 LoRA / QLoRA
对照。

不使用 Unsloth。优先使用 Transformers、PEFT、bitsandbytes 和
PyTorch。复用 Datawhale 官方 QLoRA 环境 Profile。

开发验证 GPU 为 RTX 5090；教程需要评估 RTX 4090 24GB。不能把 5090
的实测结果写成 4090 的实测结果。

## 2. 基本机制

QLoRA 的组成：

1.  将冻结的基座模型权重量化到 4-bit。
2.  使用 NF4 表示量化权重。
3.  可启用 Double Quantization 压缩量化常数。
4.  使用 BF16 等计算 dtype 执行前向计算。
5.  冻结量化基座权重。
6.  仅训练 LoRA Adapter。

QLoRA 不会将激活、Adapter、梯度和优化器状态全部变成 4-bit。

## 3. 为什么需要 QLoRA

Qwen3-4B 的 BF16 基座权重会占用数 GB 显存。训练还需要激活、Adapter
梯度和优化器状态。4090 24GB 可能需要 QLoRA
才能容纳目标序列长度和训练配置，但不能仅凭显存容量推断一定可运行。

## 4. 对照实验

  实验   基座精度             Adapter   目标
  ------ -------------------- --------- ------------------------
  L0     BF16                 LoRA      精度与速度参考
  Q0     NF4 4-bit            LoRA      低显存参考
  Q1     NF4 + Double Quant   LoRA      检查 Double Quant 影响

控制相同数据、随机种子、LoRA Rank / Alpha、最大序列长度、Batch /
Gradient Accumulation、学习率和训练 Token Budget。

报告峰值显存、Tokens/s、训练耗时、Loss、验证准确率、Adapter
大小、保存和重新加载结果。

## 5. 首轮配置原则

-   Per-device Batch Size 从 1 开始。
-   开启 Gradient Checkpointing。
-   使用 BF16 Compute（硬件支持时）。
-   LoRA Rank 从较小值开始，再做 Rank 对照。
-   首轮关闭 Packing，先保证样本边界和标签 Mask。
-   限制 Max Sequence Length。
-   先运行短序列 Smoke Test，再运行完整轨迹。

具体数值由 RTX 5090 实测确定，不直接照搬 Nemotron 30B 选手的配置。

## 6. 与 vLLM 的兼容性

训练阶段使用 QLoRA，不代表推理阶段也必须加载 4-bit
基座。预期部署方式：训练时使用量化基座 + LoRA；推理时使用 BF16 基座 +
LoRA + vLLM。

必须验证 Adapter 能否在 BF16 基座上正确加载，并比较 Transformers 与 vLLM
的输出：

1.  Adapter 保存。
2.  Adapter 重新加载。
3.  BF16 基座加载 Adapter。
4.  vLLM 加载 Adapter。
5.  同一 Prompt 的答案一致性。
6.  Thinking 输出和终止 Token。
7.  每次 RL 更新后 Adapter 的同步。

如果不兼容，先定位 PEFT 配置、Target Modules
和权重命名差异，不要默认通过合并权重解决。

## 7. 环境原则

-   复用 Datawhale 官方 QLoRA 环境 Profile。
-   不在 Notebook 中反复 `pip install`。
-   不使用 Unsloth。
-   vLLM 与训练环境分离。
-   同一 GPU 同步串行运行。
-   记录 PyTorch、CUDA、Transformers、PEFT、bitsandbytes 版本。

## 8. 比赛经验的迁移边界

原比赛模型是 Mamba-2 / Transformer / MoE
混合架构。部分选手在该模型上遇到 LoRA
Projection、权重合并和量化兼容问题。这些问题不能直接推广到 Qwen3-4B
Dense Transformer。

第 5 名报告曾尝试 Unsloth 4-bit 训练但未取得预期效果，作者将其与当时
Nemotron 架构和实现环境联系起来。该结果不能证明 QLoRA 在 Qwen3-4B
上无效。

本项目必须以 RTX 5090 的实际测试为准。

## 9. 交付物

BF16 LoRA 与 QLoRA 配置、峰值显存和吞吐对照、Adapter 保存 /
加载测试、vLLM Adapter 兼容测试、RTX 4090 24GB
可运行性结论，以及未通过测试的配置和错误记录。

## 10. 参考资料

-   [Datawhale QLoRA
    Notebook](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb)
-   [Datawhale LeetCode
    仓库](https://github.com/datawhalechina/llm-algo-leetcode)
-   [第二名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution)
-   [第五名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/domdolus-tolus-solution)
