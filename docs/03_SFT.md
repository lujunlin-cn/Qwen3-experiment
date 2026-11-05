# SFT 指导：从可验证轨迹到可学习的推理行为

版本：v0.1 \| 对应 Notebook：`03_SFT`。LoRA / QLoRA 另见 `04_QLoRA.md`。

## 1. 目标

将课程提供的人类先验转化为模型可以记忆、检索和执行的监督信号。重点不是堆叠大量
CoT，而是构造正确、可从 Prompt
推导、格式稳定、长度可控且模型能够学习的训练轨迹。

## 2. 选手经验与可迁移结论

### 第一名：合成问题 + Solver Trace + 两阶段 SFT

第一名使用规则 Solver
生成合成问题和轨迹，并将训练拆成记忆与执行两个阶段。其复现仓库展示了大规模数据和长时间训练。

**迁移：**
研究可复用知识记忆与完整问题执行是否应分开训练，不复制其数据规模。

### 第三名：Memorization → Execution

第一阶段使用 Recall Drill 学习 Signature /
候选映射；第二阶段学习在完整轨迹中调用记忆并执行求解。

**迁移：** 对 Equation Numeric 和 Bit Manipulation
设计规则记忆与规则执行对照。

### 第九名：先修订 CoT，再增加训练量

第九名没有额外扩展合成问题，而是重点修订
CoT，使其更一致、更容易人工审计。

**迁移：** 先检查教师轨迹质量，再决定是否增加数据。

### 第十五名：Targeted Drill

将容易失败的局部操作拆成 Drill，再混入完整任务训练。

**迁移：** Drill 应由错误分析驱动，不应无差别重复所有基础规则。

### 第 251 名：每一行都必须可复现

其 Trace Format Engineering
要求每行只陈述当前可验证的操作，不能提前断言尚未证明的结论；同时过滤超过长度预算的轨迹。

**迁移：** Trace 是训练目标，不是 Solver 日志。Solver 正确不等于 Trace
可学习。

### 第二名：复杂训练技巧未必优于标准 CE

作者尝试 Focal Loss、Token Reweighting
和多阶段训练，但没有发现它们稳定超过标准交叉熵。

**迁移：** 先建立标准 CE 基线。复杂 Loss 只能作为独立消融。

### 第十名：禁止 Answer Leakage

教师 Solver 必须从 Prompt 自身推导答案，不能读取 Ground Truth
后再反向构造推理过程。第八十名也报告过 Equation Numeric CoT 中的
discovery leakage。

**迁移：** Trace 必须
Prompt-derivable，只使用题目已提供的信息和课程公开先验。

## 3. 监督样本格式

建议字段：`prompt`、`reasoning`、`final_answer`、`task_type`、`trace_type`、`source`、`source_id`、`token_length`、`verification_status`。

训练时使用模型 Chat Template。Prompt Token 不参与 SFT Loss，Assistant
Completion Token 参与 Loss。必须检查 `input_ids` 与 `labels`
的边界，避免静默错位。

## 4. SFT 实验设计

  实验    方法                     目标
  ------- ------------------------ ----------------------------------
  SFT-A   Direct Full-Trace        建立基础 SFT 结果
  SFT-B   Joint SFT                完整轨迹与 Recall Drill 混合训练
  SFT-C   Two-stage SFT            先 Memorization，再 Execution
  SFT-D   Targeted Drill（选做）   针对错误模式修复局部能力

SFT-B / SFT-C 必须控制训练 Token Budget。若两阶段使用更多 Token
或更多优化步数，必须单独报告，不能把训练量增加误认为训练顺序收益。

## 5. 轨迹质量审计

每条轨迹至少检查：最终答案正确；每一步可从
Prompt、公开先验和前序步骤推出；没有引用 Ground Truth
中才存在的信息；没有跳过关键验证；变量已定义；格式一致；长度在预算内。

推荐记录： - **Teacher Solvability**：参考 Solver 能否解决该题。 -
**Trace Fidelity**：模型是否忠实复现预期算法步骤。

Teacher Solvability 高不代表模型一定能学会 Trace。

## 6. 训练实现

-   PyTorch 自定义训练循环。
-   Transformers 加载 Qwen3-4B。
-   PEFT LoRA / QLoRA。
-   标准 Cross Entropy 作为首个基线。
-   Gradient Checkpointing 控制激活显存。
-   首轮关闭 Packing，先验证样本边界和 Label Mask。
-   保存 Adapter、配置、随机种子和训练日志。

不要使用高层 SFT Trainer 替代本教程需要展示的 Loss 与训练循环。

## 7. 评测指标

按任务报告 Accuracy、Macro Accuracy、Answer Extraction Success、Trace
Verification Rate、Teacher Solvability、平均 Completion
Tokens、训练耗时、峰值显存和训练前后能力退化。

Equation Numeric 分开报告 Deduce / Guess。Bit Manipulation
分开报告基础操作、组合深度和结构 OOD。

## 8. 主要风险

错误教师轨迹会污染监督信号；只看最终答案会掩盖错误推理；Token Budget
不一致会破坏消融公平性；过度重复 Drill 可能损害其他任务；只看 Loss
不能判断推理准确率；验证集过小会产生统计波动。

## 9. 交付物

Direct SFT 基线、Joint / Two-stage 对照、可选 Drill
数据和结果、轨迹质量审计、分任务评测、最佳 SFT Checkpoint。

## 10. 参考资料

-   [第一名复现仓库](https://github.com/xrwr/kaggle-nvidia-nemotron-model-reasoning-challenge-1st-place)
-   [第三名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution)
-   [第四名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution)
-   [第九名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/9th-place-solution-human-auditable-cot)
-   [第十五名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/wip-lb15-extension-to-huikangs-progress-prize)
-   [第 251
    名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/trace-format-engineering-and-svd-downsampled-lora)
-   [第二名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution)
-   [第十名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/10th-place-solution)
