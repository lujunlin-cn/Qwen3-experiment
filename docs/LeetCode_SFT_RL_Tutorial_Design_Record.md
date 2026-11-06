> **最新明确决策（2026-11-06，任务替换轮）**：默认正式任务调整为 Numeral System（Easy）、Gravity（Medium）、Equation Numeric（Hard）。Bit 只保留高级扩展，不进入新版数据及 Baseline。维护者工具预先清洗、分类与筛选官方 CSV；00 只读取已清洗数据，完成分层划分、泄漏审计与基线并写入独立版本 `tasksets/roman_gravity_equation_v1/`；01 合成工程完全可选。所有三类均计划参加 SFT；SFT 后分别判断 RL 必要性，本轮不更改 RL 入口、算法或 Reward，不修改 03–06 的模型/LoRA/QLoRA 配置。Validation 用于预算校准，Holdout 只做固定审计和未来最终评估。后文与此冲突的旧任务集合及路径视为历史记录。真实 Baseline 证据见 [修改报告](../NOTEBOOK_REFACTOR_REPORT.md)。

> 架构修订（2026-11-06）：原 Baseline Notebook 拆为 `00_Baseline_Diagnosis.ipynb` 和完全可选的 `01_Synthetic_Data_Engineering.ipynb`，02–06 名称编号不变。00 使用维护者已清洗官方数据，负责 80/10/10 划分、四类泄漏审计与 Validation Baseline；03 默认 Real Only，显式 `USE_SYNTHETIC_DATA=1` 才追加通过 Verifier 和三集合泄漏审计的 Accepted。原官方 CSV 清洗代码保留在 `utils/prepare_official_dataset.py`。以下六阶段路线是历史设计记录，当前文件边界见 [文档索引](README_文档索引.md)。本轮不实施教学练习改造。

# LeetCode 式 SFT & RL 课程大作业：项目设计总记录

> 文档用途：保存项目讨论中已经确认的决定、当前建议、技术假设、待验证事项、Kaggle
> 经验和后续工作安排。\
> 文档版本：v0.2\
> 记录日期：2026-11-05\
> 项目状态：核心任务、模型、课程定位、官方仓库集成方式和环境管理原则已确认；vLLM、QLoRA 与完整 RL 闭环仍需在目标 GPU 上验证。\
> 重要说明：本文严格区分“已确认”“当前建议”“待验证”和“未决定”。技术兼容性、显存和耗时没有实测前，不得写成已验证事实。2026-11-05 的最新决策覆盖本文早期与之冲突的建议。

------------------------------------------------------------------------

## 0. 如何使用本文档

> **最新决策优先级（2026-11-05）**：本项目将合并进官方 Datawhale 仓库；只使用 Gravity、Equation Numeric、Bit Manipulation；课程提供人类先验；数据工程可选；直接复用官方环境；不使用 Unsloth；4090 优先验证 QLoRA；vLLM 与训练环境分离并在同一 GPU 上同步串行运行。若后文早期内容与上述决定冲突，以本版本更新后的对应章节为准。


本文是项目讨论的长期记录，不是已经完成的技术设计说明书，也不是经过实测的复现报告。

阅读时遵循以下标记：

  标记                    含义
  ----------------------- ------------------------------------------------------
  **已确认**              讨论中用户明确表达并认可的决定
  **当前建议**            当前推荐方案，但用户尚未明确最终批准，或仍可调整
  **待验证**              有合理技术依据，但尚未在目标硬件和固定软件版本上实测
  **未决定**              需要后续讨论或实验后才能决定
  **已放弃 / 暂不采用**   讨论后不再作为当前主线的方案

后续更新本文时，应优先更新对应章节的状态，不要只在文末追加零散结论。

### 0.1 项目总目标

设计一套面向大模型后训练学习者的课程大作业。课程以推理任务为统一实验对象，以
SFT 和 RL 为核心技术路线。

学生需要完成从原始模型评测、Prompt
Engineering、监督微调、偏好优化到强化学习的实验，并通过统一评测判断不同方法是否带来真实收益。

课程应同时满足三个要求：

1.  有一条可以跟随完成的参考实验路线。
2.  有清晰的参考代码、测试和实验验收标准。
3.  为学生保留自主设计实验和提出不同结论的空间。

### 0.2 一句话项目定位

**带有参考答案、强调可复现实验、鼓励自主延拓的推理模型后训练课程大作业。**

项目不是单纯介绍 SFT、DPO、PPO、GRPO 的公式，也不是追求复现 Kaggle
排行榜名次。项目希望训练学生根据任务能力缺口、监督数据质量、奖励信号和计算预算，选择合适的后训练方法。

------------------------------------------------------------------------

# 第一部分：已确认的项目决定

## 1. 教学定位与课程形式

### 1.1 已确认：采用 LeetCode 式课程大作业

用户明确希望课程采用类似 `llm-algo-leetcode` 的学习体验。

这意味着 Notebook
不应只是完整代码展示，也不应只是理论讲义。学生需要理解问题、完成
TODO、运行测试、查看参考答案，再分析实验结果。

课程需要提供参考答案，但不能把学生限制在唯一的实验结论上。

建议每个 Notebook 保留以下结构：

1.  本节导读：说明学习目标和核心问题。
2.  运行环境：列出依赖、硬件和运行方式。
3.  前置阅读：介绍必要概念、公式和背景。
4.  Step 1--N：拆分任务和实现过程。
5.  TODO：让学生完成关键代码。
6.  测试：检查实现是否正确。
7.  参考代码与解析：提供可复现的参考实现。
8.  可选 GPU 实验：运行真实模型或训练任务。
9.  思考题：解释实验现象。
10. 延拓实验：鼓励学生修改数据、算法或超参数。

### 1.2 已确认：遵循 Datawhale 现有 Notebook 风格

**已确认：本项目不是独立教程仓库，而是准备合并进官方 `datawhalechina/llm-algo-leetcode`。**

官方仓库：<https://github.com/datawhalechina/llm-algo-leetcode>

开发和测试必须遵循官方仓库现有的 Notebook 组织、命名、环境 Profile、Quickstart、结果记录和贡献规范。不要另建一套平行仓库或独立运行时。

已经检查的官方教学模式包括：

- `13_SFT_Training_Loop.ipynb`：监督边界、Label Masking、SFT Loss 和训练循环。
- `14_LoRA_Tutorial.ipynb`：手写 `LoRALinear`，再与 PEFT 对照。
- `16_RLHF_PPO_Memory.ipynb`：拆解 PPO 的 Actor、Critic、Ratio 和 Clipping。
- `17_DPO_Loss_Tutorial.ipynb`：手写 Response Log Probability 和 DPO Loss。
- `18_GRPO_Loss_Tutorial.ipynb`：手写 Group Advantage、Ratio、KL 和 Clipping。
- `84_DPO_Preference_Project.ipynb`、`85_GRPO_Groupwise_Alignment_Project.ipynb`：先做 CPU 机制检查，再进入受控 GPU 项目实验。

本项目应延续官方模式：**原理讲解 → TODO 实现 → 单元测试 → 参考实现 → 框架/数值核对 → 真实 GPU 实验 → 结果分析**。

### 1.2.1 已确认：项目直接合并进官方仓库

- 不创建独立 GitHub 教程仓库。
- 不创建独立的 Notebook 目录体系；按官方 `02_PyTorch_Algorithms/` 和相关专题的现有结构安排文件。
- 不维护独立的 `requirements.txt`、`constraints.txt`、安装器、`setup.sh` 或 Preflight 工具。
- 不制作 Docker 镜像。
- 环境安装和依赖版本直接复用官方 Environment Profile 与 Quickstart。
- 如果项目确实需要官方现有 Profile 未覆盖的依赖组合，应通过官方仓库贡献修改 Profile，而不是让学生在 Notebook 中临时拼装环境。

### 1.2.2 已确认：算法由学生实现，底层基础设施复用官方生态

Students implement the key calculations for SFT, DPO, PPO, and GRPO in PyTorch. GSPO is outside the first six Notebooks.

官方高层 Trainer 不作为本项目的核心教学实现。成熟框架可以用于可选的结果核对，但不能替代学生对算法核心的实现。

### 1.3 已确认：课程必须兼顾标准答案和开放延拓

教学任务分成三个层次：

  ----------------------------------------------------------------------------
  层次                                学生任务
  ----------------------------------- ----------------------------------------
  基础必做                            按 Notebook
                                      完成规定实验，运行测试，生成标准结果

  思考分析                            比较不同方法，解释性能、成本和失败原因

  自主延拓                            改变 Prompt、数据、LoRA 配置、Reward 或
                                      RL 算法，提出自己的实验
  ----------------------------------------------------------------------------

参考实现负责保证课程可完成；开放问题负责训练学生的研究判断能力。

## 2. 任务基准、任务范围与人类先验

### 2.1 已确认：以 NVIDIA Nemotron Model Reasoning Challenge 为任务来源

比赛链接：

<https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge>

本项目参考比赛公开训练数据、题型和优秀选手的 Solver / SFT / RL 经验，但**目标不是复现排行榜名次**。目标是构建一个可在消费级单卡 GPU 上完成的后训练教学实验。

### 2.2 已确认：第一版只保留三个题型

| 任务 | 教学定位 | 主要研究能力 | RL 定位 |
|---|---|---|---|
| Gravity | 简单计算控制组 | 公式识别、数值计算、精度控制 | 通常 SFT 已足够；不进入 RL 主训练，用于观察基础能力是否退化 |
| Equation Numeric | 中等难度核心任务 | 从输入输出示例识别隐藏运算规则，并应用于 Query | 主要 RL 实验之一 |
| Bit Manipulation | 困难任务 | 识别并组合位移、旋转和布尔操作 | 主要 RL 实验之一 |

其他比赛题型（例如 Cipher、Cryptarithm、Numeral、Unit Conversion 等）不进入第一版必做路线。可以在后续扩展，不应增加首版课程的必要复杂度。

Equation Numeric 需要区分：

- **Deduce**：Query 使用的运算符已经在示例中出现，模型需要识别已展示的规则。
- **Guess**：Query 使用的运算符没有在示例中出现，需要根据任务规则空间和已有约束作出推断。

第一版先保证 Deduce 主线可运行，再把 Guess 作为进阶实验。评测必须分别统计两类结果。

### 2.3 已确认：人类先验由课程统一提供

学生不需要花时间重新逆向发现隐藏规则，也不以复现第一名的规则发现过程为必做任务。课程提供统一的 Prior Specification，明确规则空间、数学定义、约束和表达格式；具体题目的隐藏规则仍然需要模型根据示例推断。

建议提供的先验资产：

| 任务 | 课程提供的先验 | 学生需要学习的内容 |
|---|---|---|
| Gravity | 物理公式、单位约定、数值精度和合法参数范围 | 从示例识别参数并执行计算 |
| Equation Numeric | 候选运算规则库、正向/反向结构定义、全局约束 | 规则记忆、候选检索、约束筛选和 Query 求解 |
| Bit Manipulation | SHL、SHR、ROT、AND、OR、XOR、NOT 等操作定义，以及有限 Boolean DSL / 表达式语法 | 规则召回、表达式验证、组合推理和执行 |

**课程提供规则空间，不提供每道题的具体隐藏规则或最终答案。**

### 2.4 已确认：课程研究重点是先验知识的学习、检索和运用

项目主题可表述为：**Prior-Guided Reasoning Post-Training**。

重点研究：

1. 如何将课程提供的规则先验表示为结构化知识。
2. 如何通过 Prompt 将先验放入上下文。
3. 如何通过 Recall Drill / SFT 将规则知识内化到模型参数。
4. 如何让模型根据题目召回候选规则。
5. 如何将候选规则用于约束检查和完整推理。
6. 如何用 DPO / PPO / GRPO 改善已有能力。

课程不把 Solver 逆向工程作为学生必做任务。参考 Solver 和 Verifier 由课程提供或提供参考实现，学生重点实现数据构造、模型训练和评测机制。

### 2.5 已确认：默认真实数据由课程提前筛选和整理

官方训练数据整体较干净。课程维护者提前过滤其他题型，只保留 Gravity、Equation Numeric、Bit Manipulation，并提供统一字段格式和默认数据划分。学生不需要重复完成基础清洗。

默认路线直接使用课程准备好的三任务数据。数据划分必须在训练前固定，并检查精确重复、同源变体和规则实例跨集合问题。

### 2.6 已确认：数据工程是可选分支，不是必做前置

学生可以选择三种路径：

- **直接跳过**：直接下载 Kaggle 官方公开训练数据，或使用课程准备好的三任务筛选版本；按课程提供的最小过滤方式保留三个题型，然后进入 Baseline / Prompt / SFT，不生成合成数据。
- **只做 Leakage Audit**：检查训练、验证、测试之间的重复样本、近重复、同源题目和生成来源交叉。
- **完整数据工程拓展**：在 Leakage Audit 基础上，尝试 Synthetic Data Generation、Verifier 筛查、歧义检查、难度审计和分布比较。

合成数据不是必需条件。当前没有证据表明 Qwen3-4B-Thinking-2507 必须依赖大规模合成数据才能完成课程。需要通过 Real-only 与 Real + Synthetic 对照实验判断合成数据是否有收益。

合成数据筛查至少包括 Schema Validation、Semantic Verification、Answer Verification、Identifiability、Deduplication 和 Difficulty Audit。必须记录拒绝原因和不同难度的接受率，避免 Solver Rejection Sampling 系统性丢弃困难样本。

### 2.7 数据划分和泄漏控制

- 训练集、验证集和测试集先划分，再进行合成数据生成。
- 验证集和测试集不参与训练数据生成器、规则库或 Solver 的拟合。
- 同一道原始题目的增强版本必须与原题位于同一数据分区。
- OOD 测试优先按规则组合或结构家族隔离，而不是仅随机切分。
- 不将 Kaggle 隐藏测试数据或未经授权的数据放入公开仓库。

### 2.8 LeetCode 式教学，不等于必须使用代码生成题

LeetCode 式描述的是课程组织方式：题目驱动、明确输入输出、分步骤实现、TODO、测试、参考实现和开放延拓。Nemotron 比赛提供的是可验证推理任务素材。第一版不引入代码执行沙箱。

## 3. 基座模型

### 3.1 已确认：选择 Qwen3-4B-Thinking-2507

模型 ID：

`Qwen/Qwen3-4B-Thinking-2507`

官方模型卡：

https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507

选择原因：

1.  参数规模在 5B 以下。
2.  属于 Dense Transformer 路线，避免本项目处理 Mamba-2 等特殊架构适配。
3.  适合研究数学、逻辑和代码推理。
4.  学生可以直接评测模型已有推理能力。
5.  可以在标准 Transformers 生态中开展模型加载和训练适配。
6.  有利于使用 PEFT、vLLM 等通用工具。

此前讨论过的其他候选：

  ----------------------------------------------------------------------------------------------
  模型                                      讨论结论
  ----------------------------------------- ----------------------------------------------------
  `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`   暂不作为主线。其混合 Mamba/Attention 架构带来额外
                                            LoRA 和推理适配工作

  `Qwen3-4B-Instruct-2507`                  不作为当前主线。用户希望保留显式推理能力

  `Qwen3-4B-Thinking-2507`                  当前正式选定

  `Qwen2.5-Coder-3B-Instruct`               不优先。任务更偏代码生成，且许可证条件需要单独核对
  ----------------------------------------------------------------------------------------------

### 3.2 Thinking Mode 的处理

已确认采用 Thinking 模型，不将关闭 Thinking 作为主线实验。

模型应使用官方 Chat Template 和推荐生成配置。

需要注意：

-   不应随意删除或改写思考标记。
-   训练数据需要明确思考部分和最终答案的边界。
-   不应假设每条输出都一定显式出现 `<think>`。
-   应记录模型实际生成的 Token IDs。
-   应测试截断是否导致推理轨迹不完整。
-   生成长度需要设置上限，避免长推理消耗过多时间和显存。

Thinking
输出预算是待验证的工程参数，不应直接照搬高难度评测中的超长生成配置。

## 4. 硬件与运行时间

### 4.1 已确认：面向单卡 24--32GB 显存

目标硬件：

  GPU          显存 定位
  ---------- ------ --------------
  RTX 4090     24GB 最低目标硬件
  RTX 5090     32GB 推荐硬件

不以 PRO 6000、H100 或多卡集群作为课程必要条件。

### 4.2 已确认：总时间目标约 6 小时

用户希望学生不需要花十几个小时才能完成大作业。

6 小时是课程设计目标，不是已经验证的实测结果。

需要明确区分：

-   环境准备时间；
-   模型下载时间；
-   数据准备时间；
-   GPU 实验时间；
-   Notebook 阅读和思考时间。

建议将核心 GPU 实验的 6
小时目标定义为：环境和模型权重已经准备好后，学生能够完成核心实验链路。

如果将首次下载模型、安装所有依赖和全部选做实验也纳入 6
小时，必须先测量再决定是否可行。

### 4.3 已确认：训练和推理共用同一张 GPU

不要求学生租两张 GPU。

训练和推理必须共卡。

### 4.4 已确认：采用同步执行，不做异步 Rollout

不建设异步 Actor-Learner 系统，不引入复杂的分布式任务队列。

推荐的初始执行逻辑：

`Rollout → 保存轨迹 → 释放推理引擎 → 训练更新 → 保存策略 → 重新加载推理引擎`

这种方式可以降低系统复杂度，并减少策略版本不一致的问题。

代价是每轮权重导出、合并和推理引擎重载可能增加耗时。该代价必须纳入实验预算。

## 5. 环境与交付：直接复用官方 Datawhale 体系

### 5.1 已确认：不制作 Docker 镜像

不把 Dockerfile、预构建镜像或镜像仓库作为第一版交付。

### 5.2 已确认：项目合并进官方仓库，环境也使用官方环境

学生和开发者均使用 `datawhalechina/llm-algo-leetcode` 现有的环境管理体系。环境配置由官方仓库维护，学生按对应 Quickstart 创建环境、注册 Kernel 并执行预检。

本项目不创建独立依赖清单、安装器、`setup.sh`、Preflight 或环境 Profile。Notebook 不应在执行过程中临时 `pip install` 一组未经验证的包。

### 5.3 当前可复用的官方 Environment Profile

| 官方 Profile | 用途 | 本项目使用方式 |
|---|---|---|
| `PT-CPU` | 后训练 CPU 机制练习和框架核对 | Loss、Verifier、数据处理和单元测试 |
| `PT-SFT-CUDA` | 单卡 SFT / LoRA | BF16 LoRA 训练路径 |
| `Q-QLORA` | 单卡 NF4 QLoRA | 4090 上优先验证的低显存训练路径 |
| `PT-RL-CUDA` | DPO、GRPO 和奖励信号机制实验 | 自研 RL 训练循环；该 Profile 当前禁止安装 vLLM |
| 官方 vLLM Profile | 独立推理 / Rollout 环境 | Baseline 和 RL Rollout；具体 Profile 由 Qwen3 模型兼容性测试决定 |

官方环境索引：<https://github.com/datawhalechina/llm-algo-leetcode/blob/main/docs/gpu_environment_assets.md>

官方后训练 Quickstart：<https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/post_training_optimization/quickstart.md>

### 5.4 vLLM 与训练环境隔离

官方 `PT-RL-CUDA` 当前禁止安装 vLLM。不得为了方便将 vLLM 强行装入该环境。采用同一张 GPU 的同步串行流程：

1. 激活官方 vLLM Profile，启动 vLLM 并生成 Rollout。
2. 保存 Token IDs、Log Probability、策略版本和采样配置。
3. 关闭 vLLM，释放 GPU 显存。
4. 切换到官方训练 Profile，执行 SFT / DPO / PPO / GRPO 更新。
5. 保存 Adapter / Checkpoint。
6. 重新激活 vLLM Profile，加载更新后的策略并进入下一轮。

训练和推理共用一张 GPU，但不同时驻留。该流程是同步、串行执行，不建设异步 Rollout。

### 5.5 QLoRA 依赖与 RL 环境

`Q-QLORA` 是官方 NF4 QLoRA 环境。4090 24GB 上优先验证 QLoRA；BF16 LoRA 作为可选对照。QLoRA 训练后的 Adapter 能否直接被所选 vLLM Profile 加载，必须实测。

如果 DPO / PPO / GRPO 的训练 Profile 缺少 QLoRA 所需依赖组合，应先检查官方 `Q-QLORA` 与 `PT-RL-CUDA` 的包兼容性，再通过官方仓库贡献 Profile 变更。不要让学生在 Notebook 内临时混装依赖。

### 5.6 学生的环境准备责任

学生负责准备 AutoDL / 本地单卡 GPU 资源，并按官方 Quickstart 创建或选择环境。课程负责给出准确的 Profile 入口、模型和数据资产说明，以及 Notebook 级环境检查。依赖版本冲突由官方 Profile 管理，不要求学生逐包排障。

# 第二部分：当前推荐但尚未全部最终确认的工程方案

## 6. 推理引擎选择

### 6.1 当前主线：vLLM

使用 vLLM 负责 Baseline 推理和 RL Rollout。主要原因是多次采样任务需要较高生成吞吐，避免用 Transformers 逐条生成造成过长运行时间。

SGLang 不作为第一版必学内容。除非 vLLM 在目标模型或官方环境中存在无法解决的兼容问题，否则不增加第二套推理后端。

### 6.2 官方环境隔离

vLLM 使用官方仓库已有的独立 vLLM Profile。具体 Profile 需要根据 Qwen3-4B-Thinking-2507 的支持情况和官方锁定版本确定。不要在 `PT-RL-CUDA` 中安装 vLLM。

### 6.3 待验证的兼容性

在 RTX 4090 24GB 上验证：

1. Qwen3-4B-Thinking-2507 加载和 Chat Template。
2. Thinking 输出和终止标记。
3. Token IDs、生成 Log Probability 的获取。
4. LoRA / QLoRA Adapter 的加载和更新后重载。
5. vLLM 与 Transformers 对相同 Token 序列的 Log Probability 一致性。
6. 峰值显存、吞吐和单轮 Rollout 时间。

QLoRA Adapter 是否能直接加载到 vLLM 尚未验证。如果不兼容，需要评估官方支持的 Adapter 导出、合并或转换路径。

### 6.4 Rollout 记录

至少保存：

- `prompt_ids`
- `completion_ids`
- attention mask 和 completion mask
- EOS / truncation 状态
- 任务 ID 和 Prompt ID
- 源策略版本 / Adapter 版本
- Verifier 版本
- Temperature、Top-p、生成上限
- behavior policy log probabilities（算法需要时）
- Reward 和各奖励分量

不能只保存解码后的文本。

## 7. 训练框架选择

### 7.1 已确认：不使用 Unsloth

用户已明确决定不将 Unsloth 纳入本项目。后续不以 Unsloth Trainer 或其专用接口作为教学路径。

### 7.2 已确认：借鉴 MiniMind 的轻量实现方式，但不直接依赖 MiniMind

MiniMind 只作为代码组织和教学思路参考。项目使用 Qwen3-4B-Thinking-2507，不直接套用 MiniMind 自身模型实现。

### 7.3 训练技术栈

- Transformers：模型和 Tokenizer 加载。
- PyTorch：Loss、反向传播、优化器和训练循环。
- PEFT：LoRA Adapter 管理。
- bitsandbytes：NF4 QLoRA（通过官方 `Q-QLORA` 环境）。
- vLLM：推理和 Rollout（使用官方独立 Profile）。
- Python Verifier：确定性答案和可验证过程检查。

### 7.4 学生需要实现的算法核心

| 模块 | 教学实现 |
|---|---|
| SFT | Label Masking、Response Log Probability、SFT Loss 和训练循环 |
| LoRA | 复用官方 PEFT；可选手写 `LoRALinear` 机制练习 |
| QLoRA | 理解 NF4 / 量化旁路，底层量化 Kernel 复用官方依赖 |
| DPO | Response Log Probability、Reference Policy、DPO Loss |
| PPO | Actor / Critic、Old Log Probability、GAE、PPO-Clip、Value Loss |
| GRPO | Group Reward、Group Advantage、Importance Ratio、Clipping |
| GSPO | Not included in the first six Notebooks; possible future extension |
| Reward / Verifier | 任务答案验证和可确定性检查的中间步骤奖励 |

不自行实现模型架构、Tokenizer、CUDA Kernel、量化 Kernel 或推理引擎。

### 7.5 框架核对

核心教学代码由学生实现。可参考 Datawhale 现有 Notebook 的做法，用成熟框架或小型数值测试核对关键公式，但不得用高层 Trainer 取代核心 TODO。

# 第三部分：教学路线与实验设计

## 8. Overall route: six Notebooks (confirmed)

The first release contains exactly six Notebooks:

1. **Baseline Evaluation** — includes an optional Data Engineering branch. Baseline evaluation is required; Leakage Audit and Synthetic Data Generation are optional.
2. **Prompt Engineering** — prompt comparisons and inference-time reasoning strategies.
3. **SFT** — Direct / Joint / Two-stage SFT. LoRA and QLoRA comparisons are included in this Notebook, not a separate Notebook.
4. **DPO** — Rollout, Verifier, Chosen / Rejected construction, DPO Loss, and offline optimization.
5. **PPO** — Actor-Critic, GAE, PPO-Clip, Value Loss, and the training loop.
6. **GRPO** — Group Rollout, Verifier Reward, Group Advantage, Policy Ratio, Clipping, and training analysis.

Data Engineering is an optional branch inside Notebook 1. LoRA / QLoRA is a branch inside Notebook 3. RL Readiness Audit is embedded in Notebooks 4–6 and does not occupy a separate Notebook. Final cross-algorithm analysis is included in the GRPO Notebook summary and the course report.

GSPO is outside the first-release scope. It does not receive a standalone Notebook or required training experiment.

Gravity is the basic capability control task and is mainly used in Baseline, Prompt Engineering, and SFT. It is not forced into RL. Equation Numeric and Bit Manipulation are the main RL tasks.

## 9. Stage 0：Baseline Evaluation

### 9.1 目标

加载 `Qwen/Qwen3-4B-Thinking-2507`，使用固定 Prompt、Chat Template 和解码配置测量原始模型在三个任务上的能力。

### 9.2 实验内容

1. 读取课程准备的三任务数据。
2. 使用官方 Chat Template。
3. 生成 Thinking Trace 和最终答案。
4. 提取答案并运行确定性 Verifier。
5. 统计 Overall / Per-domain Accuracy。
6. 保存失败样本、生成长度、延迟和截断率。

### 9.3 主要指标

- Exact Accuracy / Pass@1
- Per-domain Accuracy
- Equation Numeric 的 Deduce / Guess Accuracy
- Bit Manipulation 的表达式和答案验证结果
- Average Output Tokens、Latency、Truncation Rate
- OOD Accuracy（如果提供独立 OOD 划分）

## 10. Stage 1：Prompt Engineering

### 10.1 目标

在不更新模型参数的条件下，研究提示词能否改善规则识别、知识调用和最终答案。

### 10.2 建议对照

- P0：统一基础 Prompt。
- P1：明确任务定义和输出格式。
- P2：要求先列出候选规则，再逐条验证。
- P3：将规则先验摘要放入上下文（In-context Prior）。
- P4：Few-shot 示例。
- P5：要求最终答案前执行确定性自检。

Thinking Mode 保持开启。记录准确率变化和额外 Token 成本。

## 11. 可选分支：Data Engineering

### 11.1 三种参与路径

| 路径 | 学生工作 | 是否必做 |
|---|---|---|
| 路径 A | 直接使用课程准备好的三任务数据 | 默认 / 可跳过数据工程 |
| 路径 B | 只完成 Leakage Audit | 选做 |
| 路径 C | Leakage Audit + Synthetic Data Generation + Screening | 进阶选做 |

### 11.2 Synthetic Data 对照

建议比较：

- Real Data Only。
- Real + Synthetic。
- Real + Quality-screened Synthetic。

控制训练 Token Budget、模型和训练配置，比较 IID、OOD 和分任务准确率。合成数据规模不预设为大规模；先判断 4B 模型是否需要额外数据。

### 11.3 数据筛查要求

- Schema Validation。
- Semantic / Answer Verification。
- Identifiability：示例能否确定 Query 答案。
- Exact / Near-duplicate 检查。
- Difficulty 和规则分布审计。
- Rejection Sampling 接受率统计。

生成器和 Verifier 尽量采用独立实现。不得用验证集或测试集答案构造合成训练数据。

## 12. Stage 2：SFT

### 12.1 目标

研究模型如何学习课程提供的人类先验，并将先验用于规则检索和完整推理。

### 12.2 Prior Specification

课程提供：

- Gravity 物理公式和单位 / 精度约定。
- Equation Numeric 候选运算规则库和全局约束。
- Bit Manipulation 操作定义、Boolean DSL、表达式语法和有限模板。

学生不需要发现这些规则，而需要实现规则知识的表示、记忆、召回、候选验证和应用。

### 12.3 建议的 SFT 对照

| 实验 | 训练方式 | 研究问题 |
|---|---|---|
| S0 | Direct Full-trace SFT | 直接学习完整推理轨迹的效果 |
| S1 | Recall Drill Only | 是否学会召回结构化先验 |
| S2 | Joint SFT（Drill + Full Trace 混合） | 同时学习记忆和执行的效果 |
| S3 | Two-stage SFT（先 Memorization，再 Execution） | 分阶段训练是否优于混合训练 |

S2 是重要对照，避免把额外训练量误判为两阶段训练顺序的收益。对比时控制总训练 Token Budget、更新步数和数据来源。

第三名的 Memorization → Execution 是核心参考；第十五名的 Drill 作为局部能力专项监督的参考。不要直接照搬其数十万条 Drill 和数亿 Token 训练规模。

### 12.4 CoT / Trace 质量

训练轨迹必须正确且可学习。保留第四名的 Trace 原则：不隐藏关键计算、步骤局部性、因果一致性、变量引用完整和表达格式统一。可参考第九名进行 Teacher Error / Student Error 诊断。

### 12.5 SFT 评估与选型

比较 Final Accuracy、Recall Exact、规则候选 Recall / Precision、Trace Validity、OOD Accuracy、Token 成本和显存。选出一个最佳 SFT Checkpoint，作为所有 RL 分支的共同初始化。

## 13. PEFT branch inside the SFT Notebook: LoRA / QLoRA

LoRA / QLoRA is part of Notebook 3 (SFT), not a separate Notebook.

### 13.1 目标

比较参数高效训练方式的显存、速度和质量。

### 13.2 当前建议

- RTX 4090 24GB：优先验证 NF4 QLoRA，配合 Gradient Checkpointing 和 Micro Batch = 1。
- RTX 5090 32GB：可额外测试 BF16 LoRA。
- LoRA Rank / Alpha、rsLoRA、DoRA 等作为选做变体，不增加核心必做负担。

QLoRA 是 4090 的推荐路径，但不是已验证结论。QLoRA 不会消除长序列激活显存，仍需实测峰值显存和训练吞吐。

### 13.3 依赖和训练后 Adapter 交接

使用官方 `Q-QLORA` Profile，不使用 Unsloth。验证 QLoRA Adapter 保存、重新加载、合并或导出，以及 vLLM 是否能够加载更新后的策略。该交接链路尚未验证。

## 14. Stage 4：RL 分支与准入诊断

### 14.1 RL Readiness Audit 是必做诊断

在 DPO / PPO / GRPO 训练前，对 Equation Numeric 和 Bit Manipulation 执行多次 Rollout，并使用确定性 Verifier 统计每道题的正确次数。RL Readiness Audit 嵌入三个 RL Notebook，不单独设置 Notebook。

至少统计：

- 0/N 全错组比例。
- N/N 全对组比例。
- 1 到 N-1 次正确的 Mixed-group Ratio。
- Mean Reward Variance。
- 各任务 / 难度子集的有效组比例。

N=8 可作为首轮诊断配置，Temperature 和采样数量需要通过实测校准，不应固定成所有任务通用的最优值。

### 14.2 不人为制造 Reward Variance

不得通过伪造答案、错误标签或人为篡改 Reward 来保证 RL 提分。若组内全部 0 或全部 1：

1. 检查采样温度、生成上限和策略多样性。
2. 适度增加采样数，评估成本收益。
3. 检查是否存在确定性可验证的中间步骤奖励。
4. 若模型完全不会，回到 SFT、Recall Drill 或可信教师轨迹。
5. 若模型已稳定掌握，停止对该任务进行不必要的 RL 更新。

Gravity 通常由 SFT 解决，不进入 RL 主训练。它保留为能力保持对照，学生需要解释为什么不必对 Gravity 进行 RL。

### 14.3 DPO

参考 The Submitter：`SFT → Rollout → Verifier → Chosen / Rejected → Offline DPO`。

- 每题先生成少量候选（例如 4 条）作为起点。
- 使用任务专用 Verifier 判断答案。
- 复现基线可选择最短正确回答作为 Chosen、最短错误回答作为 Rejected。
- 如果模型没有生成正确回答，可使用可信的确定性 Solver / 参考答案作为 Chosen，但必须记录来源。
- 可选研究 Hard Negative 和 Process-level Preference。

DPO 不要求 GRPO 式组内奖励方差，但必须有可信 Chosen / Rejected。记录 Preference Pair Rate、标签噪声和数据覆盖。

### 14.4 标准 PPO

Kaggle 公开作品没有提供足够完整、可直接迁移的标准 PPO 成功实现。本项目自行设计标准 PPO 教学分支，手写：Actor、Critic、Old Policy Log Probability、GAE、PPO-Clip、Value Loss 和必要的 KL 约束。不得将 PPO 改写成 GRPO。

### 14.5 GRPO

参考第 251 名正向实验和第 80 名零方差失败案例。重点研究同一 Prompt 的组内奖励差异、有效组比例、奖励设计和 SFT 能力边界。GRPO 不保证提升所有任务。

### 14.6 GSPO scope

GSPO is not included in the first six Notebooks and has no required training experiment. It may be considered as a later extension.

### 14.7 RL 分支公平比较

DPO、PPO、GRPO 从同一个最佳 SFT Checkpoint 平行出发，不串联训练。报告算法质量、有效学习信号、Rollout 成本、训练耗时、峰值显存和独立评测。

# 第四部分：执行架构与资源控制

## 15. 同步、共卡执行架构

### 15.1 已确认约束

- 训练和推理使用同一张 GPU。
- 不做异步 Rollout。
- vLLM 与训练环境按官方 Profile 隔离。
- 同步、串行执行，不要求两套模型同时驻留 GPU。

### 15.2 推荐执行流程

```text
SFT / 当前策略 Adapter
        |
        v
激活官方 vLLM Profile
        |
        v
同步生成 Rollout
        |
        v
保存 Token IDs / Logprobs / 策略版本
        |
        v
关闭 vLLM，释放 GPU
        |
        v
激活官方训练 Profile（PT-SFT-CUDA / Q-QLORA / PT-RL-CUDA）
        |
        v
Verifier + Loss + 参数更新
        |
        v
保存新 Adapter / Checkpoint
        |
        v
切回官方 vLLM Profile，加载新策略
        |
        v
下一轮 Rollout
```

权重交接、Adapter 转换和 vLLM 重载时间必须计入训练预算。每条轨迹应携带策略版本，防止使用旧策略概率计算新策略更新。

## 16. 显存管理

4090 24GB 是最低目标硬件，5090 32GB 是可选目标。4090 优先测试 NF4 QLoRA、Gradient Checkpointing、Micro Batch = 1 和受控序列长度。QLoRA 不会消除激活显存；长 Thinking Trace 仍可能 OOM。

需要实测：模型权重、KV Cache、训练激活、梯度、优化器状态、Reference Log Probability、Critic 和 CUDA Runtime 的显存占用。不能只按参数量估算是否可运行。

## 17. PPO 的显存设计

PPO 必须保留 Critic / Value Estimation。可研究共享冻结 Backbone、独立 Value Head / LoRA 的方案，但不能把 PPO 偷换成无 Critic 的 GRPO。Reference Policy 可采用顺序计算或其他节省显存方案，前提是概率计算和策略版本正确。

## 18. 6 小时实验预算

6 小时是课程核心路线目标，不是已经实测的结果。时间统计应分别记录：模型下载、环境准备、数据准备、Baseline、SFT、Rollout、训练更新、模型重载和评测。

### 核心必做建议

- Baseline 与 Prompt Engineering。
- 至少一个可复现的 SFT 实验。
- LoRA / QLoRA 中一条默认路径。
- RL Readiness Audit。
- DPO 和至少一条在线 RL 主线（PPO 或 GRPO）。
- 最终统一评测和实验分析。

### 进阶选做

- Synthetic Data Engineering。
- 多种 SFT 训练策略对照。
- LoRA 变体。
- DPO Hard Negative / Process Preference。
- PPO、GRPO 的超参数和 Reward 消融。
- GSPO 后续扩展，不属于首版六个 Notebook。
- 多随机种子和更大 OOD 评测。

六个 Notebook 的完整运行时间必须通过真实硬件计时确认；不得在没有实测时承诺全部能在 6 小时内完成。

# 第五部分：Kaggle 优秀作品阅读与经验总结

## 19. 精读作品清单

### 19.1 SFT 主线：建议完整阅读 6 篇

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  顺序              作品              链接                                                                                                                                   核心内容
  ----------------- ----------------- -------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------
  1                 第一名            https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution                              Synthetic Data、Solver、Reasoning
                                                                                                                                                                             Trace、两阶段 SFT

  2                 第四名            https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution                              Trainable Traces

  3                 第九名            https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/9th-place-solution-human-auditable-cot          Human-Auditable CoT

  4                 第三名            https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution                              Memorization → Execution

  5                 第十五名          https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/wip-lb15-extension-to-huikangs-progress-prize   500K
                                                                                                                                                                             Drills、基础操作记忆、Backtracking

  6                 Public            https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-public-6th-private-place-solution           Validation、数据配比、类别干扰
                    第二、Private                                                                                                                                            
                    第六                                                                                                                                                     
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 19.2 RL 主线：建议完整阅读 4 篇

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  作品                    链接                                                                                                                                     核心内容
  ----------------------- ---------------------------------------------------------------------------------------------------------------------------------------- -----------------------------
  The Submitter           https://github.com/the-submitter/nvidia-nemotron                                                                                         SFT、Rollout、Verifier、DPO

  第 251 名               https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/trace-format-engineering-and-svd-downsampled-lora   GRPO 正向实验与失败边界

  第 80 名                https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/breaking-the-sft-ceiling-on-nemotron-3-nano-30b   GRPO 零方差失败

  Domdolus                https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/domdolus-tolus-solution                           Pseudo-GRPO、DPO 失败、回归
                                                                                                                                                                   SFT
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 19.3 补充阅读

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  作品                    链接                                                                                                                                       建议重点
  ----------------------- ------------------------------------------------------------------------------------------------------------------------------------------ --------------------------
  第二名                  https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution                                  Focal Loss、Token
                                                                                                                                                                     Reweighting、多阶段训练

  第七名                  https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/7th-place-solution-solving-bitmanipulation-via-st   Bit Tokenization、Dynamic
                                                                                                                                                                     Loss Masking

  第十名                  https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/10th-place-solution                                 Solver 性能上限和错误监督

  第十三名                https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/13th-place-solution                                 三阶段 SFT、数据重采样

  第十八名                https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/18th-solution-from-deterministic-solvers-to-learn   实验迭代、合成数据、LoRA
                                                                                                                                                                     Fusion
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 20. Kaggle 经验提炼

### 20.1 第一名：完整 SFT Pipeline

核心链路：

`Synthetic Problem Generation → Rule-based Solver → Reasoning Trace Generation → SFT`

值得学习：

-   逆向分析原始数据；
-   根据规则分布生成合成题目；
-   编写确定性 Solver；
-   自动生成推理轨迹；
-   进行两阶段 LoRA SFT。

教学启发：

SFT
的核心工作不仅是选择训练框架，更是构建高质量、可验证、可学习的监督数据。

### 20.2 第三名：Memorization → Execution

先训练可复用的结构化知识，再训练完整推理执行。

教学启发：

复杂任务可以拆分为基础记忆和组合执行两个阶段。

### 20.3 第四名：Trainable Traces

核心思想：

正确的推理轨迹不一定适合模型学习。

需要关注：

-   是否隐藏计算；
-   是否局部、连续地展示计算；
-   是否保持因果一致；
-   是否引用已经定义的变量；
-   是否使用统一且数量有限的规则。

教学启发：

监督目标的可学习性本身就是数据工程问题。

### 20.4 第九名：Human-Auditable CoT

重点修复原有推理轨迹中的错误和不一致，而不是盲目增加合成数据。

教学启发：

固定数据规模和训练配置，只改变 CoT 质量，可以形成有价值的 SFT 消融实验。

### 20.5 第十五名：500K Drills

把复杂 Cryptarithm 能力拆成基础操作 Drill，例如乘法表记忆，再通过
Backtracking 完成完整求解。

教学启发：

模型的复杂推理失败可能来自某些基础操作不稳定，而不一定是模型整体容量不足。

### 20.6 Public 第二、Private 第六：Validation 与类别干扰

重点研究：

-   验证集设计；
-   分任务 Public LB 探测；
-   数据比例；
-   不同类别之间的训练干扰；
-   增加数据量可能损害其他类别。

教学启发：

必须分任务评测，不能只看整体准确率。

### 20.7 第二名：复杂 Loss 不一定更好

作者尝试 Focal Loss、Token Reweighting
等训练方法，但复杂方案不一定超过标准 Cross Entropy。

教学启发：

算法复杂度不能替代对照实验。

### 20.8 第七名：Bit Tokenization 与 Dynamic Loss Masking

重点研究特殊输入表示和动态监督位置。

教学启发：

输入表示和监督范围可能影响模型学习算法结构的方式。

### 20.9 第十名：Solver Ceiling

如果教师 Solver 存在系统性错误，扩大数据规模可能只会扩大错误监督信号。

教学启发：

数据生成器必须有独立验证。

### 20.10 第十三名：多阶段 SFT

重点研究分阶段学习率、数据重采样和困难类别补充。

教学启发：

训练阶段和数据配比需要通过消融验证，不能直接照搬比赛超参数。

### 20.11 第十八名：实验迭代

作者展示了多个版本的实验演进，包括 Solver 改进、合成数据、答案辅助消歧和
LoRA Fusion。

教学启发：

必须保存实验配置、数据版本、结果和失败原因。

### 20.12 The Submitter：DPO 实践

核心链路：

`SFT → Rollout → Verifier → Chosen / Rejected → DPO`

作者报告 DPO 带来小幅 Public 和 Private LB 提升。

GRPO / GSPO 工程尝试没有获得可用的成功训练结果。

教学启发：

DPO 可以作为离线偏好优化的实践案例，但不能把其结果推广为所有任务都有效。

### 20.13 第 80 名：GRPO 零方差失败

作者报告每题 8 条回答中：

-   66.8% 的组全部正确；
-   33.2% 的组全部错误；
-   没有部分正确的组。

组内奖励缺少差异，导致 GRPO 相对优势信号退化。

教学启发：

RL 需要有区分能力的奖励和足够的探索空间。

### 20.14 第 251 名：GRPO 正向实验

作者报告部分任务的本地验证指标从 SFT 到 SFT + GRPO 有提升。

教学启发：

GRPO 可能适合修复强 SFT
模型仍然存在的部分错误，但不一定能恢复缺乏基础能力的弱模型。

### 20.15 Domdolus：偏好优化失败后回归 SFT

作者尝试 Pseudo-GRPO + DPO，多轮迭代后仍未学会部分简单任务，最终转向 SFT
路线。

教学启发：

模型生成的正确 /
错误回答并不自动构成高质量偏好数据。偏好信号的质量和覆盖范围非常重要。


### 20.16 本项目的选题收敛

综合上述作品后，第一版只选 Gravity、Equation Numeric 和 Bit Manipulation。课程提供人类先验，学生重点学习 Memorization、Retrieval 和 Execution。Cryptarithm 不进入第一版必做范围，避免将大量时间投入组合搜索 Solver 的逆向工程。

### 20.17 RL 经验对本项目的直接约束

- The Submitter：Verifier-guided Offline DPO；偏好对产生率可能很低，Chosen / Rejected 质量比 Rollout 数量更重要。
- 第 251 名：强 SFT 上的 GRPO 可能修正残余错误，但收益依赖任务和 SFT Checkpoint。
- 第 80 名：全对组和全错组都可能没有组内相对奖励信号；先做 RL Readiness Audit，不得假定所有题目都适合 GRPO。
- Gravity：作为 SFT 能力保持控制组，不为覆盖算法而强行进入 RL。

## 21. Kaggle 经验最终形成的原则

1.  数据质量优先于盲目扩大数据量。
2.  正确 CoT 不一定是可学习 CoT。
3.  确定性 Solver 需要独立验证。
4.  复杂训练方法必须通过消融验证。
5.  SFT 需要关注基础能力和类别覆盖。
6.  RL 需要有效的奖励差异。
7.  强 SFT 与弱 SFT 的 RL 收益可能不同。
8.  训练损失和 Reward 不能代替独立任务指标。
9.  Public LB 和 Private LB 可能给出不同结论。
10. 失败实验也应保留，作为方法选择的证据。

上述具体成绩和实验细节来自前期对 Kaggle Writeups、GitHub
仓库和讨论的阅读记录。正式用于公开技术分享前，应重新打开原文核对数字、实验条件和作者表述，避免将作者报告的局部验证结果写成普遍结论。

------------------------------------------------------------------------

# 第六部分：我们自身项目的失败经验与方法论

## 22. 过去项目经验

此前讨论中提到三个案例。

这些案例适合用作技术分享素材，但正式公开之前需要回查原始实验日志、配置和评测记录。

### 22.1 AIC 项目

此前讨论记录：

-   `gif_hard=None` 被处理为全零标签；
-   1,983/2,508 个样本变成全零标签。

需要进一步回查数据预处理代码和标签统计。

方法论：

**训练前必须检查标签语义、缺失值处理和类别分布。**

### 22.2 RL 课程大作业

此前讨论记录：

-   FULL 的整体指标在 step 80 从 0.356 降到 0.293；
-   IID 指标从 0.528 降到 0.411；
-   OOD 指标从 0.011 提高到 0.056。

当时实验同时改变了 Adapter 和训练数据，不能把指标变化直接归因于 RL
算法。

方法论：

**比较算法时必须控制初始化模型、数据、训练预算和评测协议。**

### 22.3 视频高光项目

此前讨论记录：

-   JSON 合规率从 26.7% 提高到 95%；
-   官方 Mr.HiSum 分数仍低于基线。

方法论：

**局部格式指标改善，不等于目标任务质量改善。**

## 23. 经验归纳

这些失败经验与 Kaggle 作品共同支持以下原则：

1.  训练前先检查数据和标签。
2.  先测量模型能力缺口。
3.  先尝试低成本推理时优化。
4.  SFT 优先解决示范、数据和基础能力问题。
5.  只有存在可靠偏好或奖励信号时才进入偏好优化或 RL。
6.  每种方法都必须通过独立评测。
7.  训练成本和推理成本必须进入最终决策。
8.  允许实验得出"复杂算法不如 SFT"的结论。

------------------------------------------------------------------------

# 第七部分：Notebook 与 GitHub 仓库规划

## 24. Six-Notebook delivery plan for the official repository

This project will be merged into datawhalechina/llm-algo-leetcode. It will not create a separate tutorial repository or an independent top-level directory. Exact file numbering and placement follow the official maintainers' directory conventions.

| Order | Notebook | Scope |
|---|---|---|
| 1 | Baseline Evaluation | Original-model baseline on the three tasks; optional Data Leakage Audit and Synthetic Data Engineering |
| 2 | Prompt Engineering | Zero-shot, structured reasoning, in-context prior, few-shot, and self-check comparisons |
| 3 | SFT | Direct / Joint / Two-stage SFT; LoRA / QLoRA comparison included |
| 4 | DPO | Rollout, Verifier, preference-pair construction, handwritten DPO Loss, offline training |
| 5 | PPO | Actor-Critic, Old Policy, GAE, PPO-Clip, Value Loss, training loop |
| 6 | GRPO | Group Rollout, Verifier Reward, Group Advantage, Policy Ratio, Clipping, training analysis |

The following items are embedded and do not create additional Notebooks:

- Data Engineering is an optional branch in Notebook 1.
- LoRA / QLoRA is included in Notebook 3.
- RL Readiness Audit is a shared prerequisite inside Notebooks 4, 5, and 6.
- Final cross-algorithm analysis is included in the GRPO Notebook summary and course report.
- GSPO is not part of the first six-Notebook release.

Each Notebook follows the official pattern: introduction, prerequisites, concept explanation, implementation steps, TODOs, unit tests, reference implementation, GPU experiment, result analysis, and extension questions.

## 25. 代码复用原则

Notebook 负责教学步骤和实验控制。可复用的模型加载、数据处理、Verifier、训练循环辅助函数和结果记录代码，按官方仓库既有 `tools/` / 项目代码组织规范放置。不要另建完整训练框架，也不要把关键算法藏进黑盒。

学生必须能看到并实现关键算法公式对应的代码。重复的工程操作可以封装，但 SFT、DPO、PPO、GRPO 的关键 Loss 和 Advantage 计算应保留 TODO 与测试。GSPO 不属于首版六个 Notebook。

## 26. 环境初始化与依赖管理

完全复用官方 Environment Profile、requirements、constraints、安装器和 Quickstart。学生不自行决定依赖版本，不在 Notebook 中临时安装 vLLM 或其他核心依赖。

当前相关官方 Profile：

- `PT-CPU`：CPU 机制练习和测试。
- `PT-SFT-CUDA`：BF16 LoRA / SFT。
- `Q-QLORA`：NF4 QLoRA。
- `PT-RL-CUDA`：DPO / GRPO 等 RL 训练机制。
- 官方独立 vLLM Profile：推理和 Rollout，具体 ID 根据 Qwen3 模型兼容性选择。

`PT-RL-CUDA` 当前禁止安装 vLLM，因此 Rollout 和训练必须通过不同官方环境串行完成。若当前官方 Profile 不能覆盖 QLoRA + RL 训练依赖组合，应向官方仓库提交 Profile 变更，不得另建独立环境系统。

Unsloth 已明确排除，不进入依赖和教学路径。

# 第八部分：周五汇报与周末技术分享

## 27. 周五：教程大致设计汇报

目标：让老师和团队理解项目目标、教学结构、实验路径和工程可行性。

建议 7 个部分：

  顺序   内容         展示重点
  ------ ------------ -------------------------------------------
  1      项目背景     为什么需要一套 SFT-RL 课程大作业
  2      教学定位     参考答案 + 开放延拓
  3      教学路线     Baseline → Prompt → SFT → RL
  4      模型和硬件   Qwen3-4B-Thinking-2507、4090/5090
  5      工程架构     vLLM、Transformers、PyTorch、自研训练代码
  6      算法实验     SFT、LoRA/QLoRA、DPO、PPO、GRPO
  7      风险和计划   6 小时预算、兼容性验证、Notebook 交付

汇报时不要声称尚未验证的方案已经能够在 4090 上运行。

## 28. 周末：技术分享

目标：展示如何从过去失败经验和 Kaggle 选手经验中形成项目设计。

建议叙事：

1.  我们过去为什么失败？
2.  Kaggle 优秀选手如何解决相似问题？
3.  我们从成功和失败中提炼了什么方法论？
4.  如何把方法论变成课程实验？
5.  我们希望学生最终学会什么？

技术分享不要按排行榜名次逐篇念文章。应按数据质量、推理轨迹、SFT、Reward、RL
和评测等主题组织。

## 29. 技术分享的核心结论

建议最终收敛到：

-   数据和标签质量决定监督信号是否可信。
-   推理轨迹需要正确，也需要可学习。
-   Prompt Engineering 是低成本基线，不应跳过。
-   SFT 是建立任务能力的重要手段。
-   DPO、PPO、GRPO 的训练信号和系统成本不同。
-   RL 不保证超过 SFT。
-   实验结论必须由独立评测和成本记录支持。

------------------------------------------------------------------------

# 第九部分：未决问题与风险清单

## 30. 高优先级未决问题

## 30. 高优先级待验证事项

| 编号 | 问题 | 当前状态 | 验证要求 |
|---|---|---|---|
| U01 | Qwen3-4B-Thinking-2507 在官方 vLLM Profile 中是否稳定支持 | 待验证 | 模型加载、Thinking、Chat Template、Token IDs、Logprobs |
| U02 | 4090 24GB 下 QLoRA 是否能完成 SFT | 待验证 | NF4、Forward / Backward、保存、重载、峰值显存 |
| U03 | QLoRA Adapter 能否被 vLLM 加载或转换 | 待验证 | Adapter 加载、合并 / 导出和策略更新后重载 |
| U04 | 官方 Q-QLORA 与 PT-RL-CUDA 能否覆盖 QLoRA RL 训练依赖 | 待验证 | 核对官方 Profile；必要时提交官方仓库变更 |
| U05 | vLLM 与 Transformers Log Probability 是否一致 | 待验证 | 相同 Token 序列逐 Token 对照 |
| U06 | 同卡串行 Rollout / Update 的重载开销 | 待验证 | 计时模型启动、保存、切换、重载和更新 |
| U07 | 6 小时核心路线是否可达 | 待验证 | 在 4090 上进行端到端计时 |
| U08 | Equation / Bit 的 Mixed-group Ratio | 待验证 | RL Readiness Audit，按任务和难度统计 |
| U09 | 可验证中间过程奖励是否能改善零方差组 | 未决定 | 仅对有确定性验证标准的步骤做消融 |
| U10 | PPO Critic 与 Reference Policy 的显存设计 | 待验证 | 标准 Actor-Critic 最小闭环和峰值显存 |
| U11 | GSPO 是否作为后续扩展 | 不纳入首版六个 Notebook | 首版交付后再评估 |
| U12 | 真实数据与合成数据的增益 | 未决定 | Real-only / Real+Synthetic / Screened Synthetic 对照 |
| U13 | 最终 OOD 划分 | 待验证 | 按规则组合 / 结构家族隔离 |

已从“未决定”转为“已确认”的事项：三个题型、课程提供先验、可选数据工程、官方仓库集成、官方环境复用、不使用 Unsloth。

## 31. 主要技术风险

### R1：Thinking 输出过长

风险：

-   Rollout 时间增加；
-   KV Cache 占用增加；
-   轨迹缓存变大；
-   PPO 训练序列变长；
-   6 小时目标难以实现。

应对：

-   先测输出长度分布；
-   设置生成上限；
-   记录截断率；
-   比较准确率与输出预算；
-   不要直接截断所有长推理轨迹而不检查完整性。

### R2：推理和训练概率不一致

风险：

-   PPO 重要性比率不准确；
-   GRPO / GSPO 策略更新偏差；
-   训练不稳定。

应对：

-   固定模型版本；
-   固定 Tokenizer；
-   固定 Chat Template；
-   对照 Log Probability；
-   记录 vLLM 和 Transformers 版本。

### R3：同卡同步导致模型重载成本过高

风险：

-   每轮 Rollout 和训练交接耗时；
-   6 小时预算超出；
-   Adapter 合并成为瓶颈。

应对：

-   测量启动、关闭、保存、合并和加载时间；
-   降低同步更新轮次；
-   每轮 Rollout 后执行多个 Mini-batch Update；
-   保留固定的策略版本标识。

### R4：GRPO 奖励退化

风险：

-   同一题目所有回答都正确；
-   同一题目所有回答都错误；
-   组内奖励方差为零；
-   相对奖励无法区分候选。

应对：

-   统计组内奖励分布；
-   选择具有学习空间的题目；
-   分析采样温度和输出预算；
-   只在中间步骤可验证时考虑过程奖励。

### R5：QLoRA 兼容性和质量风险

风险：

-   量化模型加载失败；
-   部分模块不支持；
-   训练数值不稳定；
-   量化损害推理能力。

应对：

-   先做最小训练；
-   保存模型和 Adapter；
-   独立评测；
-   保留 BF16 LoRA 回退路线。

### R6：训练数据污染评测集

风险：

-   合成数据重复；
-   同一规则实例跨切分；
-   训练数据包含测试答案；
-   结果虚高。

应对：

-   按规则家族隔离；
-   检查精确重复和近重复；
-   保存数据生成器版本；
-   固定测试集。

------------------------------------------------------------------------

# 第十部分：开发阶段与验收计划

## 32. P0：技术准入验证

## 32. P0：官方环境与目标硬件准入验证

所有验证都在官方 Datawhale 环境 Profile 中执行，不另建项目环境。

### P0-1：模型和 vLLM

- 在官方 vLLM Profile 加载 Qwen3-4B-Thinking-2507。
- 检查 Chat Template、Thinking 输出和 EOS。
- 获取 Prompt / Completion Token IDs 和所需 Log Probability。
- 记录 4090 峰值显存、吞吐和单题耗时。

### P0-2：LoRA / QLoRA

- 使用官方 `PT-SFT-CUDA` 验证 BF16 LoRA。
- 使用官方 `Q-QLORA` 验证 NF4 QLoRA。
- 完成 Forward、Backward、Optimizer Step、保存和重新加载。
- 记录训练显存和 Tokens/s。

### P0-3：vLLM 与 Adapter 交接

- 训练一个小型 Adapter。
- 保存 Adapter。
- 在官方 vLLM Profile 中加载或转换 Adapter。
- 检查更新前后输出和 Log Probability。
- 记录重载耗时。

### P0-4：RL 最小闭环

- Rollout 生成并保存真实轨迹。
- Verifier 计算 Reward。
- DPO：完成 Chosen / Rejected 和手写 Loss 更新。
- PPO：完成 Actor-Critic、GAE、PPO-Clip 和 Value Loss。
- GRPO：完成 Group Advantage 和一轮更新。
- GSPO：不纳入首版六个 Notebook，后续再评估。

### P0 验收记录

记录模型加载、训练、推理峰值显存；单题生成耗时；Rollout 时间；训练更新耗时；Adapter 保存 / 加载耗时；Log Probability 最大误差；是否完成真实 GPU 更新。没有实际运行记录，不得标记通过。

## 33. P1：Baseline、Prompt 与任务验证

- 冻结三任务数据版本和划分。
- 完成 Baseline。
- 完成 Prompt Engineering 对照。
- 输出 Overall / Per-domain 指标和错误样本。
- Gravity 作为简单任务控制组。

## 34. P2：SFT 与 PEFT

- 实现 Direct SFT。
- 根据时间实现 Joint / Two-stage SFT 对照。
- 比较 LoRA / QLoRA。
- 选择最佳 SFT Checkpoint。
- 检查 OOD 和能力保持。

## 35. P3: DPO / PPO / GRPO Notebooks

- 先运行 RL Readiness Audit。
- Gravity 不进入 RL 主训练。
- Equation Numeric 和 Bit Manipulation 作为 RL 主实验。
- 完成 DPO、标准 PPO 和 GRPO。
- GSPO is not included in the first six Notebooks.
- 对零方差组记录诊断结果，不人为制造奖励差异。

## 36. P4：官方仓库交付

- 按官方 Notebook 风格整理内容。
- 增加 TODO、测试和参考实现。
- 复用官方环境 Profile 和 Quickstart。
- 补充模型 / 数据来源和许可说明。
- 保存实验配置、结果和失败案例。
- 提交官方仓库 PR；不创建独立 Docker 或依赖体系。

## 37. 每个实验必须记录的字段

  类别       字段
  ---------- --------------------------------------------
  实验身份   `experiment_id`、日期、负责人
  模型       Model ID、Revision、Tokenizer Revision
  数据       数据版本、切分、样本数量、Token 数
  Prompt     Prompt 版本、Chat Template
  训练       算法、学习率、Batch、有效更新步、Epoch
  PEFT       LoRA Rank、Alpha、Target Modules、量化配置
  RL         Reward 版本、Rollout 数、采样参数、KL 系数
  硬件       GPU 型号、显存、驱动、CUDA
  软件       Python、PyTorch、Transformers、PEFT、vLLM
  质量       Overall Accuracy、Per-domain、OOD
  资源       峰值显存、训练时间、推理时间、Token 吞吐
  产物       Checkpoint、Adapter、配置、日志
  结论       Accept / Tune / Reject
  限制       随机种子、数据量、评测范围、未验证事项

## 38. 实验结论模板

每个实验按照以下顺序撰写：

1.  实验假设；
2.  实验条件；
3.  控制变量；
4.  改变变量；
5.  评价指标；
6.  实验结果；
7.  与基线比较；
8.  失败样本分析；
9.  资源成本；
10. 结论；
11. 限制；
12. 下一步实验。

不要只写"效果变好"或"效果不好"。

要说明具体在哪个指标、哪个任务、什么条件下发生变化。

------------------------------------------------------------------------

# 第十二部分：当前决策摘要

## 39. 已确认事项（截至 2026-11-05）

Confirmed Notebook count: six. Data Engineering is embedded as an optional branch in Baseline; LoRA / QLoRA is embedded in SFT; the RL Notebooks are DPO, PPO, and GRPO.

| 事项 | 当前决定 |
|---|---|
| 项目归属 | 合并进官方 `datawhalechina/llm-algo-leetcode`，不建独立教程仓库 |
| 教学风格 | 完全遵循官方 LeetCode Notebook 风格 |
| 基座模型 | `Qwen/Qwen3-4B-Thinking-2507`，保留 Thinking |
| 任务范围 | Gravity、Equation Numeric、Bit Manipulation |
| 人类先验 | 课程统一提供；学生研究记忆、检索和应用，不必逆向发现规则 |
| Gravity | 简单控制组；通常 SFT 足够，不进入 RL 主训练 |
| Equation Numeric | 规则归纳任务；区分 Deduce / Guess；主要 RL 任务 |
| Bit Manipulation | Boolean DSL 组合推理；主要 RL 任务 |
| 默认数据 | 课程提前筛选三类任务并提供固定数据划分 |
| 数据工程 | 可选：跳过、只做 Leakage Audit，或做 Synthetic Data + Screening |
| 合成数据 | 不作为必需条件；是否有收益通过对照实验决定 |
| 硬件 | RTX 4090 24GB 为主要目标，RTX 5090 32GB 可选 |
| 时间 | 核心实验目标约 6 小时，尚待实测 |
| Docker | 不制作 Docker 镜像 |
| 环境 | 直接复用官方 Environment Profile 和 Quickstart |
| 依赖安装 | 不维护独立 requirements / installer；Notebook 不临时安装核心依赖 |
| 推理 | vLLM 优先，使用官方独立 Profile |
| 训练 | PyTorch + Transformers + PEFT，自研轻量训练循环 |
| Unsloth | 明确不使用 |
| 量化 | 4090 优先验证 QLoRA；BF16 LoRA 可选对照 |
| RL 执行 | 同卡、同步、串行；不做异步 Rollout |
| RL methods | DPO, standard PPO, and GRPO are included; GSPO is outside the first release |
| RL 准入 | 必须先做 RL Readiness Audit；不人为制造 Reward Variance |
| 训练目标 | 参考答案 + 开放延拓；不预设 RL 一定优于 SFT |

## 40. 当前建议

| 事项 | 当前建议 |
|---|---|
| Notebook organization | Six fixed topics; exact numbering and placement follow the official repository |
| SFT 对照 | Direct SFT、Joint Drill + Full Trace、Two-stage Memorization → Execution |
| QLoRA | 4090 默认优先验证 NF4 QLoRA，BF16 LoRA 作为可选对照 |
| Rollout | vLLM 生成，保存轨迹后关闭引擎，再切换训练环境 |
| DPO | 参考 The Submitter 的 Verifier-guided Offline DPO |
| PPO | 自行设计标准 Actor-Critic + GAE + PPO-Clip |
| GRPO | 参考第 251 名正向案例和第 80 名零方差失败案例 |
| GSPO | Not included in the first six Notebooks; possible future extension |
| Reward | 先使用确定性答案奖励；过程奖励仅在中间步骤可独立验证时加入 |
| 数据工程 | 可选，不把大规模合成数据作为必做前提 |

## 41. 尚未解决的问题

1. Qwen3-4B-Thinking-2507 在官方 vLLM Profile 中的具体版本兼容性。
2. 4090 24GB 下 QLoRA 的真实显存和训练速度。
3. QLoRA Adapter 与 vLLM 的加载 / 合并 / 更新后交接。
4. 官方 Q-QLORA 与 RL 训练依赖组合如何复用，是否需要贡献官方 Profile 变更。
5. vLLM 与 Transformers Log Probability 的数值一致性。
6. 同卡串行 Rollout / Update 的模型重载成本。
7. 6 小时核心实验目标是否可达。
8. Equation Numeric 和 Bit Manipulation 的 Mixed-group Ratio 是否足够。
9. 过程奖励能否在不引入标签噪声的前提下改善稀疏奖励。
10. PPO Critic / Reference Policy 的显存实现。
GSPO 是否作为首版六个 Notebook 之外的后续扩展。
12. Real-only 与 Synthetic Data 的增益。
13. 最终 OOD 数据划分和任务难度覆盖。

# 第十三部分：参考资源

## 42. 教学仓库

-   Datawhale LeetCode 算法教程：\
    https://github.com/datawhalechina/llm-algo-leetcode

-   MiniMind：\
    https://github.com/jingyaogong/minimind

## 43. 基座模型

-   Qwen3-4B-Thinking-2507：\
    https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507

-   Qwen3-4B-Instruct-2507（曾讨论的备选）：\
    https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

## 44. 推理引擎

-   vLLM：\
    https://github.com/vllm-project/vllm

-   vLLM 文档：\
    https://docs.vllm.ai/

-   SGLang：\
    https://github.com/sgl-project/sglang

-   SGLang 文档：\
    https://docs.sglang.ai/

## 45. 训练工具

-   Transformers：\
    https://github.com/huggingface/transformers

-   PEFT：\
    https://github.com/huggingface/peft

-   bitsandbytes：\
    https://github.com/bitsandbytes-foundation/bitsandbytes

## 46. Kaggle 比赛与优秀作品

-   NVIDIA Nemotron Model Reasoning Challenge：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

-   第一名：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution

-   第三名：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution

-   第四名：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution

-   第九名：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/9th-place-solution-human-auditable-cot

-   第十五名：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/wip-lb15-extension-to-huikangs-progress-prize

-   Public 第二、Private 第六：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-public-6th-private-place-solution

-   第七名：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/7th-place-solution-solving-bitmanipulation-via-st

-   第十名：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/10th-place-solution

-   第十三名：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/13th-place-solution

-   第十八名：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/18th-solution-from-deterministic-solvers-to-learn

-   The Submitter：\
    https://github.com/the-submitter/nvidia-nemotron

-   第 251 名 GRPO：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/trace-format-engineering-and-svd-downsampled-lora

-   第 80 名 GRPO：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/breaking-the-sft-ceiling-on-nemotron-3-nano-30b

-   Domdolus：\
    https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/domdolus-tolus-solution

------------------------------------------------------------------------

# 第十四部分：后续维护规则

## 47. 决策更新规则

后续讨论出现新结论时：

1.  如果用户明确确认，更新"已确认事项"。
2.  如果只是助手建议，保留在"当前建议"。
3.  如果需要运行代码或硬件测试，放入"待验证"。
4.  如果方案被否决，标记为"已放弃"，并保留原因。
5.  如果实验结果改变原有判断，更新结论并记录实验条件。
6.  不删除失败实验记录。
7.  不把作者报告的局部实验结果推广为通用规律。

## 48. 版本记录

  --------------------------------------------------------------------------------------------------------
  版本                    日期                    更新内容
  ----------------------- ----------------------- --------------------------------------------------------
  v0.1                    2026-11-04              初版：汇总教学定位、模型选择、硬件约束、训练推理架构、Kaggle 经验和待验证事项
  v0.2                    2026-11-05              确认合并官方 Datawhale；收敛三任务；课程提供人类先验；数据工程改为可选；复用官方环境；排除 Unsloth；明确 QLoRA / vLLM 分环境串行；增加 RL Readiness Audit
  v0.3                    2026-11-05              Freeze the first release at six Notebooks: Baseline (+ optional Data Engineering), Prompt Engineering, SFT (+ LoRA / QLoRA), DPO, PPO, GRPO; embed RL Readiness Audit in the three RL Notebooks; exclude GSPO from the first release.
  --------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## Final Summary

The first release is fixed at six Notebooks and will be contributed to the official Datawhale LeetCode repository:

1. Baseline Evaluation, with optional Data Engineering.
2. Prompt Engineering.
3. SFT, including LoRA / QLoRA.
4. DPO.
5. Standard PPO.
6. GRPO.

The course uses Qwen3-4B-Thinking-2507 and three task families: Gravity, Equation Numeric, and Bit Manipulation. Human prior knowledge is provided by the course. Students focus on knowledge memorization, rule retrieval, reasoning execution, and post-training algorithms.

RL Readiness Audit is embedded in the three RL Notebooks. Final cross-algorithm analysis is included in the GRPO Notebook summary and the course report. GSPO is outside the first release.

The project retains the following principles:

- One fixed base model and a consistent evaluation protocol.
- Single-GPU execution on 24–32 GB cards.
- Synchronous, serial rollout and training on the same GPU.
- Readable PyTorch implementations of the core training algorithms.
- Reuse of the official Datawhale environment profiles.
- No Docker image and no Unsloth dependency.
- A six-hour target for the core route, subject to hardware validation.
- Preserve negative results and do not assume RL must outperform SFT.
