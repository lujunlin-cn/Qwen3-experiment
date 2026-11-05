# Nemotron SFT-RL 教程：技术指导文档索引

版本：v0.1 \| 日期：2026-11-05

## 1. 文档体系

本项目采用文档驱动开发。每份文档只指导一个技术模块。实现 Agent
每次只读取本索引、当前模块文档和
`LeetCode_SFT_RL_Tutorial_Design_Record.md`，不要将全部要求拼成一条超长提示词。

  ----------------------------------------------------------------------------------------
  文档                    内容                                     对应 Notebook
  ----------------------- ---------------------------------------- -----------------------
  `01_数据工程.md`        官方数据清洗、划分、可选合成数据与筛查   1\. Baseline Evaluation

  `02_Prompt工程.md`      Prompt 对照、先验注入、输出格式          2\. Prompt Engineering

  `03_SFT.md`             轨迹质量、监督目标、数据配比与训练       3\. SFT

  `04_QLoRA.md`           LoRA / QLoRA、NF4、显存与部署兼容        3\. SFT 内的 PEFT 分支

  `05_DPO.md`             Rollout、Verifier、偏好对构造、DPO Loss  4\. DPO

  `06_PPO.md`             Actor-Critic、GAE、PPO-Clip              5\. PPO

  `07_GRPO.md`            Group Rollout、Reward、Group             6\. GRPO
                          Advantage、零方差诊断                    
  ----------------------------------------------------------------------------------------

七份技术文档对应六个 Notebook。QLoRA 单独成文，但不单独创建 Notebook。

## 2. 全项目固定约束

-   基础模型：`Qwen3-4B-Thinking-2507`。
-   任务：Gravity、Equation Numeric、Bit Manipulation。
-   课程统一提供人类先验；学生研究先验如何进入
    Prompt、训练数据和模型参数，以及如何检索和执行规则。
-   项目合并进 `datawhalechina/llm-algo-leetcode`。
-   首版六个 Notebook，遵循官方 Notebook 写作风格。
-   所有教学实现写在 Notebook 的 Markdown / Code Cell 中；不创建教学用
    `.py` 文件。
-   不使用 Unsloth；不使用高层 Trainer 隐藏核心算法。
-   复用 Datawhale 官方环境 Profile；vLLM 与训练环境分离，在同一张 GPU
    上同步串行执行。
-   开发验证 GPU 为 RTX 5090；教程需评估 RTX 4090 24GB 的可运行性。
-   不把未实测的显存、耗时和性能写成事实。
-   真实数据先清洗和划分，再生成合成训练数据。Validation / Holdout Test
    不参与合成数据生成和调参。
-   分任务报告指标，不能只报告总体准确率。

## 3. 六个 Notebook 的边界

1.  **Baseline Evaluation**：清洗官方数据、基线评测；可选 Data Leakage
    Audit 和 Synthetic Data Engineering。
2.  **Prompt Engineering**：比较原始 Prompt、先验注入、结构化推理和
    Few-shot。
3.  **SFT**：比较 Direct / Joint / Two-stage SFT；LoRA / QLoRA 放在本
    Notebook。
4.  **DPO**：离线偏好数据构造与 DPO 更新。
5.  **PPO**：标准 Actor-Critic PPO。
6.  **GRPO**：组内采样、奖励诊断和策略更新。

Gravity 主要作为 Baseline、Prompt、SFT 的基础能力控制组。通常不强制进入
RL。Equation Numeric 和 Bit Manipulation 是主要 RL 实验任务。GSPO
不进入首版六个 Notebook。

## 4. 证据使用规则

每份文档区分： - **选手报告的观察**：原比赛模型和环境中的结果。 -
**可迁移的方法**：适合在 Qwen3-4B 上重新验证的设计。 -
**本项目建议**：为教学完整性提出的实现方案。 - **待验证事项**：需要在
RTX 5090 上实测的内容。

不同模型、数据集、硬件和评测条件下的分数不能直接横向比较。

## 5. 主要参考作品

### 数据、Solver、合成数据与 SFT

-   [第一名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution)
-   [第一名复现仓库](https://github.com/xrwr/kaggle-nvidia-nemotron-model-reasoning-challenge-1st-place)
-   [第二名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution)
-   [第二名公开榜 /
    第六名私榜方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-public-6th-private-place-solution)
-   [第三名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution)
-   [第四名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution)
-   [第九名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/9th-place-solution-human-auditable-cot)
-   [第十五名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/wip-lb15-extension-to-huikangs-progress-prize)
-   [第十名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/10th-place-solution)

### RL 与失败分析

-   [The Submitter DPO
    Pipeline](https://github.com/the-submitter/nvidia-nemotron)
-   [第 251 名 GRPO
    方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/trace-format-engineering-and-svd-downsampled-lora)
-   [第 80 名 GRPO
    失败分析](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/breaking-the-sft-ceiling-on-nemotron-3-nano-30b)

### 官方总结与教学风格

-   [NVIDIA：Lessons From the
    Leaderboard](https://developer.nvidia.com/blog/lessons-from-the-leaderboard-what-5000-kagglers-taught-us-about-improving-ai-reasoning/)
-   [Datawhale LeetCode
    教程](https://github.com/datawhalechina/llm-algo-leetcode)
