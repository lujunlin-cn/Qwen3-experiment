# Prompt Engineering 指导：让先验进入上下文并可执行

版本：v0.1 \| 对应 Notebook：`02_Prompt_Engineering`

## 1. 目标

研究 Prompt 能否改善 Qwen3-4B-Thinking-2507 对 Gravity、Equation
Numeric、Bit Manipulation
的求解能力。课程已经提供人类先验，学生不需要重新发现隐藏规则，而是研究如何在
Prompt 中表达规则空间、约束和求解步骤。

Prompt Engineering 是独立实验阶段。不能把后续 SFT 的教师轨迹提前混入
Baseline。

## 2. 从选手报告提炼的经验

-   **第二名**：强调使用 Prompt 中所有可用示例、检查全局规则、处理正常 /
    反转模式，并明确未见运算符的 Guess 逻辑。
-   **第三名**：推理轨迹每一步都应从 Prompt
    和已展示的前序步骤推出，不能出现隐藏跳跃。
-   **第四名与第九名**：格式应稳定、简洁、可审计；模型对符号和规则选择的格式变化较敏感。
-   **NVIDIA 官方总结**：压缩重复结构，但不能隐藏关键推理信号。

这些是设计启发，不代表选手的最终成绩由 Prompt Engineering 单独带来。

## 3. Prompt 对照组

  ----------------------------------------------------------------------------------------------
  编号                    Prompt 类型             内容
  ----------------------- ----------------------- ----------------------------------------------
  P0                      Original                保持比赛原始 Prompt，不额外注入规则

  P1                      Output Contract         增加最终答案格式和简洁输出约束

  P2                      Structured Reasoning    解析示例、提出候选、验证规则、计算 Query

  P3                      Prior In-context        将课程提供的候选规则和操作定义放入上下文

  P4                      Few-shot                从训练集选择少量代表性示例

  P5                      Verify-before-answer    要求检查候选规则能否解释全部示例后再输出答案
  ----------------------------------------------------------------------------------------------

P0 是 Baseline。P1--P5
逐项增加组件，不能一次性叠加所有策略后声称某一项有效。

## 4. 三类任务的 Prompt 设计

### Gravity

提供物理公式、参数定义、单位和精度规则。模型需要从题目示例中识别参数，再计算
Query。不要直接提供当前题目的隐藏参数值。

### Equation Numeric

提供候选运算规则、数字反转规则、负数格式规则、同题全局约束，以及 Deduce
/ Guess 的区别。

建议步骤：解析示例；生成候选规则；检查候选是否解释全部示例；检查全局模式；判断
Query 运算符是否出现；执行 Deduce 或 Guess 策略；计算最终答案。

不得把标准答案写进 Prompt。不能将 Guess 题伪装成唯一可解的 Deduce 题。

### Bit Manipulation

提供 8-bit 位编号、Shift / Rotate
定义、布尔操作、表达式优先级和允许的组合语法。

建议步骤：解析输入输出字节；提取候选操作；组合候选表达式；使用全部示例验证；对
Query 应用表达式；规范化 8-bit 输出。

课程提供的是候选规则空间，不是当前题目的隐藏答案。

## 5. 统一输出约定

-   关键中间结果采用固定标签。
-   不输出与求解无关的长篇解释。
-   最终答案使用 `\boxed{...}`。
-   Bit Manipulation 统一输出 8-bit 二进制或明确约定的整数格式。
-   Gravity 明确数值精度。
-   Equation Numeric 明确整数、负号和格式规则。

Qwen3 Thinking 的聊天模板由 Tokenizer
统一处理。不要手工拼接未经验证的模型特殊 Token。

## 6. 实验控制

固定 Base Model、数据划分、解码参数、最大生成长度、Verifier 和 Prompt
版本。首轮优先使用与 Baseline 一致的确定性解码配置。若测试
Temperature，必须作为独立变量。

Prompt 调优只允许使用 Validation。Holdout Test 不得参与 Prompt 选择。

## 7. 评价指标

每个任务分别报告 Accuracy、Answer Extraction Success Rate、Invalid
Output Rate、平均生成 Token、平均推理耗时、Overall Accuracy 和 Macro
Accuracy。

额外记录规则识别正确率、示例验证通过率、Query 计算正确率和错误类型。

## 8. 不应采用的做法

-   不把隐藏规则答案直接放进 Prompt。
-   不允许模型跳过示例验证。
-   不用自然语言解释长度代替推理质量指标。
-   不同时改变 Prompt、Temperature 和 Max Tokens。
-   不在测试集上反复修改 Prompt。
-   不把 Prompt 的提升归因于 SFT 或 RL。

## 9. 交付物

P0--P5 模板、三任务评测结果、错误案例、最佳 Prompt 及其适用边界、Token
消耗和推理耗时对照。

## 10. 参考资料

-   [第二名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution)
-   [第三名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution)
-   [第四名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution)
-   [第九名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/9th-place-solution-human-auditable-cot)
-   [NVIDIA
    官方比赛总结](https://developer.nvidia.com/blog/lessons-from-the-leaderboard-what-5000-kagglers-taught-us-about-improving-ai-reasoning/)
