# Qwen3-0.6B 教学实验综合报告与下一日排查计划

记录日期：2026-11-14。下一次计划日期：2026-11-15。本文根据已取回的本地代码、原始输出、Verifier 记录、Checkpoint Manifest 和统计编写；没有启动新训练，也没有重新修改历史实验。

**当前结论：低显存训练的主要工程路径已经得到验证，数据质量和学习效果还没有整体验收通过。明天按“合成数据 → SFT → DPO → GRPO → PPO”处理，每一步先解决输入与信号问题，再决定是否训练。这个顺序是排查顺序；三个 RL 方法仍从同一个选定 SFT Checkpoint 独立出发。**

## 1. 当前状态与任务分工

| 环节 | 已确认的事实 | 尚未解决的问题 | 明天优先处理 |
|---|---|---|---|
| 合成数据 / NeMo | 同样的 9 条 Seed 均通过 Solver、Verifier、泄漏审计；72 次真实模型调用成功 | 生成后的 Accepted 只有 1/72，且仅 Roman；Gravity/Equation 为 0 | 逐条定位复制、结构、数值证明失败，分别修复 Prompt Variant 与 Trace Variant |
| SFT | 第一轮实际更新 72 次；第二轮复用其第 1 轮 Adapter | 新面板上三题均未显示稳定收益；没有学过第二轮闭合 JSON 输出契约 | 审核实际序列化教师答案，先补齐可验证轨迹与推理/输出契约覆盖 |
| DPO | 第二轮仅 Gravity DPO-B，真实生成 64 条候选 | 正确且完整的长短偏好对为 0，未实际训练 | 先得到同题双侧正确、完整、有长度差的候选 |
| GRPO | 第二轮真实生成 4×8 条 Equation 候选 | F/P/S 全 0，组方差全 0，未实际训练 | 先检查新 SFT 的可验证部分过程、正确性与组内方差 |
| PPO | 第二轮两次真实更新；Adam、Value Head 与 Adapter 恢复通过 | 原题 Equation 仍 0/8，Gravity 有回退；参数更新不等于正确性学习 | 在真实任务信号充分后检查更新，并完成固定同题对照 |

任务范围保持为：

- **SFT：**覆盖三类题，Roman 为主要展示方向，同时为 Gravity 偏好和 Equation RL 提供合格起点。
- **DPO：**只做 Gravity 的 DPO-B，正确短答优于正确长答；不做 DPO-A，不训练 Roman/Equation。
- **GRPO、PPO：**只训练 Equation；Gravity、Roman 用于能力保持评测。
- **奖励：**先保留一个复合配方 `R1=F+0.20P+0.05S`，长度权重为 0，不同时运行多奖励消融。
- **效果评测范围：**按最近指令，当前完整模型效果对照只保留 Base/SFT/PPO。明天处理 DPO/GRPO 的数据、信号与训练问题，不自动追加其完整模型评测；没有相应效果评测时，只能验收工程机制，不能宣布它们改善了任务。

## 2. 版本、数据与验收口径

| 项目 | 本轮固定记录 |
|---|---|
| 仓库 | `lujunlin-cn/Qwen3-experiment` |
| 当前已归档结果提交 | `d19b376` |
| 第一轮 Run / 执行源码 | `course8h_20261109_r2_cuda_graph` / `98787e2` |
| 第二轮 Suite / 执行源码 | `round2_20261114_phasec_r1` / `f1502fc` |
| 模型 | `Qwen/Qwen3-0.6B` |
| 模型 revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| 第二轮共同 SFT 父权重 | 第一轮 SFT r1，SHA256 `d565b67a4e09d41b260b3929d5f3600b96e78ea8e7a8f3712dbfc749c190d371` |
| 实测设备 | 服务器报告 RTX 4080 SUPER、32760 MiB |
| 已记录GPU软件环境 | Torch2.8.0+cu128；vLLM0.11.0；NeMo Data Designer0.10.0；推理Transformers4.57.1、训练Transformers5.18.0 |
| 显存阈值 | `8,000,000,000 bytes`，每 0.25 秒采样整卡用量 |
| 输出长度 | 仅生成 Token，包含 Thinking、错误、截断及 EOS，不含输入 |
| 原题评测解码 | greedy；Roman cap8192，Gravity/Equation cap16384；Thinking 开启 |
| 第二轮已完成面板 | 每任务 8 道 Tune Validation，Base/SFT/PPO 同题同配置 |

8 GB 是本项目的采样阈值，不是本次设备的物理容量。已测低于阈值不证明采样间瞬时上界，也不等于已经在实体 8 GB 卡上验证。显存必须区分 rollout、真正反向训练、模型加载、无信号检查和重载。

原题条件与 `closed_trace_v1` 是两个独立条件。闭合证明只认证可解析的最终证明对象，不认证任意自由文本 Thinking。原题没有要求该结构时，Format/Exact Trace 记“未测”，不能记为 0% 并解释成全部推理错误。

## 3. 两轮实验分别证明了什么

### 3.1 第一轮：四种方法都有真实更新

| 方法 | 实际优化器更新 | rollout 采样峰值 GB | 反向训练采样峰值 GB |
|---|---:|---:|---:|
| SFT | 72 | 不适用 | 1.724 |
| DPO | 4 | 5.198 | 2.416 |
| PPO | 3 | 5.198 | 4.796 |
| GRPO | 2 | 5.198 | 4.412 |

这些是历史配置的工程证据，不能代替第二轮新偏好与新奖励的验收。第一轮 DPO 包含 Gravity/Equation，第二轮仅 Gravity DPO-B；两轮 GRPO 的奖励、归约定义也需要按对应协议区分。

第一轮每任务只有 4 道调参题，独立最终评测未完成。GRPO 第 6 轮原始停止原因是 `budget_exhausted`，不是显存超限。不能根据该小面板承诺三类任务已被解决。

第一轮 SFT 三轮累计 72 次更新；第二轮使用的是第 1 轮 Adapter，不是把三轮累计更新都算到该选定权重上。第一轮已观察到 SFT 改善部分截断和长度，因此“只有 RL 才能改善截断/长度”不能作为预设结论。

### 3.2 第二轮：只有 PPO 进入了真实反向更新

| 环节 | 真实候选数 | 实际更新 | 采样峰值 GB | 结论 |
|---|---:|---:|---:|---|
| NeMo R0/A/B/C | 72 | 不适用 | 3.808 | 调用完成，质量仅接受 1 条 |
| Equation readiness | 32 | 0 | 5.198 | 所有组无奖励差异 |
| Gravity DPO-B rollout | 64 | 0 | 5.198 | 偏好对 0，不能训练 |
| DPO-B 筛选 | — | 0 | 0.271 | 不是反向训练峰值 |
| PPO 第 1 轮训练 | 8 | 1 | 3.360 | 实际更新并保存 |
| PPO 第 2 轮训练 | 8 | 1 | 3.177 | 恢复 Adam/Value Head 后实际更新 |
| Equation GRPO rollout | 32 | 0 | 5.198 | 全部组奖励方差为 0 |
| GRPO 无信号检查 | — | 0 | 2.036 | 不是反向训练峰值 |

第二轮没有重新训练 SFT。DPO-B 和 GRPO 没有产生训练后策略；不能把 SFT 权重当作它们的结果，也不能把它们此次较低的加载/筛选显存写成训练通过。

## 4. 已完成的同题效果对照

以下为第二轮**原题条件**，每行 n=8，不与第一轮每类 4 题的结果混算。

| 任务 | 模型 | 正确 /8 | 截断 /8 | 平均输出 Token |
|---|---|---:|---:|---:|
| Roman | Base | 5 | 1 | 2613.12 |
| Roman | SFT | 2 | 2 | 3540.00 |
| Roman | PPO | 3 | 3 | 4518.75 |
| Gravity | Base | 8 | 0 | 6017.25 |
| Gravity | SFT | 7 | 1 | 6059.50 |
| Gravity | PPO | 6 | 1 | 8056.75 |
| Equation | Base | 0 | 6 | 13618.88 |
| Equation | SFT | 0 | 8 | 16384.00 |
| Equation | PPO | 0 | 7 | 15139.12 |

PPO 相比 SFT 的 Equation 仅有一题少输出 9959 Token，其余七题仍到上限；这解释了平均长度减少 1244.875 Token。正确率仍为 0/8，不能将这一变化称为解题能力提升或稳定简洁化。

PPO 对照 SFT：Roman 净增 1 道正确，但净增 1 道截断；Gravity 丢失 2 道正确、新增 1 道正确，净减 1 道，均长增加 1997.25 Token。三任务准确率精确 McNemar p 值均为 1。n=8 时每题占 12.5 个百分点；0/8 的 Wilson 95% 区间约为 [0,32.44%]。未显著不能解释为等效或非劣效。

闭合证明条件未完整取回。断连前最后只读观察为 Base 12/24 条，本地最近同步副本只保存了其中 4 条 raw partial；SFT/PPO 闭合条件没有完成证据。原题 72 条结果已完整取回并复算。服务器最后返回 `Connection refused`，远端最终执行状态未知；不能直接判断它正常结束或已经停机。

## 5. 明天第一项：合成数据质量

### 已确认

| 检查 | 结果 | 含义 |
|---|---|---|
| 输入 Seed | 9/9 Solver、Verifier、Leakage 通过 | 原始参考数据可以作为本批测试输入 |
| 四组题目与标准答案 | 相同 | 没有通过修改输入制造收益 |
| 模型调用 | 72/72 成功 | vLLM—NeMo 传输链路可用 |
| R0/A/B/C Accepted | 0/18、0/18、0/18、1/18 | 输出质量问题仍未解决 |
| B 的 Trace JSON Schema | 9/9 通过 | 结构约束已生效 |
| B 的 Trace 数学验证 | 0/9 通过 | 合法 JSON 不能代替正确计算 |

唯一 Accepted 是 C 组 Roman `100→C` 的 Trace；其余 71 条保持 Preview。36 条 Prompt Variant 没有合格项；36 条 Trace Variant 只有上述 1 条通过。R0 内容与第一轮对应输出完全相同，不是额外的独立质量改善证据。

### 下一步操作

1. 先复用现有 72 条输出逐条诊断，不先扩大生成量。每条对齐 `Seed→Generator→Solver→Verifier→Screening→Accepted`；列出原题保护、JSON、数学步骤、Leakage 的失败原因，重叠原因不能加成不同样本。
2. Prompt Variant 优先解决原题完整复制。若改为“模型只生成允许的引导句、程序逐字拼接原题”，必须另立新实验并标记为程序组装版本；不能宣称模型已经学会完整复制。原题仍需独立求解、验数与三集合审计。
3. Trace 单独处理。先对每类一条已知失败输入比对实际 HTTP Schema、字段、原始输出与数值计算，再注册同题新一轮。一次只改变提示词、结构约束或 Solver 辅助中的一项，保留对照。
4. 如使用 Solver 提供中间量，记录其内容和来源，标记为 Solver 辅助合成；不向训练/评测题偷偷注入官方 Query 答案或隐藏生成规则。Solver 生成的教师轨迹与 Qwen 原生生成轨迹分开记来源。
5. Gravity 检查 `g_i=2d_i/t_i²`、既定 OLS、Query 时间代入与最终距离。示例存在舍入时，不能要求所有 `g_i` 逐位相等；按预先定义的关系和数值精度验证，不临时放宽标准。
6. Equation 必须提供合法规则映射，重算全部示例和 Query，保留运算符、前导零及前后缀。Roman 必须检查标准映射、规范分量、求和和拼接。

**继续条件：**新批次每条 Accepted 独立验证与三集合 Audit 全通过；作为最小三任务工程检查，每类至少出现一条合格 Trace，Prompt Variant 也至少有合格原题保护实例。这个最低数量不是接受率或泛化效果证明。若仍有题型为 0，则记录该失败并停止扩量；不把未经验证输出交给 SFT。

合成质量暂未修好时，SFT 可继续使用已有真实 Train 的独立验证轨迹，不需要用 Preview 填充数据。两条数据来源分别记录，不能把真实 Solver 教师称作 NeMo 生成收益。

## 6. 明天第二项：SFT 起点与输出契约

### 已确认的输入与实现

已有真实教师池为 2943 条：Roman1260、Gravity1277、Equation406。第一轮每轮实际使用 192 条，三类各 64；不是每轮训练全部 2943 条。第一轮记录每轮有效 assistant Token22082、最长完整序列287，教师未截断。

本次只读检查 `03_SFT.ipynb` Cell8/10/17 和冻结的 192 条面板发现：

- `_format_trace` 输出自然语言步骤和 `Final answer: \boxed{...}`；192 条都不是第二轮要求的闭合 JSON 教师。
- `encode_chat_example` 将 reasoning 写入 Qwen 的 Thinking，将最终答案写入 content；已有前缀定位和 Prompt `-100` 掩码检查。
- 64 条 Equation 教师都有两个候选规则字段及 Query 输出字段，未发现“候选规则字段全丢失”的情况；但教师正文主要列候选和查询结果，没有按新契约逐项输出全部 `example_outputs`。

本次只读序列化检查的输入Hash、数量和范围见 [SFT格式审查](../data/processed/round2/report_review_20261114/sft_format_audit.json)。这是文本格式与字段覆盖检查，没有把它标为新一轮Tokenizer、数学或GPU验收。

**据此确认了训练输出格式与第二轮 RL 契约不一致。它是需要优先处理的问题，但现有结果不能单独证明它造成了全部正确率回退。** 教师覆盖、更新量、Thinking 终止、训练 NF4/推理 BF16 的差别也需要分开检查，不能一次全部改掉。

### 下一步操作

1. CPU 审查“实际序列化后的教师回答”，不能只审查原始 `trace` 字典。逐条用独立 Verifier 验证；检查字段顺序、示例完整性、结束标记、Thinking 边界、Response mask 和 EOS。
2. 从真实 Train 的现有验证轨迹构造与推理相同的闭合证明教师；保留原题和答案，标注这是确定性 Solver 教师。训练数据不能依赖已经看过的 Validation 错题选择。
3. 首个对照只改变教师输出契约及与其一致的训练指令，尽量保持题单、优化器、精度、学习率、LoRA、解码和预算一致。新格式与新任务数据的收益必须分清；后续再单独检查规模或精度。
4. 先用少量固定 Train 题做可复算的生成检查，观察能否从 Thinking 结束并输出完整证明；检查生成实际内容，不以 loss 下降代替该检查。小批训练题复现通过也不能当作 Validation 提升。
5. 再在固定 Tune 面板比较 Base、旧 SFT、新 SFT，原题与闭合契约分别记录；Roman 是展示重点，但 Gravity/Equation 的退化必须披露。

**继续条件：**教师序列本身全部合格；保存/重载与 label mask 通过；生成端能出现真实、正确、可验证的完整输出，而不只是合法 JSON。新 SFT 同时记录末答与过程覆盖，供下游 readiness 判断。若新格式下仍长期全错或全截断，先解决 SFT，不直接靠 RL 补救。

现有 `round2_run.py` 注册的是 baseline/readiness/DPO/PPO/GRPO，不含第二轮新 SFT 的独立 Run；明天应通过 Notebook03 的实际训练入口记录新 SFT Manifest，再把其真实 Hash 注册给三个 RL 分支，不能声称旧 Suite 已支持这条新路径。

## 7. 明天第三项：Gravity DPO-B

当前 16 题×4 候选中：末答正确7/64、格式合格20/64、完整证明0/64、截断0/64，平均输出6984.59 Token，偏好覆盖0/16。44 条 JSON 解析失败，另20条没有完整数值证明。

原题 Gravity 的 Base8/8、SFT7/8 与上述采样正确率不是同一条件：题单、采样和输出契约都不同，不能直接计算“加入契约导致下降多少”。同题闭合条件结果尚未完整取回。

### 下一步操作与门槛

1. 先固定选定的新 SFT，检查 Gravity 的正确完整回答覆盖；保持每题4条、同题同 Prompt/Policy、真实生成 Token IDs。
2. 只在两个回答都正确、完整且必要证明通过时配对：Chosen 为最短合格答，Rejected 为最长合格答。长度差至少64 Token，且不小于长答的20%；至多每题1对。错误、截断、跨题、不同条件回答不参与正式偏好。
3. 先记录覆盖率和长度分布，再决定训练规模。没有合格对就写 `no_preference_signal`；若 SFT 已普遍简洁，无法构成长短对，也可能表示没有足够长度优化空间，不人为拉长或伪造 Rejected。
4. 有真实对后才做标准 sigmoid DPO：冻结初始新 SFT Reference；Policy 从同一 Adapter 初始化；检查 sequence-sum logprob、Response mask、β、累积与顺序反传。参考起点 β=.1、LR2e-6、累积4、2epochs，数据不足时不靠增加 epoch 凑结论。
5. 至少完成一次真实更新、独立保存/重载及反向显存记录，才能验收该配置的训练路径。

当前不自动追加 DPO 完整效果面板。若只做到上面的数据和训练验证，结论只能写“DPO-B 能训练”。以后要写“DPO 达成目标”，必须有固定同题 Gravity 对照，同时满足正确率不低于 SFT、均长下降、截断不增加、必要证明保持；不能以偏好训练 loss 或更短输出代替效果评测。

借鉴 The Submitter 的是同题采样、独立验证和冻结 Reference 流程；DPO-B 是本项目针对简洁性的扩展，不是其原始偏好方案。上游榜单增益不作为本项目预期收益。原始源码复核见 [DPO设计](DPO_GRAVITY_DESIGN.md)。

`g=2d/t²` 是有效的中间关系检查，但本次64条中仅2条被识别出该关系，且没有一条因此获得完整证明认证。公式出现不自动获得完整正确奖励，也不能代替全部数值步骤。

## 8. 明天第四项：Equation GRPO

当前32条候选全部末答错误，格式、过程和 Combined Reward 全0；四组 population variance 与 advantage 全0，实际更新0。12条存在 JSON 解析错误，其余20条未通过结构或数学验证。这里没有可验证过程信号，不应给错误样本补“正确性奖励”。

奖励定义保持明确：F为完整、未截断且官方末答正确；S为严格结构合规；`P=(E+G+Q+C)/4`，E为全部示例中精确解释的比例，G为全局映射合法，Q为在合法且全部示例解释后Query计算正确，C为答案与该Query结果一致。格式不合规时P=0，截断时所有分量为0。错误完整回答最高.25，正确完整回答至少1；这是单条奖励优先级，不保证策略更新后的正确率单调增加。

R1的过程定义与权重是本项目设计，不是已经由优秀选手验证过的原配方。可借鉴的是可复算轨迹→SFT→RL与组方差诊断；第251名的成功方向、第80名的零边界组失败、The Submitter的实验性奖励应分清来源，详见 [Reward研究与定义](PPO_GRPO_EQUATION_REWARD.md)。

### 下一步操作与门槛

1. 用选定的新 SFT 重新做预注册 Train readiness；每题8条，原题与输出契约不变，逐候选保存 F/S/P。
2. 将组按最终正确性分成全对、全错、Mixed；另报组合奖励方差，不能把有格式分的全错组称为 Mixed。
3. 保留 `R1=F+.20P+.05S`。P由合法全局规则、示例一致比例、Query计算与答案抄录的程序验证组成；格式不合法时P=0。不能靠模型自评、模糊文字相似度或长度给错误答案补分。
4. 存在真实部分正确中间步骤时，过程分可以区分全错组；所有候选仍同分时，就继续记录零优势。全对组也可能零方差，不强造差异。
5. 确认行为 logprob 对齐、population std、sequence-mean 归约、token mask、ratio/clip 后，才对有信号批次反传。记录有效组比例、实际 step、梯度、保存/重载和显存。

**继续条件：**readiness 至少出现真实非零组内奖励方差，并可追溯到正确性或独立验证的格式/过程差别；完成真实更新才能验收训练路径。只有格式分差异时，必须标为格式信号，不能写成数学学习通过。若仍为全零，不反复扩大 rollout，更不提高温度直到凑出有利组而隐瞒采样成本。

Equation 官方 Train586题中，既有24规则分析为406题 Query唯一、180题存在多个合法输出。保留官方答案和评分；对歧义单独分层记录。合成数据的唯一解要求不能偷偷替代官方真实题评分，Validation 难题不能被事后删除。

本轮 GRPO 的完整效果评测仍不自动扩展。过程与格式信号出现、甚至参数有更新，都不足以证明 Equation 正确率提升。

## 9. 明天第五项：Equation PPO

本轮首轮8条全部奖励0，第二轮8条中1条末答正确，格式/过程仍0。两轮重新采样且策略变化，不能将0/8→1/8当作配对收益。全零奖励时非零 Value 估计仍可产生 Critic 校正和策略优势，这与 GRPO 的组内零方差机制不同。

### 下一步操作与门槛

1. PPO 从与 GRPO 相同的新 SFT 独立初始化；不能继承 DPO 或 GRPO 权重，也不能使用它们的旧样本冒充 on-policy。
2. 保持同一个R1，先看真实 F/S/P 分布；首轮若只有 Value 归零方向的更新，应明确记为 Critic/工程行为，不宣布解题能力学习。
3. 检查 old/current logprob、动作前 hidden state 的 Value、终端奖励位置、γ=λ=1、GAE/Return、全局优势标准化、clip和padding/EOS/length mask。
4. 验证第二轮恢复 Adam 与 Value Head；保留更新前后参数与 logprob probe，记录真实 step，而不以历史轮数代替更新次数。
5. 完成固定同题 Base/SFT/PPO 对照，Equation 为主指标，Roman/Gravity 为能力保持；在完整结果保存之前，不把训练候选或 partial 当评测结论。

**继续条件：**真实数学或部分过程学习信号、正确动作时间轴、实际更新/恢复和显存检查均有记录；最终效果必须看逐题正确率、截断与长度。不能承诺再训练若干轮就会提高。

当前 Reference KL 系数0，表示没有冻结 SFT Reference KL 正则；记录的 old-policy approximate KL 不是该正则。vLLM与训练后端对齐存在数值差异，PPO第二轮平均绝对logprob差约.019，单Token最大1.163；重载probe差为0是另一项检查。继续报告差异，不通过替换行为分母掩盖它。

## 10. CPU 数学与工程验证状态

| 已执行检查 | 实际结果 | 能支持的结论 |
|---|---|---|
| 远端课程 CPU 回归 | 11项通过 | 核心训练/恢复机制在小模型CPU路径可执行 |
| 远端数值与奖励回归 | 13项通过 | DPO梯度、PPO GAE、GRPO优势、奖励及配对统计机制检查通过 |
| 远端 NeMo SDK 检查 | 3项通过 | 字段、Schema和筛查的模拟传输检查通过，不是模型质量 |
| 远端解释器隔离检查 | 1项通过 | NeMo环境入口隔离检查通过 |
| 完整词表分块/Dense等价 | 既有CPU 7种检查通过 | 对已测FP32/BF16、padding/checkpoint等条件的loss/梯度等价，不是任意CUDA kernel逐bit证明 |
| 原题结果独立复算 | 72/72完整输出通过一致性检查 | 原始回答/Token、官方答案、评分与指标可对齐 |
| 篡改反例 | 5项全部被拒绝 | 修改答案、回答、Token、评分或截断会触发归档检查失败 |
| 中断归档 | 通过 | 默认拒绝把running导出为完成；显式恢复副本保持incomplete |
| PPO二进制备份 | 两轮Adapter/Adam/Value Head Hash一致 | 本地保存/恢复证据可追溯 |

数值机制通过不代表Qwen学会数学。明天修改Notebook、奖励、输出契约或恢复逻辑后，只重跑受影响回归，再做真实GPU短程；不修改历史测试日志来覆盖失败。

算法教学仍写在Notebook01/03/04/05/06；`.py`承担Runner、数据和检查。实际运行配置以新注册Manifest为准，不能让Notebook展示与Worker执行出现两套Dropout、奖励或Mask定义。

## 11. 显存与时间预算

### 保留已验证的压显存路径

rollout：KV Cache3GiB、`max_num_seqs=1`、prefill512Token、最大上下文20480、batch1 decode CUDA Graph。训练：microbatch1、LoRA r8/alpha16/dropout0、梯度累积、decoder checkpointing、`use_cache=False`、64Token分块完整词表投影。DPO共享冻结基座并顺序反传长短两侧。**没有1F1B或多卡流水线。**

继续限制单GPU阶段进程隔离、整卡采样与8GB阈值。不能通过截断教师、降低正式max_tokens或删掉错误/截断样本伪造低显存、低长度和高正确率。不同序列长度的实测峰值不可当作任意16384输出反传上界。

### 当前实测时间

| 阶段 | 已完成耗时 | 说明 |
|---|---:|---|
| NeMo四组 | 280.51秒 | 含各组启动/退出 |
| Gravity DPO-B 64候选 | 2432.29秒，约40.5分钟 | 真正训练0步 |
| Equation readiness32候选 | 923.58秒，约15.4分钟 | 真实生成与检查 |
| GRPO32候选 | 861.59秒，约14.4分钟 | 真正训练0步 |
| PPO两轮rollout | 463.24秒，约7.7分钟 | 每轮8候选 |
| PPO两轮更新计算 | 34.74秒 | 包含初始化/对齐/保存的两个训练守卫合计84.20秒 |
| Base/SFT/PPO原题评测 | 3233.99秒，约53.9分钟 | 三模型各24题 |

已完成18个守卫阶段累计8331.50秒，约2小时19分；不包含安装、阶段间开销和未完成的闭合评测。第一轮SFT三轮纯训练约337.50秒、验证约1341.75秒，也表明长输出生成与验证是主要时间成本。没有可靠的“本课程收敛总时长”。

### 若明天仍按8小时GPU窗口规划

| 顺序 | 预留分钟 | 目标与超时处理 |
|---|---:|---|
| 取回断连产物、核对环境与Hash | 20 | 只读识别旧进程与产物，不能直接重启旧队列覆盖状态 |
| 合成数据定位与一轮定向修复 | 45 | 先旧证据后小批次；仍0覆盖就停止扩量 |
| SFT契约修复与短程验证 | 80 | 固定首个对照；先通过教师与生成检查 |
| DPO-B数据与训练路径 | 60 | 先偏好门控；0对不浪费训练预算 |
| GRPO readiness与训练路径 | 50 | 先组信号门控；0方差不反复盲采 |
| PPO短程更新与恢复 | 45 | 数学信号、真实step与保存检查 |
| 固定Base/SFT/PPO评测 | 120 | 提前保留，不拿去追加训练；不覆盖锁定final |
| 故障与完整归档缓冲 | 60 | 预算不足就保留未完成，不临时减cap |
| 合计 | 480 | 规划分配，不是五项全部解决的保证 |

上述评测预算优先覆盖固定短程面板，并非承诺全部扩大Tune和final、两种契约都能测完。若SFT关口未过，后续RL转为诊断/准备并保存结果，不硬凑五种方法成功。正式长训仍需在短程质量、信号、更新和资源门槛满足后另行决定。

## 12. 明天的统一评测与交付

官方Train/Validation/Holdout不改。现有第二轮面板为Roman Tune48/final98、Gravity48/100、Equation24/37，Phase C每类8题是Tune子集。已暴露的8题可用于后续Tune诊断，不能重新包装成未见final；Holdout不用作调参或补统计功效。

各模型必须同题、同Prompt/契约、同cap、同解码比较；若改契约，两侧模型都用新契约并与原题条件分开汇总。保留逐题错误与截断，不只报有利的平均值。

统一记录Accuracy/Wilson95、Truncation、Mean Output Tokens、Correct Output Mean Tokens、双方均正确子组长度、Exact Trace、Format、F/P/S/Combined；GRPO另报组类型、方差、优势和有效组比例。准确率/截断做配对McNemar，长度做配对区间与检验，探索性结果不夸大为显著。没有正确回答时正确均长记N/A。

新实验建议使用独立命名 `round3_20261115_<stage>_<attempt>`（仅命名建议，尚未注册）。每个Run冻结Git commit、源码Hash、模型revision、config/data/panel Hash、父Checkpoint、实际环境、生成参数与Verifier版本。若选中新SFT，必须用它的真实Hash重新注册三个RL分支；不能往旧协议里悄悄替换父权重。

| 每一步需保留 | 最低产物 |
|---|---|
| 数据质量 | 原始Seed/请求/输出、逐阶段拒绝原因、Accepted/Preview Manifest、三集合Audit |
| SFT | 实际序列化教师与独立验证、Token/Label审计、训练日志、Checkpoint及重载 |
| DPO | 每题候选、双侧验证、Token长度、偏好覆盖、Reference logprob来源 |
| GRPO | F/P/S与Combined、全对/全错/Mixed、方差/优势、真实step |
| PPO | 行为概率、Value/GAE/Return、终端位置、真实step、Adam/Value恢复 |
| 资源与评测 | rollout/train分别计时与显存、逐题结果、配对统计、未完成清单 |

明天结束时，五个环节分别给出“工程通过 / 质量通过 / 效果证据不足 / 未执行 / 中断”等实际状态，附证据；不使用一个总的passed覆盖其中的零Accepted、零偏好、零梯度或缺失评测。

## 13. 本地证据入口

- [第一轮GPU总结](GPU_VALIDATION_SUMMARY_20261110.md)：历史更新、SFT训练规模、显存、预算中断。
- [第二轮GPU详细结果](ROUND2_GPU_RESULTS_20261114.md)：当前完整与未完成结果的边界。
- [NeMo输入Seed与输出门控复核](../data/processed/round2/real_gpu_20261114_phasec/nemo_seed_and_output_gate_audit.json)、[分阶段统计](../data/processed/round2/real_gpu_20261114_phasec/nemo_stage_metrics.json)。
- [DPO-B原始结果](../data/processed/round2/real_gpu_20261114_phasec/dpo_gravity_b/RESULTS.md)、[GRPO原始结果](../data/processed/round2/real_gpu_20261114_phasec/grpo_equation_r1/RESULTS.md)、[PPO两轮结果](../data/processed/round2/real_gpu_20261114_phasec/ppo_equation_r1/RESULTS.md)。
- [独立评测汇总与逐题配对](../data/processed/round2/real_gpu_20261114_phasec/evaluation_snapshot/independent_summary.json)、[闭合证明失败分类](../data/processed/round2/real_gpu_20261114_phasec/closed_trace_failure_categories.json)。
- [代码审查与CPU回归](ROUND2_CODE_AUDIT.md)、[原始环境日志](../data/processed/round2/real_gpu_20261114_phasec/environment/manifest.json)、[备份Hash](../data/processed/round2/real_gpu_20261114_phasec/checkpoint_backup_integrity.json)。
- [断连记录](../data/processed/round2/real_gpu_20261114_phasec/connection_interruption.json)：本地恢复副本状态为`snapshot_incomplete`，原始running未被改写。
- 本地PPO二进制备份：`.autodl/round2_20261114/ppo_full/`；本地只读恢复快照：`.autodl/round2_20261114/recovered_snapshot_20261114_connection_refused/`。二进制不提交Git。

旧诊断和实验设计文档记录的是其编写时点；GPU是否执行以及最新成绩，以本报告和2026-11-14真实证据为准。本文的明日步骤均为待执行计划，不是已经完成的修复。
