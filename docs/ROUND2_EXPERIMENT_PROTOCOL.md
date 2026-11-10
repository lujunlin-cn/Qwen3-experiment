# 第二轮实验协议与审核门槛

版本：2026-11-10 v1；状态：**只完成诊断、设计与CPU准备，等待人工审核和GPU开机；不启动Phase C/D**。

本轮只做 Gravity DPO-B；PPO/GRPO只训练Equation；SFT、DPO、PPO、GRPO不得串行继承。NeMo修复接受的数据不自动并入SFT，不用变动训练数据同时解释Reward收益。

## 1. 资产与不变量

- 新准备Run：`round2_20261110_phaseab_v1`。不复用 `course8h_20261109_r2_cuda_graph`。
- 模型 Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`。
- 唯一SFT父权重：第一轮r1，SHA256 `d565b67a4e09d41b260b3929d5f3600b96e78ea8e7a8f3712dbfc749c190d371`。初始化SFT Adapter，不继承其Adam到新的RL方法。各RL后续轮次才恢复自身Adam/Value Head。
- 官方划分和最终答案不变，官方评分不放宽；Train/Validation/Holdout SHA256见面板Manifest。Holdout只用于完整性/泄漏审计，不用于任何训练、调参、早停或Run选择。
- `.py` 为维护者Runner/数据/统计/测试；算法和Verifier教学实现保留在对应Notebook。
- 每个正式Run保存git commit、dirty/source hashes、模型/tokenizer fingerprint、config SHA256、data/panel SHA256、verifier版本、parent checkpoint、采样参数、真实token IDs及行为概率。GPU执行必须从审核后的干净提交注册。

审核稿配置：[round2_design.json](../configs/round2_design.json)。它不是旧八小时Runner可执行配置；旧schema校验会拒绝，避免误启动历史串行调度。审核后注册独立执行目录，不悄悄修改旧manifest source hash。

## 2. Phase A：已执行的诊断和设计

已完成：原18个NeMo输出逐项诊断/严格重筛、四算法路径审查、DPO-B质量门控、P0–P3确定性奖励定义、实际Validation规模清点与面板Hash冻结。根因与逐条证据见 [诊断报告](ROUND2_DIAGNOSIS.md)。

没有执行：新模型诊断API调用、新NeMo模型生成、真实第二轮rollout/train/final。所有计划配置明确Not Run。

## 3. Phase B：CPU 与小模型数值验证

执行真实Torch反传，模型为随机初始化的小型Qwen3+PEFT，不是用Mock梯度替代；模型质量不作推断。覆盖：

1. 标准sigmoid DPO，多个β的loss/梯度对照Dense。
2. 手算GAE/Return，PPO Clip的解析梯度，Value零结果仍可校正；action轴/terminal/padding。
3. 全对/全错/mixed的GRPO优势与梯度；不同长度sequence mean；完整证明才给长度分。
4. 独立过程Verifier部分正确、错误Query和重复JSON键；不调用Solver也能验证。
5. 完整151936词表投影与Dense、padding/checkpoint交叉条件的loss/梯度等价。
6. 同算法Adam恢复等价于不断点两步；四方法各两轮真实参数更新、Adapter重载、PPO Value Head恢复。
7. 空Accepted、三集合泄漏、实际NeMo SDK72个模拟HTTP的字段/Schema传递，以及预算不足不能标final完成。

结果与局限见 [代码审查](ROUND2_CODE_AUDIT.md)。GPU内核、性能、显存、真实模型质量必须留到C验证。

## 4. Phase C：审核后的短程真实GPU顺序

| 顺序 | 独立Run/工作 | 初始采样规模 | 通过条件 |
|---|---|---|---|
| C0 | 环境/父权重/模型指纹、设备空闲与守卫检查 | 无长训 | 不覆盖历史、环境匹配、守卫有效 |
| C1 | NeMo R0、A、B、C | 各9 Seed×2列，共72输出 | 传输、原题/证明/Audit可信，接受率如实；0Accepted可作为质量失败结果保留 |
| C2 | 冻结SFT的真实Reward readiness probe | Equation 4题×8 | F/S/P/L、组方差、时间与输出长度可追溯，不要求强造mixed |
| C3 | Gravity DPO-B | 16题×4候选；至多16对、2epochs | 双侧正确完整/Trace合格、有真实optimizer.step、保存/重载、峰值<阈值 |
| C4 | Equation PPO P0/P1/P2/P3 | 每消融8题×1；首轮同SFT | GAE/Value/terminal对齐、真实更新、分量可追溯、峰值<阈值 |
| C5 | Equation GRPO P0/P1/P2/P3 | 每消融4题×8 | population std/优势正确，有效组与更新数如实、峰值<阈值 |
| C6 | 相同Tune子集跨三任务能力评测 | 每类8题，Base/SFT及训练后模型 | 逐题结果齐全、同题配对、失败/截断不删 |

Run命名样式分别为 `round2_dpo_gravity_b_*`、`round2_ppo_equation_p0_*`…、`round2_grpo_equation_p0_*`…；每算法/消融/重复分别建目录。不得把所有配置塞入同一个GPU Run并中途改权重。

P0–P3只能复用“相同SFT行为策略的首批冻结Rollout”做分量比较；训练分叉后不得继续复用旧策略样本冒充on-policy。DPO reference固定为SFT，与PPO/GRPO无Reference KL是不同约定。

Phase C训练题ID也已冻结在 `phase_c_train_panels/`，Gravity16、PPO Equation8、GRPO Equation4（PPO子集），各消融复用同一题单；选择不查看Validation/Holdout结果。

任何方法0更新只说明没有练到训练路径，不能算显存训练通过。GRPO若全无信号则C失败/需重新设计；不补噪声。NeMo可接受0 Accepted，但此时不能声称已修好质量；Phase D使用新合成数据必须另行审核合格Manifest。

## 5. 显存和耗时

单GPU、单进程组阶段隔离；vLLM退出并回收后才训练。设备总采样峰值阈值 **8,000,000,000 bytes**，不是8GiB；0.25s轮询，同时记录Torch allocated/reserved。监控异常、达到阈值立即终止本实验进程组；不终止无关服务。采样未捕获的瞬时峰值明确未验证。

拟用KV3GiB、max_num_seqs1、batched_tokens512、max_model_len20480，保留第一轮batch1 decode CUDA Graph；训练microbatch1、LoRA、decoder checkpointing、use_cache=False、64Token词表投影。没有1F1B或多GPU流水线。

原始cap不自适应降低：Roman8192、Gravity/Equation16384。训练和评测分别记录初始化、rollout、验证、forward/backward、optimizer、保存/重载及总墙钟时间；阶段停止原因区分预算耗尽、无信号、失败、预设轮数、Tune平台期。短程停止不能称充分收敛。

预算先按第一轮吞吐作规划，再用C2实际长度/吞吐修订。当前配置给出4–6小时的中位规划区间，**不是完成保证**。仅DPO64+PPO32+GRPO128条训练候选，若全部到16384即3,670,016输出Token；两种契约、5模型×每类8题评测另最多3,276,800 Token。按200Token/s，上述生成上限已约9.65小时，尚未计NeMo、readiness、训练与启动。若只提供8小时，应分会话推进并保留未完成状态，不削减既定final、不偷改cap、不声称全阶段完成。

## 6. Phase D：扩大评测与正式训练

只有人工批准且A/B/C工程与真实更新门槛满足后注册D。若某算法C无信号或数值/显存不通过，则暂停该分支，保留其他分支证据，不能用它的空训练结果冒充“所有算法都通过”。

面板：Roman Tune48/final98；Gravity48/100；Equation24/37。第一轮每类12个已暴露Panel题全部排除。Hash已冻结；Phase C的8题从Tune取，不触碰final。正式训练题量/总更新预算根据C覆盖率与耗时另冻结，不从final错误反选训练题。

先注册固定轮数/最大step、按Tune准确率优先的checkpoint选择规则，再运行。DPO选择必须符合准确率与截断不回退；PPO/GRPO选择以Equation准确率为主，报告跨任务变化。长度不允许抵消明显准确率下降。最大轮数和平台期如实标识，不能都叫收敛。

全量final只在选择的各方法Checkpoint、Reward/提示词/停止规则全部冻结后一次执行。所有5模型在同一题、同一模板、greedy、同cap、同Thinking和同verifier下比较。原题-only和闭合证明契约两个条件分开报告，不能把换Prompt的收益归为训练收益。预算不足就记录missing job/未完成，必要时另立continuation记录，不覆盖失败attempt。

## 7. 指标、统计与验收

逐题保存：原题ID/任务/原split、prompt与token Hash、官方answer、raw completion、生成token IDs、finish/truncation、官方评分、Format、闭合证明验证与范围、Reward分量、policy/parent与Verifier版本。

统一汇总Accuracy/Wilson95、Truncation Rate、Mean Output Tokens、Correct Output Mean Tokens、Exact Trace Pass Rate、Format Valid Rate、Reward Mean，以及GRPO Group Variance/有效组比例。Equation闭合证明条件下，所有模型另用相同P3诊断奖励汇总；训练时各P0–P3分量另列，不能直接拿不同定义的均值互比。不存在过程验证的原题输出记“未测/不可验证”及分母，不把缺字段伪报0%或100%。全部输出统计包含Thinking、错误、截断与EOS，不包含输入。

统计采用 [round2_evaluation.py](../utils/round2_evaluation.py)：严格匹配题ID、prompt IDs、verifier、采样/seed、variant/cap；准确率与截断精确McNemar，长度全题配对bootstrap95和sign-flip检验，另报双方均正确子组。长度随机化检验依赖零假设下可交换/对称假设，小样本谨慎解释。主准确率家族为DPO Gravity、PPO Equation、GRPO Equation，Holm校正；其他指标标为次要/探索性，不择优报显著。

DPO-B四项观察成功条件：准确率不低于SFT、平均输出下降、截断不增加、必要闭合证明保持。PPO/GRPO看Equation正确率的逐题配对证据，不能把短答或格式率提升等同于推理改善。Roman/Gravity跨任务保持评测必须披露回退。Equation final37题功效有限，不能承诺检测几个百分点的收益，也不能用Holdout补调参。

## 8. 本次交付与审核范围

本次提交六份文档、Notebook直接修改、独立Verifier/奖励/偏好分支、CPU回归证据、R0/A/B/C配置、冻结面板与Hash。当前没有任何第二轮真实GPU成绩或Accepted增长承诺。

下一步人工审核的是任务分工、DPO仅B、Reward权重/门控、面板与资源预算。审核且GPU开机后再注册并执行C；不因服务器可连接就自动开始D。
