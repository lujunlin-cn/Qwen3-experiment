# Equation 合成数据修复与SFT增强实测报告

合成增强B组Equation为3/50，原始R1为0/50，等监督Token真实Train对照C为1/50。B−C精确配对检验p=0.625，当前没有足够证据证明合成数据比真实Train增量训练更有效。Macro Accuracy为B 15.3%、C 16.0%。训练和450条Validation输出均已真实完成；一般自由语言证明的完整验证仍未解决，不能宣称过程正确率提升。

最终有效 Run：`equation_sft_augmentation_20261116_r2`。代码实现提交 `629c3fd`；具体执行源码、补丁登记与Hash见 `source_commit_binding.json`、`sft_experiment_registration.json`、`loader_registration.json`；最终源码与完整产物绑定见 `final_source_binding.json`、`artifact_archive_manifest.json`。实验只修改合成数据和独立SFT；未修改DPO/PPO/GRPO或默认SFT入口。

## 结论的适用范围

本轮数据工程门槛通过：384条Equation Proof/Renderer候选，正式B组使用100 Equation+10历史Gravity。A/B/C在同一固定Validation面板评测，每任务50题。所有均采用同一基座、Tokenizer、原始Prompt和Thinking开启的greedy解码，Roman输出上限2048，Gravity/Equation4096。**这不是历史8192/16384上限的复现，也不是Holdout成绩。**

全部新Equation训练记录为Deduce；Guess仅作评测。Roman的标准映射在历史R1已训练，属于能力保持，不能解释为映射OOD泛化。

## 1. 基线与数据隔离

真实父Adapter SHA256：`d565b67a4e09d41b260b3929d5f3600b96e78ea8e7a8f3712dbfc749c190d371`。基座Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`，权重文件SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。不是CPU not_run Manifest或24条实验Adapter。

原R1真实192条教师，各任务64条，24次更新。实际使用taskset `roman_gravity_equation_v1`；与Notebook03当前默认Manifest不同。本轮对两套三集合都重新做Prompt/重复审计，训练与评测资格只绑定真实R1的划分。不能将alternate Manifest的Prompt去重通过说成alternate Train-only来源通过。

原20条Preview均重审通过。Gravity10条记录和Completion字节保持不变。Equation10条公开数学重新计算，教师补齐逐例运算和全部Assignment。来源仍为Python Solver+Renderer，不是Nemotron自主CoT。

历史R1实际64条Equation教师中，58条的最终字符串与当前精确Solver一致，21条Trace列出过不能精确匹配全部示例的候选规则。原始官方答案64/64与精确Solver一致；六条差异由旧Solver教师输出引入，并非官方Label错误。历史记录与权重不变，见`sft_r1_equation_teacher_quality_audit.json`。因此A vs增强组也包含教师合同质量修正因素，不能只归因为合成来源。

## 2. Equation精确修复与候选

中间规则与Query输出以字符串严格相等判定，保留符号/前导零/反转/拼接；官方1%最终评分保持原样。独立Verifier不调用Solver的解析、算术和候选函数，采用独立Cartesian约束枚举。对比完整候选集合、完整Assignment集合与全部Query结果，分别记录规则唯一和答案唯一。

冻结500个新候选，374通过；旧10条重审通过，合计384条。126条拒绝均答案不唯一（91 Guess、35 Deduce），没有改答案或放宽标准凑数。训练B选固定顺序前100条，加Gravity10条。实测来源/精确数学/泄漏/Renderer/Tokenizer/Boxed回环全通过。

生成仅读冻结Train Prior及其Manifest/源Train字节Hash。每条经验特征保留source_id、source_row、position；Numeric24规则作为固定Task Specification单独Hash。旧候选池在正式修复r2中原字节复用，零重新采样。

`}`运算符曾导致Boxed提取错误：r1备用Preview5条、Real Control2条。旧训练/部分评测均标记排除并保留。r2只转义LaTeX花括号，原Prompt/Answer、候选池和Real Control ID均不改。604条逐项提取回环全部通过。

## 3. Token匹配的真实训练

A不训练，保留真实R1。B/C分别独立从该R1加载；均BF16 LoRA，继承r8/alpha16/dropout0，AdamW LR2e-5、weight_decay0.01、eps1e-8、microbatch1、累积8、max_grad_norm1、两轮、seed20261116、max_length4096。原R1在NF4训练，本轮B/C同用BF16；不将二者混写。

| Arm | 训练记录 | 累计监督Token | 更新 | 训练秒 | Torch分配峰值GB | Torch保留峰值GB | 改变Tensor |
|---|---:|---:|---:|---:|---:|---:|---:|
| A R1 | 0新增 | 0新增 | 0新增 | — | — | — | — |
| B Synthetic | 110 | 77106 | 28 | 75.33 | 1.457 | 1.556 | 392 |
| C Real Control | 110 | 77116 | 28 | 72.36 | 1.458 | 1.556 | 392 |

B/C监督Token差异0.01297%，低于1%；两组完整使用全部110条两轮，未截断Completion。C的原始ID和Prompt均排除了R1已有教师，且逐条核对实际Train/held-out隔离。

本机为共享GB10，其他既有进程未停止。仅报告本进程Torch峰值，**不是整卡显存小于8GB的证明**；nvidia-smi对整卡UMA总量不可用，已有其他进程远超8GB。训练时间受共享负载影响，不当作独占性能基准。真实训练曲线：`sft_training_curves.png`。

B/C规则族比例分别为47%/64% Flip。任务与Token匹配不等于语义难度完全匹配；只有一个训练Seed，置信区间反映本面板逐题不确定性，不涵盖训练随机性。详细分布见`sft_training_distribution_comparison.json`。

| Arm | Adapter SHA256 |
|---|---|
| B | `dafb648cd850b996a3e0d154291222ef6a32301af8080420c220eaa5a5a9cc3a` |
| C | `7033241d41efb086762df20789ce2b70764bb3a5c524e29b8d8f846d3254e3f4` |

权重、Optimizer保存在Git外`.autodl/`独立路径，完整文件Hash见`checkpoint_archive_manifest.json`。原R1不变。`sft_actual_inclusion_manifest.jsonl`单独记录真正进入隔离训练的记录；旧Preview中的未合并标志保留历史含义。默认SFT数据未合并。

## 4. 固定Validation逐题评测

本轮选择50题/任务，合计150，三组共450条完整逐题记录。选择按固定RNG在生成模型结果前登记，不按效果选题。输入Token不计入输出长度；生成ID长度包含Thinking和EOS、错误及截断。

| Task | Arm | Correct /50 | Accuracy (Wilson95) | Truncation | Extract Success | Mean Output | Median Output | Mean Correct Complete | Median Correct Complete |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| numeral_system | A | 10/50 | 20.0% [11.2%, 33.0%] | 68.0% | 32.0% | 1835.12 | 2048.00 | 1301.00 | 1246.00 |
| numeral_system | B | 12/50 | 24.0% [14.3%, 37.4%] | 0.0% | 58.0% | 645.18 | 170.00 | 163.08 | 146.00 |
| numeral_system | C | 11/50 | 22.0% [12.8%, 35.2%] | 6.0% | 66.0% | 577.48 | 336.00 | 271.00 | 260.00 |
| gravity | A | 4/50 | 8.0% [3.2%, 18.8%] | 92.0% | 8.0% | 4014.88 | 4096.00 | 3082.00 | 2913.00 |
| gravity | B | 8/50 | 16.0% [8.3%, 28.5%] | 0.0% | 100.0% | 786.54 | 721.00 | 926.88 | 1044.50 |
| gravity | C | 12/50 | 24.0% [14.3%, 37.4%] | 0.0% | 100.0% | 1139.42 | 1100.50 | 1175.42 | 1217.00 |
| equation_numeric | A | 0/50 | 0.0% [0.0%, 7.1%] | 98.0% | 2.0% | 4049.58 | 4096.00 | NA | NA |
| equation_numeric | B | 3/50 | 6.0% [2.1%, 16.2%] | 28.0% | 64.0% | 1601.70 | 561.50 | 737.67 | 483.00 |
| equation_numeric | C | 1/50 | 2.0% [0.4%, 10.5%] | 26.0% | 74.0% | 1445.00 | 289.00 | 165.00 | 165.00 |

Macro Accuracy：A=9.3%, B=15.3%, C=16.0%.

评测时间及引擎累计Torch保留峰值（包含初始化/预热；同一引擎串行切换Adapter，不作为独立整卡峰值）：

| Arm | Decode Seconds | Engine Torch Reserved Peak GB |
|---|---:|---:|
| A | 2306.77 | 5.771 |
| B | 1073.43 | 5.771 |
| C | 1120.96 | 5.771 |

### Equation子类型

| Arm | Type | n | Correct | Accuracy | Truncation | Mean Output |
|---|---|---:|---:|---|---|---:|
| A | deduce | 42 | 0 | 0.0% | 100.0% | 4096.00 |
| A | guess | 8 | 0 | 0.0% | 87.5% | 3805.88 |
| B | deduce | 42 | 3 | 7.1% | 28.6% | 1665.40 |
| B | guess | 8 | 0 | 0.0% | 25.0% | 1267.25 |
| C | deduce | 42 | 1 | 2.4% | 26.2% | 1453.29 |
| C | guess | 8 | 0 | 0.0% | 25.0% | 1401.50 |

Equation最终答案与官方目标完全同字符串的数量：A=0/50, B=2/50, C=1/50。这是附加字符串指标，官方Accuracy仍按原评分器计算。

### 完整数学过程验证的覆盖限制

独立Verifier仅认证可精确重建的公开数学Proof。模型自由自然语言没有闭合Schema，本轮使用保守的规范Full Trace全文证书（忽略空白）；无法匹配的自由表达标记**unverified**，不能写成数学推理错误。Validation公开Solver Reference覆盖Roman50、Gravity50、Equation36；其余Equation14题仍参与官方答案评分，不换题、不改标签，但不能获得本Verifier的完整过程证书。

| Task | A certified | B certified | C certified |
|---|---:|---:|---:|
| numeral_system | 0/50 | 0/50 | 0/50 |
| gravity | 0/50 | 0/50 | 0/50 |
| equation_numeric | 0/50 | 0/50 | 0/50 |

因此Full Trace认证率是一个保守下界及格式覆盖指标，不能将其直接解释为所有自然语言证明的真实正确率。本轮三组均无规范全文匹配，无法回答一般过程质量是否改善；这不是三组所有数学过程均错误的证明。

## 5. 同题配对分析

主要对照是C vs B，差值方向为B−C。精确McNemar、10,000次配对Bootstrap；其余为探索性指标。p值未做多重比较校正，不能将不显著解释为等效、无损或非劣。

| Comparison | Task | Accuracy difference | Paired Bootstrap95 | Exact McNemar p | Mean output difference | Correct-complete paired n |
|---|---|---:|---|---:|---:|---:|
| A_vs_B | numeral_system | 4.0% | [-10.0%, 18.0%] | 0.79053 | -1189.94 | 4 |
| A_vs_B | gravity | 8.0% | [-2.0%, 20.0%] | 0.28906 | -3228.34 | 2 |
| A_vs_B | equation_numeric | 6.0% | [0.0%, 14.0%] | 0.25000 | -2447.88 | 0 |
| A_vs_C | numeral_system | 2.0% | [-14.0%, 16.0%] | 1.00000 | -1257.64 | 3 |
| A_vs_C | gravity | 16.0% | [4.0%, 30.0%] | 0.03857 | -2875.46 | 2 |
| A_vs_C | equation_numeric | 2.0% | [0.0%, 6.0%] | 1.00000 | -2604.58 | 0 |
| C_vs_B | numeral_system | 2.0% | [-10.0%, 14.0%] | 1.00000 | 67.70 | 6 |
| C_vs_B | gravity | -8.0% | [-22.0%, 6.0%] | 0.38770 | -352.88 | 4 |
| C_vs_B | equation_numeric | 4.0% | [-4.0%, 12.0%] | 0.62500 | 156.70 | 0 |

截断的配对检验、完整正确的共同子集长度Bootstrap、全部输出长度Bootstrap均保存在`sft_statistical_analysis.json`。不把错误答案变短解释为收益。逐题结果包含官方正确率与额外精确字符串正确标志；官方标准未改。

## 6. 决策

1. Equation正确率：A=0/50，B=3/50，C=1/50。B−A=6.0个百分点，B−C=4.0个百分点；主要配对检验p=0.62500。没有足够证据认定合成数据优于Token匹配的真实Train增量训练。
2. 完整数学过程：仅可报告上表的程序证书覆盖，不能用未验证自由文本制造过程收益。
3. 训练Token因素：B/C步数、数据条数与监督Token相匹配；A vs B的改变包含额外训练本身，来源效应主要看B vs C。
4. Gravity保持：A/B/C分别4/50、8/50、12/50。按实际差值报告，不能用不显著证明无回退。
5. Roman保持：A/B/C分别10/50、12/50、11/50。按实际差值报告，不能用不显著证明无回退。
6. 扩量：数学与来源生产链可复用，但先验布局筛选排除了本批全部Guess。本轮未显示相对真实Train的可靠收益，暂不建议直接扩大数据规模；先解决Guess可识别性与自由语言过程验证，再登记独立重复实验。不自动扩量或合并默认SFT。

## 7. 工程验证、失败与复现

训练前85项CPU回归通过，最终99项相关CPU回归通过，含240,000次Numeric24 Generator/Solver/Verifier全操作数对比、唯一/非唯一Assignment、精确字符串、前导零、符号/前后缀、全局冲突、错误过程和Boxed回环。实际Tokenizer复核604条。完整日志、失败启动和排除r1产物全部保留。

训练使用本地Torch2.11.0+cu130 / Transformers5.12.0；评测使用固定Docker镜像vLLM0.27.1-aarch64（image ID在后端登记，最终batch8、KV4GiB、无前缀缓存）。容器`--network none`，无Hosted API。为可行性将初始慢速Torch评测统一替换为本地vLLM，三组全量重新开始，旧部分输出排除；不因结果好坏切换模型或解码。

完整数据在训练后又接受了Query表达式/公式元数据负例审查；所有384条实际证明保持通过。另修复拒绝日志将答案不可识别误标为解析器不一致的问题：126条拒绝样本的Solver/Verifier实际全部一致，失败原因是Query不唯一。补丁登记明确为CPU认证/诊断加强，没有改变训练数据或任何参数更新。

本机共享内存短时不足导致多次加载失败，失败都在更新前，后续CPU-first模型加载绕过可选allocator warmup；真实CUDA分配仍受本进程限制。科学配置与模型权重没有因此变更。

自动离线评分链曾在容器输出交接时发生两次PermissionError：容器目录归属被误设为UID1000，本机执行用户实际为UID1001。更正三个评测目录的所有权后，离线评分与统计成功，原始450条Completion和Token IDs不变，失败日志全部保留。未重新推理，也未重新训练。

维护入口：

```bash
# 阶段只允许首次写入；历史Run禁止覆盖。
.venv-low-memory/bin/python -m utils.run_equation_augmentation --help
.venv-low-memory/bin/python -m utils.run_sft_augmentation --help
# 评测原始数据完整后，可在新结果目录离线复算：
.venv-low-memory/bin/python -m utils.score_sft_augmentation
.venv-low-memory/bin/python -m utils.report_sft_augmentation
```

本轮没有读取Holdout做效果评测，也没有修改或替换官方答案。之后若做Holdout，必须锁定全部模型和协议，独立登记最终评测，不能继续调参。
