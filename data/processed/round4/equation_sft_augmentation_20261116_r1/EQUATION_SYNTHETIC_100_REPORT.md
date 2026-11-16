# Equation 精确数学修复与100条SFT数据验收

Run：`equation_sft_augmentation_20261116_r1`。随机种子20261116。没有调用NVIDIA Hosted模型，没有使用或要求NVIDIA_API_KEY。

## 修复

原反例 `50+50=100`：旧Solver接受 add/add_p1/add_m1，还以1%容差合并Query输出。现在候选规则严格比较字符串，只保留add，Query `51+50`精确输出`101`。官方最终答案评分比较器保持原样。

Solver保留全部合法Assignment，而不是前20个；每个Query按完整字符串求值，undefined操作不构成合法可计算Assignment。全join规则的normal/flip重复表示被去重。分别记录规则唯一和答案唯一；仅答案唯一时提供展示用witness，不将第一个Assignment当作唯一性证明。

独立Verifier使用独立Prompt解析、独立算术/格式实现及Cartesian Assignment枚举。对候选集合、完整Assignment集合、每个Assignment的Query、全部示例和最终字符串逐项核对。Renderer仅使用公开题目重新计算的引用，不读取Generator规则作为教师。

## 候选与来源

原注册Train Prior v5保持冻结；来源为真实R1使用的Train，3123条、Equation586条。Task Specification的24条规则/全局约束与经验来源分开记录。每个layout与操作数对均保留source_id/source_row/feature/position。未读取Train Answer参与生成。

先固定500个新候选，再加载两个Manifest的三集合做独立审计。候选SHA256：`f2cd1d922125be3fca95f6b7dd98c0b44a62eccf70aa21a350b1fcaed26fadae`。没有依据Holdout审计重新采样。先前已存在历史审计；本轮不宣称Holdout此前从未被读取。

| 范围 | 候选 | 完整通过 | 拒绝 |
|---|---:|---:|---:|
| 原有Equation Preview重审 | 10 | 10 | 0 |
| 新冻结池 | 500 | 374 | 126 |
| 合计 | 510 | 384 | 126 |

通过包括来源、公开Solver、独立Verifier、Generator/Solver/Verifier三方一致、三集合重复/暴露审计和完整SFT Renderer。384条均拥有declared来源。不存在用结构相似指纹替代精确重复判定的操作。

新候选409 Deduce/91 Guess，接近Train的475/111经验比例。最终384条全部为Deduce。126个拒绝中91为Guess、35为Deduce；均存在多个合法Assignment产生不同Query字符串，无法唯一识别答案。拒绝记录完整保留；没有将这些样本改标或加入训练。

**这批数据覆盖Deduce，不代表已经解决Guess。** 筛选明显改变接受集合的子类型分布，这一限制不能用总体384条掩盖。未按Holdout结果调整采样，更没有人为降低难度重新生成来填补Guess。

## SFT教师

继续使用自然语言数学步骤与 `Final answer: \boxed{...}`。逐一展示全部公开示例计算、候选公式、全部合法Assignment和Query结果。多个Assignment答案相同时明确写出`rule_assignment_unique=False`、`answer_unique=True`。

实际本地Qwen3-0.6B Tokenizer计算Completion；384条重复渲染与新独立Verifier再次全部通过。默认任务池、USE_SYNTHETIC_DATA和历史文件未改写。原Gravity10条逐记录字节不变。Roman仅程序化引导句Preview，原题原答案不变，暴露审计事实保留。

本轮明确选取固定顺序前100条Equation，包含重审后的旧10条。剩余284条保持独立Preview，不自动加入训练。专用B组额外使用历史Gravity10条。C组使用真实R1的Train中未被原R1使用的100 Equation/10 Gravity。

B/C每轮完整监督Token：38553/38556，差异0.00778%；两轮累计77106/77112。两组均完整使用110条，不截断Completion。最长选定完整Chat分别670/672 Token，低于4096。

## 复核文件与命令

- `equation_candidates.jsonl`与`candidate_pool_freeze.json`：先冻结的500次尝试。
- `equation_accepted.jsonl`、`equation_rejected.jsonl`：384/126逐条结果。
- `equation_solver_audit.json`、`equation_rejection_details.json`：完整数学证据与拒绝原因。
- `equation_lineage_audit.json`、`equation_distribution_audit.json`：逐特征来源与描述性偏移。
- `equation_sft_rendered_preview.jsonl`：自然语言教师。
- `final_equation_render_reverification.json`：384条新复核，确认真实监督序列包含完整reasoning。
- `sft_actual_inclusion_manifest.jsonl`：明确区分历史Preview快照和实际隔离训练的纳入状态。

CPU回归最终84项通过，另在低显存环境执行17项Equation/训练数学测试通过。保留首次错误导入路径与中止启动的日志；不将失败尝试当作成功。

维护命令见 `utils/run_equation_augmentation.py --help` 与 `utils/run_sft_augmentation.py --help`。各阶段拒绝覆盖已有产物；不能直接再次执行本Run生成。
