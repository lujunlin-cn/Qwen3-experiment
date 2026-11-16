# Gravity DPO-B：SFT C 最终实验报告

Run：`dpo_gravity_sft_c_20261116_r1`。**最终状态：`no_preference_signal`。**

全部312条真实GPU Rollout已完成（26题×初始8条，再全题各补4条）。最终官方答案通过32/312，独立自然语言证明认证0/312，有效DPO-B偏好0/26，未达到16对门槛。实际Optimizer Steps为0；未创建DPO Checkpoint、未执行Tune/Locked Final比较，不能宣称DPO收益。

## SFT C权重与数据来源

Base：Qwen/Qwen3-0.6B，Revision `c1899de289a04d12100db370d81485cdf75e47ca`。Policy和计划中的冻结Reference同为SFT C。Adapter：`.autodl/equation_sft_augmentation_20261116_r2/C/adapter`。

Adapter SHA256：`7033241d41efb086762df20789ce2b70764bb3a5c524e29b8d8f846d3254e3f4`。

启动前逐文件核对Base权重、config、tokenizer、vocab、merges和generation_config，共7项匹配历史指纹。BF16 LoRA r=8、alpha=16、dropout=0，无R1或其他RL回退；原Adapter Hash保持不变。证据：`identity_audit.json`。

16条历史Gravity真实Train题逐一核对实际SFT Train的ID、Prompt、Answer、source_row；10条现有Gravity合成题保留`synthetic_train_prior`、Prior Manifest Hash和逐特征来源，来源ID/行号均属于实际Train。26题ID及Normalized Prompt无重复，与实际Validation/Holdout无Normalized Prompt重复。证据：`train_panel.jsonl`、`data_audit.json`。

Tune48及Locked Final100复用历史冻结面板，均来自Validation，并非官方Holdout Test。二者无交集，Hash已冻结；可能曾被历史实验评测，不能声称首次盲测。本轮因数据门槛失败，没有对这些面板执行推理。

## 自然语言数学Verifier

核心实现在Notebook01新增Gravity DPO-B Section；Notebook04保留DPO-B选对及数学loss教学实现。专用Runner/Worker只做调度、模型I/O、归档并接入`course_worker.train_dpo`，不执行Legacy DPO Notebook入口，不要求JSON证明。

Verifier独立解析全部公开时间/距离，用`2*math.fsum(t*t*d)/math.fsum(t**4)`重新计算OLS，不调用Generator/Solver计算函数。检查最终回答中的物理关系、全部示例拟合依据、g、Query时间、未舍入距离、最终盒装答案及可解析的中间算式。计算精度保持rel_tol=1e-4、abs_tol=1e-5；Query时间是已知输入，数值精确匹配。无需逐条打印g_i，但打印索引g_i时会独立检查。

物理表达式通过闭合AST转为符号表达式验证等价性；数值算式不执行eval。不使用模型自评，不用全文/模板字符串相等作为数学标准。原有10条SFT Gravity教师及自然语言改写正例通过CPU测试。

Thinking只保留审计：模型启用Thinking时，仅取最后一个`</think>`后的最终回答。输入Chat Template已开启但响应未关闭的Thinking也拒绝。所有Thinking、错误/截断输出和EOS仍计入Response长度以及计划中的Sequence Log Probability。

最终v1.2将**官方答案提取/比较与Proof Format分开**。原v1.1采用较窄的numeric-box提取，最终对全部原始输出使用现有官方extract_final_answer；本轮二者均为32条。这消除了把Proof Format当成Accuracy开关的实现问题，没有放宽证明格式、数学容差或偏好标准，证明与偏好仍为0。v1.1原始状态及报告保留。

**解析限制：**这是保守的有限语法证书检查，尚不能覆盖任意等价LaTeX/自由表述。0/312是认证通过数，不是证明所有推理都错误。不可解析、缺少关键拟合/Query计算或无法验证完整过程的回答标为unverifiable，不加入偏好。

## 真实GPU采样结果

硬件为本机共享NVIDIA GB10，非历史RTX4080。未终止其他GPU进程。模型只收到原始公开Prompt及固定Chat Template；未追加JSON指令，未发送标准答案、Solver Proof或隐藏规则。

Temperature1.0，top_p1.0，top_k=-1，Thinking开启，Max Tokens16384；每题同一采样合同。原始Completion、Prompt/Response Token IDs、逐Token Behavior Logprobs、finish_reason、Policy Fingerprint及来源均归档。

|阶段|实际回答|生成耗时（不含初始化）|
|---|---:|---:|
|初始8条/题|208|1932.54s|
|补充4条/题|104|1093.84s|
|合计|312|3026.38s|

|最终离线审计|结果|
|---|---:|
|官方最终答案通过|32/312|
|Format Valid|98/312|
|独立完整证明认证|0/312|
|有效偏好对|0/26|
|Truncation|0/312|
|finish_reason=stop|312/312|
|Response Token总数（含Thinking/EOS）|287099|
|Mean Response Tokens|920.19|
|Behavior Logprobs对齐且有限|True|
|所有stop回答包含真实EOS|True|

冻结generation_config中的合法EOS为151645和151643；实际分别出现311次和1次，均保留并计入Token总数。

本进程Torch CUDA峰值reserved为5771362304 bytes（约5.771GB），低于8,000,000,000 bytes分配器守卫。UMA设备无法可靠提供全设备显存；该值不是整机显存，也不保证涵盖所有非Torch CUDA分配。

逐来源、逐题和非互斥失败原因见`final_quality_summary_v12.json`。多个失败原因可属于同一条，不能相加当作独立失败数。逐条Verifier结果见`final_rollouts_audited_v12.jsonl`。官方答对但过程未认证的原始案例及参考值见`official_correct_but_unverified_cases.jsonl`，包括只输出答案、缺少拟合/Query计算和中间数值错误。

这些是Train Rollout质量统计，不是Validation准确率；不能以它们替代SFT C与DPO的效果比较。

## 偏好门槛、真实训练与评测

Chosen取同题最短、Rejected取最长的正确完整认证回答，差>=64Token且>=长回答20%，每题最多1对。错误、截断和不可验证回答均排除。初始208条不足16对，自动对全26题各补4条；达到每题12条上限后仍无有效对，按协议停止。没有借用教师回答、伪造Chosen或改成正确/错误偏好。

|阶段|真实状态|
|---|---|
|Optimizer Steps|0，未启动DPO训练|
|Loss/真实梯度|无训练结果，不填虚构数值|
|Checkpoint及重载|未产生，未执行|
|Tune SFT C vs DPO|未执行|
|Locked Final SFT C vs DPO|未执行|
|Wilson/McNemar/Paired Bootstrap|无两模型评测数据，未计算|
|DPO-B目标|无法验收，无收益证据|

已实现标准Sigmoid DPO、冻结SFT C Reference、完整Response求和mask、低显存分块logprob、beta0.1、LR2e-6、accum4、2epochs、clip1.0及梯度/曲线/Checkpoint重载后自动Tune/Final。CPU测试通过不代表本轮发生过真实GPU DPO更新；数据门槛阻止了它。

## CPU回归、完成hook与历史隔离

初始GPU前60项CPU回归通过；查看本轮Completion前的附加反例修复矛盾g、尾随文字中的错误算式、未关闭Thinking等问题，后续阶段前64项通过。旧source-hash守卫阻止旧完成链继续；登记v1.1后自动续链，初始208条未重生成。aarch64 Python缺少os.pidfd_open的hook启动失败日志保留，随后使用glibc pidfd_open/select等待完成，未轮询训练进度。

最终离线修复官方答案提取与Format混用，增强可选索引g和已知Query时间检查。**最终69项CPU测试实际通过，耗时4.455s**，证据`cpu_final_tests.log`。对全部312条原始输出重审，仍为no_preference_signal；未新增GPU/API请求。v1.1自动结果、旧注册、失败日志和原始回答均保留，v1.2单独归档。

SFT C、Notebook03默认入口、PPO/GRPO、历史DPO/SFT产物及原始数据保持不变；无NVIDIA Hosted API调用。完整性见`final_publication_manifest.json`；原始v1.1自动报告见`automatic_final_report_v11.md`。

## 最终判断

本轮完成312条真实GPU采样和全部严格数据审计，但未得到启动DPO-B所需的16对偏好。不能宣称完成真实DPO训练、缩短输出、保持正确率或改善数学证明。正确率、长度、截断率和必要证明率四项效果条件均没有两模型比较证据。当前瓶颈是SFT C最终回答的可验证完整性及解析覆盖，未通过降低门槛制造学习信号。
