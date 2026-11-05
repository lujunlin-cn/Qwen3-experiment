# 第一个 Notebook 实现说明

交付文件为 `01_Baseline_Evaluation_and_Dataset_Preparation.ipynb`。Notebook 保留官方数据清洗、分层 IID 划分和 Baseline 主线，默认 `RUN_BASELINE=False`；所有教学实现都在 Notebook Code Cell 中，没有新增 `.py` 教学文件。

## 官方数据主线

- 原始 `train.csv` 重新读取为 9,500 行、`id/prompt/answer` 三列；无缺失、重复 ID 或完全重复 Prompt。
- 精确模板分类得到 Gravity 1,597、Bit Manipulation 1,602、Equation Numeric 732；Equation Symbolic 823 被单独审计并排除。Unit Conversion、Encryption、Numeral System 和 Unknown 也保留排除原因。
- 清洗结果保留原始 `id`、Prompt、Answer、`source_row` 和来源信息。原始 `train.csv` SHA256 为 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`，未修改。
- 真实数据按任务分层、固定种子 42 做 80%/10%/10% IID 划分。精确 ID/Prompt 没有跨集合重复；数字归一化 Prompt 存在跨集合重复。由于没有可靠的规则实例或结构家族键，Notebook 不宣称 Structural OOD 无泄漏。

## 合成数据 Pilot

Notebook 中的确定性 Generator、Solver、Verifier 和 Screening 已实际运行，每类生成 100 条候选，共 300 条：

| task | candidates | accepted | rejected |
|---|---:|---:|---:|
| Gravity | 100 | 100 | 0 |
| Equation Numeric | 100 | 100 | 0 |
| Bit Manipulation | 100 | 100 | 0 |
| 合计 | 300 | 300 | 0 |

- Gravity 使用课程先验公式 `distance = 0.5*g*t^2`，在 9.50–19.50、0.01 网格上检查答案唯一性。
- Equation Numeric 已实现参考 DSL 的 24 条 `(parse, operation, format)` 规则，保留多个运算符、normal/flip family、semantic group、Join 例外和全局约束。24 条规则均在 Pilot 规则覆盖统计中出现。
- Equation Numeric 共 Deduce 80 条、Guess 20 条。Guess 只在未出现的 Query 运算符在全局约束下产生唯一输出时接受；不使用 Validation/Holdout 答案拟合规则。
- Bit Manipulation 修复了 `instantiate_bit_tree()` 的左右子树递归实例化；Prompt 只展示输入输出示例和 Query。深度不超过 2 的 8-bit DSL 通过有限候选枚举和完整 256 输入语义去重，表达式唯一性与答案唯一性分开记录。
- Equation Numeric Verifier 只依据公开 Prompt、24 条规则枚举和全局约束验证答案；`hidden_rule` 仅作为审计元数据，不参与证明答案正确。三个 Verifier 均独立于对应 Generator 的答案计算入口。

产物位于 `data/processed/synthetic/`：

- `synthetic_candidates.jsonl`
- `synthetic_accepted.jsonl`
- `synthetic_rejected.jsonl`
- `synthetic_screening.csv`
- `synthetic_statistics.json`
- `generation_metadata.json`

拒绝文件在本次 Pilot 为空，但仍已创建并保留；后续失败候选会带有逐项拒绝原因。Data Designer 默认关闭，使用 `NVIDIA_API_KEY`，Preview 结果必须重新经过确定性 Verifier。

## RTX 5090 / AutoDL 环境

目标机为 AutoDL `connect.westd.seetacloud.com:48510`，项目和环境位于 `/root/autodl-tmp/Qwen3-experiment` 数据盘。已准备并通过 `uv pip check` 的环境：

- `.venv-pt-sft-cu128`：官方 PT-SFT-CUDA Profile，实际 PyTorch `2.11.0+cu130`。
- `.venv-q-qlora-cu128`：官方 Q-QLORA Profile，含 `bitsandbytes 0.50.2`，实际 PyTorch `2.11.0+cu130`。
- `.venv-pt-rl-cu128`：官方 PT-RL-CUDA Profile，实际 PyTorch `2.11.0+cu130`。
- `.venv-vllm-cu128`：独立 vLLM Profile，vLLM `0.11.0`、PyTorch `2.8.0+cu128`。

模型已完整位于数据盘：

`/root/autodl-tmp/huggingface/models/Qwen3-4B-Thinking-2507`

三个 safetensors 文件大小已核对为 `3957900840`、`3987450520`、`99630640` 字节。首次尝试使用 `hf-mirror.com` 时发生超时；随后已核对并启用 AutoDL 学术加速入口：

```bash
source /etc/network_turbo
# http_proxy/https_proxy=http://10.37.1.23:12798
```

在该加速环境下从 `huggingface.co` 请求 `config.json` 返回 HTTP 200，下载文件与数据盘文件 SHA256 一致。因为模型已经完整存在，没有为验证而重复下载数 GB 权重；后续模型/文件下载应先执行上述入口。

单样本 vLLM smoke 已用数据盘本地权重完成：`data/processed/vllm_smoke_result.json` 记录状态为 `ok`，GPU 为 RTX 5090，vLLM `0.11.0`，生成 32 tokens，耗时约 96.8 秒。完整 Baseline 仍未运行；当前机器上的其他任务可能占用 GPU，需在资源空闲时再按 Notebook 配置执行。

## 未完成事项

- 未运行完整 Baseline，因此没有伪造 Accuracy 或耗时统计。
- 未实际调用 NVIDIA NeMo Data Designer：Notebook 提供已核对的官方 API 分支，但默认关闭；需要 `data-designer` 依赖和 `NVIDIA_API_KEY` 才能 Preview。
- 当前 Bit DSL 是课程用 8-bit 有限子集，不宣称覆盖比赛全部位操作规则。
- 尚未开发后续五个 Notebook。
