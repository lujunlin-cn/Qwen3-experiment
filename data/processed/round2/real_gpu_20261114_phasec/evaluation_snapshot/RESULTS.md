# 第二轮短程真实GPU验证结果

| 条件 | 模型 | 题型 | n | 正确 | 截断 | 平均输出Token | 正确回答平均Token |
|---|---|---|---:|---:|---:|---:|---:|
| eval_base_original | base | equation_numeric | 8 | 0 | 6 | 13618.88 | N/A |
| eval_base_original | base | gravity | 8 | 8 | 0 | 6017.25 | 6017.25 |
| eval_base_original | base | numeral_system | 8 | 5 | 1 | 2613.12 | 1851.8 |
| eval_sft_original | sft:r1:d565b67a4e09 | equation_numeric | 8 | 0 | 8 | 16384.00 | N/A |
| eval_sft_original | sft:r1:d565b67a4e09 | gravity | 8 | 7 | 1 | 6059.50 | 4584.571428571428 |
| eval_sft_original | sft:r1:d565b67a4e09 | numeral_system | 8 | 2 | 2 | 3540.00 | 1182 |
| eval_ppo_original | ppo:round2:e1b284163fc3 | equation_numeric | 8 | 0 | 7 | 15139.12 | N/A |
| eval_ppo_original | ppo:round2:e1b284163fc3 | gravity | 8 | 6 | 1 | 8056.75 | 6966.333333333333 |
| eval_ppo_original | ppo:round2:e1b284163fc3 | numeral_system | 8 | 3 | 3 | 4518.75 | 1752 |

输出含Thinking、错误、截断及EOS，不含输入。每类n=8，仅用于Phase C筛查；Wilson与配对检验见JSON。

导出状态：snapshot_incomplete；未完成评测：['eval_base_closed_trace_v1', 'eval_sft_closed_trace_v1', 'eval_ppo_closed_trace_v1']。

缺少真实训练后Checkpoint的方法：['dpo', 'grpo']。不以SFT权重冒充训练结果。

DPO-B无合格偏好，GRPO无组内奖励信号：未满足正式实验门槛，未启动Phase D或锁定final。
