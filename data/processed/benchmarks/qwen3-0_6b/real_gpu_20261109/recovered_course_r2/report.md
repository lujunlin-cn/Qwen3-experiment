# Qwen3 课程实验结果

状态：incomplete。显存阈值：8,000,000,000 bytes；设备峰值为采样值。

| 对照 | 题型 | n | 正确率 | 截断率 | 平均输出 Token | 设备峰值 GiB | 秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| base_reference | equation_numeric | 4 | 25.0% | 25.0% | 12874.00 | 4.84 | 466.2 |
| base_reference | gravity | 4 | 100.0% | 0.0% | 9407.25 | 4.84 | 466.2 |
| base_reference | numeral_system | 4 | 75.0% | 0.0% | 2137.50 | 4.84 | 466.2 |
| base_cap512 | equation_numeric | 4 | 0.0% | 100.0% | 512.00 | 4.84 | 43.8 |
| base_cap512 | gravity | 4 | 0.0% | 100.0% | 512.00 | 4.84 | 43.8 |
| base_cap512 | numeral_system | 4 | 0.0% | 100.0% | 512.00 | 4.84 | 43.8 |
| base_cap2048 | equation_numeric | 4 | 0.0% | 100.0% | 2048.00 | 4.84 | 110.3 |
| base_cap2048 | gravity | 4 | 0.0% | 100.0% | 2048.00 | 4.84 | 110.3 |
| base_cap2048 | numeral_system | 4 | 50.0% | 50.0% | 1809.00 | 4.84 | 110.3 |
| base_cap8192 | equation_numeric | 4 | 0.0% | 75.0% | 8088.00 | 4.84 | 325.3 |
| base_cap8192 | gravity | 4 | 50.0% | 50.0% | 7969.25 | 4.84 | 325.3 |
| base_cap8192 | numeral_system | 4 | 75.0% | 0.0% | 2137.50 | 4.84 | 325.3 |
| base_cap16384 | equation_numeric | 4 | 25.0% | 25.0% | 12874.00 | 4.84 | 463.4 |
| base_cap16384 | gravity | 4 | 100.0% | 0.0% | 9407.25 | 4.84 | 463.4 |
| base_cap16384 | numeral_system | 4 | 75.0% | 0.0% | 2137.50 | 4.84 | 463.4 |
| prior_reference | equation_numeric | 4 | 0.0% | 0.0% | 8696.00 | 4.84 | 338.4 |
| prior_reference | gravity | 4 | 50.0% | 0.0% | 7821.75 | 4.84 | 338.4 |
| prior_reference | numeral_system | 4 | 100.0% | 0.0% | 1757.25 | 4.84 | 338.4 |
| sft_r1_validation | equation_numeric | 4 | 25.0% | 0.0% | 9485.50 | 4.84 | 452.3 |
| sft_r1_validation | gravity | 4 | 100.0% | 0.0% | 7882.75 | 4.84 | 452.3 |
| sft_r1_validation | numeral_system | 4 | 100.0% | 0.0% | 2225.50 | 4.84 | 452.3 |
| sft_r2_validation | equation_numeric | 4 | 25.0% | 0.0% | 11616.75 | 4.84 | 511.5 |
| sft_r2_validation | gravity | 4 | 100.0% | 0.0% | 6888.00 | 4.84 | 511.5 |
| sft_r2_validation | numeral_system | 4 | 50.0% | 25.0% | 4398.50 | 4.84 | 511.5 |
| sft_r3_validation | equation_numeric | 4 | 0.0% | 0.0% | 7867.50 | 4.84 | 378.0 |
| sft_r3_validation | gravity | 4 | 100.0% | 0.0% | 6209.50 | 4.84 | 378.0 |
| sft_r3_validation | numeral_system | 4 | 25.0% | 0.0% | 3233.25 | 4.84 | 378.0 |
| dpo_r1_rollout | equation_numeric | 16 | 6.2% | 93.8% | 3847.19 | 4.84 | 596.9 |
| dpo_r1_rollout | gravity | 16 | 6.2% | 75.0% | 3208.44 | 4.84 | 596.9 |
| dpo_r1_validation | equation_numeric | 4 | 25.0% | 0.0% | 10613.25 | 4.84 | 456.8 |
| dpo_r1_validation | gravity | 4 | 100.0% | 0.0% | 7750.25 | 4.84 | 456.8 |
| dpo_r1_validation | numeral_system | 4 | 75.0% | 0.0% | 1895.00 | 4.84 | 456.8 |
| dpo_r2_rollout | equation_numeric | 16 | 0.0% | 87.5% | 3613.31 | 4.84 | 552.3 |
| dpo_r2_rollout | gravity | 16 | 12.5% | 62.5% | 3140.69 | 4.84 | 552.3 |
| dpo_r2_validation | equation_numeric | 4 | 25.0% | 25.0% | 11570.00 | 4.84 | 467.8 |
| dpo_r2_validation | gravity | 4 | 75.0% | 0.0% | 6926.25 | 4.84 | 467.8 |
| dpo_r2_validation | numeral_system | 4 | 75.0% | 0.0% | 2348.75 | 4.84 | 467.8 |
| dpo_r3_rollout | equation_numeric | 16 | 12.5% | 81.2% | 3582.12 | 4.84 | 563.5 |
| dpo_r3_rollout | gravity | 16 | 18.8% | 68.8% | 3312.25 | 4.84 | 563.5 |
| dpo_r3_validation | equation_numeric | 4 | 25.0% | 0.0% | 10927.00 | 4.84 | 485.2 |
| dpo_r3_validation | gravity | 4 | 100.0% | 0.0% | 7117.25 | 4.84 | 485.2 |
| dpo_r3_validation | numeral_system | 4 | 75.0% | 0.0% | 3330.50 | 4.84 | 485.2 |
| ppo_r1_rollout | equation_numeric | 8 | 0.0% | 100.0% | 4096.00 | 4.84 | 182.7 |
| ppo_r1_validation | equation_numeric | 4 | 25.0% | 0.0% | 12311.75 | 4.84 | 544.2 |
| ppo_r1_validation | gravity | 4 | 75.0% | 25.0% | 9607.00 | 4.84 | 544.2 |
| ppo_r1_validation | numeral_system | 4 | 25.0% | 0.0% | 1807.50 | 4.84 | 544.2 |
| ppo_r2_rollout | equation_numeric | 8 | 0.0% | 75.0% | 7230.25 | 4.84 | 325.9 |
| ppo_r2_validation | equation_numeric | 4 | 25.0% | 0.0% | 11199.25 | 4.84 | 473.5 |
| ppo_r2_validation | gravity | 4 | 75.0% | 0.0% | 6866.25 | 4.84 | 473.5 |
| ppo_r2_validation | numeral_system | 4 | 75.0% | 0.0% | 3190.00 | 4.84 | 473.5 |
| ppo_r3_rollout | equation_numeric | 8 | 37.5% | 0.0% | 10018.12 | 4.84 | 466.3 |
| ppo_r3_validation | equation_numeric | 4 | 25.0% | 0.0% | 10150.00 | 4.84 | 389.9 |
| ppo_r3_validation | gravity | 4 | 100.0% | 0.0% | 5999.00 | 4.84 | 389.9 |
| ppo_r3_validation | numeral_system | 4 | 100.0% | 0.0% | 1259.50 | 4.84 | 389.9 |
| grpo_r1_rollout | equation_numeric | 16 | 0.0% | 87.5% | 3724.31 | 4.84 | 316.7 |
| grpo_r2_rollout | equation_numeric | 16 | 6.2% | 81.2% | 7637.62 | 4.84 | 670.1 |
| grpo_r2_validation | equation_numeric | 4 | 25.0% | 0.0% | 9197.00 | 4.84 | 524.9 |
| grpo_r2_validation | gravity | 4 | 75.0% | 25.0% | 8938.25 | 4.84 | 524.9 |
| grpo_r2_validation | numeral_system | 4 | 50.0% | 25.0% | 5010.50 | 4.84 | 524.9 |
| grpo_r3_rollout | equation_numeric | 16 | 0.0% | 93.8% | 7762.19 | 4.84 | 678.6 |
| grpo_r4_rollout | equation_numeric | 16 | 25.0% | 0.0% | 9996.69 | 4.84 | 903.6 |
| grpo_r4_validation | equation_numeric | 4 | 0.0% | 0.0% | 11129.50 | 4.84 | 457.1 |
| grpo_r4_validation | gravity | 4 | 100.0% | 0.0% | 4760.25 | 4.84 | 457.1 |
| grpo_r4_validation | numeral_system | 4 | 75.0% | 25.0% | 4487.75 | 4.84 | 457.1 |
| grpo_r5_rollout | equation_numeric | 16 | 0.0% | 0.0% | 11480.69 | 4.84 | 1038.5 |

输出长度包含 Thinking 和所有生成 Token，包括截断样本；不含输入。小面板仅作课程筛查，Wilson 区间见 JSON。

停止条件与耗时（包括 rollout、train、验证、启动和保存；未达到平台期时不称为收敛）：

- sft: validation_plateau, 1714.7 秒
- dpo: validation_plateau, 3177.3 秒
- ppo: validation_plateau, 2512.7 秒
- grpo: rollout_incomplete, 5400.3 秒

缺少最终对照：['final_base_original', 'final_sft_original', 'final_dpo_original', 'final_ppo_original', 'final_grpo_original', 'final_base_prior', 'final_sft_prior', 'final_dpo_prior', 'final_ppo_prior', 'final_grpo_prior']
缺少基线/先验对照：[]
失败阶段数：2。原始日志与显存采样保存在 jobs/。

SFT 和 RL 的长度/截断变化均按实测报告；不预设只有 RL 才能改善这些指标。
