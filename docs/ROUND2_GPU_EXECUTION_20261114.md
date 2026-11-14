# 第二轮真实 GPU 执行记录（2026-11-14）

用户已提供41522端口并明确要求开展测试，审核稿中的等待审核标记由本次显式授权取代。实际执行注册为独立 `qwen3-round2-gpu/v1` 协议，不修改第一轮source manifest或历史Checkpoint。

服务器为空数据盘；核实GPU为RTX4080 SUPER，32760MiB，初始仅1MiB。同步Git已提交代码与第一轮SFT r1 Adapter，权重SHA256匹配 `d565b67a4e09d41b260b3929d5f3600b96e78ea8e7a8f3712dbfc749c190d371`。模型下载按固定revision校验，依赖从阿里镜像安装；vLLM/训练环境共享系统已有Torch2.8.0+cu128但其他依赖隔离安装，不升级系统服务环境。

维护者执行工具：`utils/round2_run.py`注册和监督独立算法Run；`utils/round2_nemo_gpu.py`执行R0/A/B/C。教学算法仍来自Notebook。新Runner保持注册输入Hash、独立进程组显存守卫、各阶段预算与失败保留；真实推理逐题保存partial evidence，只有全部完成才写完整rows/result。训练保存logprob probe，独立进程重载Adapter/PPO Value Head核对。

初始短程顺序为NeMo四组、Equation readiness、Gravity DPO-B、Equation PPO/GRPO各至多两轮（重新采样以验证同算法恢复）、三任务8题的五模型配对评测。PPO/GRPO只使用R1=F+.20P+.05S，长度权重0；不运行额外奖励消融。无合格偏好/组信号时记录0更新，不凑数据。没有实际Checkpoint的方法不会以SFT权重冒充训练后策略。

显存阈值8,000,000,000 bytes，轮询0.25秒；记录设备峰值及Torch峰值，不声称连续瞬时上界。工作队列预算初设8小时，不从安装时间扣除；不自动覆盖、延长deadline或更改cap。剩余预算与Phase C结果决定是否有条件进入扩大评测；未满足门槛不启动正式长训。

最终以新Suite原始日志、guard、rows、Reward与Preference审计、Checkpoint/reload manifest和报告为准；本文件当前记录执行设计，不是成功结论。
