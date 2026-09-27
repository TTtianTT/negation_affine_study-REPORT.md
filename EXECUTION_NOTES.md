# 执行记录

- 检查工作区根说明/AGENTS：根与父目录没有AGENTS.md；没有更改相邻研究项目。
- 登录节点187GiB RAM，约179GiB可用，无nvidia-smi；Slurm查询起初被沙箱网络阻止，经标准执行批准后使用现有B300q。
- 找到Qwen2.5-7B-Instruct、Qwen3-8B、Llama-3.1-8B-Instruct缓存，无同系列基础模型缓存。选前者是中文支持、已有权重、28层规模适合本pilot；未下载基础模型。
- 主环境缺scipy，已有encrypted项目环境缺matplotlib；最终只读复用outlier项目环境，未pip安装。
- 新目录negation_affine_study，全部新增数据、代码、缓存、日志位于其中。未上传、发布、发消息或调用付费API。
- 初次BF16提取成功，随后数据抽查发现否定范围错误。停止初次分析（只完成层0 last日志），修订数据，未查看该次性能数值。
- 第二次BF16提取触发padding检查2.66%相对误差，保留logs/extract_bf16_padding_failure.log；该次未写入缓存。
- 最终FP32提取、关闭TF32，通过1e-4阈值；results/environment.json含计时、显存、版本、模型文件SHA256。
- CPU分析第一次在保存第14层逐样本时遇到operator rank与检索rank字段冲突；修正后从头重跑，最终输出均由完整运行覆盖。纯输出字段错误，未据测试性能调整模型。
- 名义参数量：identity0；shift3584；rank r为2*3584*r+3584（不扣因子不可识别自由度）；固定随机输出子空间的可训练参数为3584*r+3584；full为3584²+3584。随机子空间最终参数字段由汇总核验校正（拟合不变）。
- 实际命令见README；运行日志extract.log/analyze.log/removal.log。单GPU，未扩大到所有层或基础模型；模板广度和独立人工审核比继续扩充模板实例更值得优先解决。
