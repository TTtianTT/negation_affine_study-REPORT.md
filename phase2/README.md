# Phase 2：否定结构与词汇、长度、池化控制

[中文报告](PHASE2_REPORT.md) · [冻结协议](PHASE2_PROTOCOL.md) · [偏离记录](DEVIATIONS.md) · [完整结果汇总](RESULTS_TABLES.md)

本目录承接第一阶段，但只读使用其数据/缓存/算法。`phase1_snapshot.json`与`results/integrity.json`用于证明原文件未修改。阶段一“第28层last完整仿射44.44%”的重现、独立新命题复核和阶段二重新训练的多表达模型分开保存。

## 实际运行

从仓库根目录运行。复用 `/dataset1/zailong/workspace/outlier_text_pilot/.venv/bin/python` 和 `/dataset1/zailong/models/Qwen2.5-7B-Instruct`，环境版本见`results/environment.json`。无下载、付费API或环境安装。

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/negation-phase2-mpl
PY=/dataset1/zailong/workspace/outlier_text_pilot/.venv/bin/python
# 当前冻结数据已在data/，通常无需重生成。以下命令重建候选数据，但不能伪造审查。
$PY phase2/scripts/prepare.py
# 查看data/review_pending.json，独立审核或明确记录AI审核。
# 提取要求data/review.json各项status为AI-reviewed或实际取得的human-reviewed，当前版本保留实际19组AI审核记录。
sbatch phase2/scripts/extract.slurm
# 等上一个作业成功；下面脚本不申请GPU。
sbatch phase2/scripts/analyze.slurm
# 独立探索性、token完全匹配的非否定补充。
$PY phase2/scripts/prepare_matched.py
# 上一个GPU提取必须已经完成，不同时申请第二张卡。
sbatch phase2/scripts/matched_extract.slurm
# 等主分析和补充提取均成功。
sbatch phase2/scripts/finish.slurm
```

如在作业仍存在时串接依赖，可使用`sbatch --dependency=afterok:<jobid> ...`；本集群已完成作业很快退出可依赖记录，遇到依赖不存在应先核对`sacct`，而非让未完成输入被提前使用。可重复运行纯CPU分析；`run_ab.py`每次清空它自己的追加结果文件并重写phase2结果，不触碰phase1。

只有`extract.slurm`和`matched_extract.slurm`申请一张GPU，其余脚本仅申请4 CPU。最终GPU作业1632、1637顺序执行；CPU分析作业1634、1638，完整性核验1639。总GPU分配88秒，两个任务无重叠。保存的统计来自FP32推理，关闭TF32；不能无记录地改用BF16。异机修改配置的模型路径及Slurm脚本中的Python路径，保持模型revision一致；缺少phase1缓存时应先依第一阶段说明复现其缓存。不要在原实验目录内混入新配置的结果。

冻结数据和协议的hash对应本次实际运行；重新生成或另作修改后，要保存新版本、重新冻结，并重新撰写报告，不能沿用旧解释。独立人工审核尚未获得，不能仅修改状态字段就声称人工金标准。

## 数据与任务

- `data/b_propositions.json`：384基础命题，每条2肯定、5否定及4困难候选，保留主题/谓词族/划分。train96/dev48/iid48/template64/topic96/joint32。训练只使用前3种否定；后2种属于留出的经历否定族。
- `data/candidates.json`：seen/unseen/all的固定候选ID与类别，索引`data/texts.json`。同方法、同查询候选完全一致。
- `data/a_edits.json`：1152个原A编辑（否定、时间替换、强调插入），保存真实token编辑统计。
- `data/a_matched_supplement.json`：384对额外的“V了→曾V”非否定控制，长度、编辑数、位置与否定精确匹配，独立标为探索性补充。
- `data/c_confirmation.json`：96条新考古/出版事件，主题/格式匹配第一阶段joint，原命题未重用。
- `data/challenge.json`：7组量词、情态、双重否定、否定词提及、情感/反义挑战，各自有范围假设；不混算统一准确率。
- `data/review.json`：19组实际AI抽查，无真人评分。
- `protocol_freeze.json`、`data_freeze.json`、`a_matched_freeze.json`：协议与数据冻结证据。

## 实现与完整输出

- `scripts/core.py`只读导入第一阶段`Design`。共享模型拟合三个已知否定表示的等权均值，等价于平方损失下的一对多均值目标；指定形式模型只对训练可见的3个形式分别拟合。
- `results/ab_metrics.csv`：6696行逐seed指标；`ab_summary.csv`及`RESULTS_TABLES.md`为汇总。MSE与多正例检索分开解释。
- `results/b_per_query.csv`：全部304128条查询评估，含候选数、机会水平、赢家类别、相似度间隔；两肯定改写不作为独立样本。
- `results/selection_grid.csv`、`choices.csv`：全部dev选择。主表面基线的选择只来自dev/seen。
- `results/a_embedding_identity.json`、`a_phase1_exact_terms.csv`：真实token embedding解析检验；目标文本是oracle诊断输入，绝非无目标预测器。
- `results/a_per_item.jsonl`、`a_edit_matching.csv`、`a_stratified_metrics.csv`、`a_common_support.json`：原A匹配程度与分层结果。
- `results/matched_metrics.csv`、`matched_per_item.csv`、`matched_spectrum_*.csv`：严格token匹配补充。
- `results/phase1_reproduction.json`：第一阶段重现差异；`c_metrics.csv`/`c_per_item.csv`包含raw及训练均值/主方向消融。
- `results/c_frozen_b_summary.csv`：沿用第一阶段算子在B的边界，与B中重新拟合的第28层结果不同。
- `results/c_expanded_candidates.csv`：同一新C数据，97候选组成消融。
- `results/challenge_outputs.json`：7例挑战逐项结果。
- `results/paired_intervals.csv`：命题内平均改写/seed后，按topic×谓词族或topic聚类的区间。模型始终固定。
- `figures/`：差分谱、编辑比较、多正例检索、复核及相似度变化图。

`cache/`中的二进制表示不提交Git，可按脚本重建；文本token审计保留。结果中NaN代表该项任务不适用（例如指定形式分支没有强行定义检索分数），不是模拟或填充未运行数据。

完成后可单独运行`phase2/scripts/presentation.py`和`phase2/scripts/report.py`重建图表/报告；`phase2/scripts/validate.py`核验完整性。最终版finish.slurm已串接这三步。实际执行时间与脚本整理说明见EXECUTION.md，资源记录见logs/slurm_accounting.psv。

## Git 发布与完整结果恢复

为满足 GitHub 文件大小限制，`b_per_query.csv` 和 `a_per_item.jsonl` 以确定性 gzip 无损压缩格式提交，保留全部记录。克隆后，先在仓库根目录运行：

```bash
python3 phase2/scripts/restore_results.py
```

恢复脚本仅使用 Python 标准库，验证压缩文件及原始文件 SHA-256；若已有不同内容的原始文件则拒绝覆盖。仅验证压缩内容可添加 `--verify-only`。文件大小及校验值见 `results/ARCHIVES.json`。本地原始结果和表示缓存不受发布操作影响。

`ARTIFACT_MANIFEST.json` 保留实验完成时的原始快照；`PUBLICATION_MANIFEST.json` 记录本次 Git 发布文件的字节数和 SHA-256（不含该清单自身）。发布时仅增加压缩包装、恢复说明与脚本，不修改实验数据、协议或数值结果。
