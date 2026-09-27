# 中文否定关系的低秩仿射研究

主报告：[REPORT.md](REPORT.md)。完整表：[RESULTS_TABLES.md](RESULTS_TABLES.md)。方法冻结及修订：[PROTOCOL.md](PROTOCOL.md)。一手来源：[related_work.md](related_work.md)。

所有工作仅位于本目录；无付费API，无新权重下载，无修改已有模型或环境。

## 当前机器复现
已有环境：`/dataset1/zailong/workspace/outlier_text_pilot/.venv/bin/python`，只读复用。依赖版本见requirements.txt、results/environment.json。模型本地路径与revision在config.json，逐文件SHA256在environment.json。

从本研究目录运行（GPU需通过Slurm分配）：

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/negation-mpl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/dataset1/zailong/workspace/outlier_text_pilot/.venv/bin/python
$PY scripts/prepare.py
srun --partition=B300q --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=01:00:00 bash scripts/run_gpu.sh
$PY scripts/check.py
$PY scripts/analyze.py
$PY scripts/surface.py
$PY scripts/removal.py
$PY scripts/length_operator.py
$PY scripts/final_audit.py
$PY scripts/summarize.py
$PY scripts/write_report.py
```

若缓存已经存在且不变，可直接从check/analyze开始；analyze核验文本hash，禁止混用旧文本缓存。surface.py创建待审抽查样本，重跑会把AI抽查状态重置为pending；当前已审记录请先另存。

异机：建立隔离Python3.12环境并安装requirements.txt，选择兼容硬件的PyTorch构建。下载`Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`后，把config.json的model改为本地路径（prepare.py重新生成配置会覆盖修改）。修改run_gpu.sh中的工作目录与Python路径；没有Slurm时，在已有CUDA设备上直接运行extract.py。脚本不自动下载。预留约32GB显存、48GB主存和1GB输出磁盘；模型文件额外约15.3GB。

## 文件含义

- data/pairs.json：480个独立原命题组，肯否与改写、主题、模板、表达、split、困难候选。
- data/challenge.json：范围、量词、情态、双重否定、反义、情感、仅包含否定词，**不能合成一个正负标签**。
- data/audit_sample.json：18组AI抽查记录；没有独立真人审核。
- cache/l*_*.npy：2061条文本×3584维FP32，排序对应texts.json。LLM隐藏状态，非检索专用embedding。
- cache/operator_*.npz：低秩拟合系数，行向量约定`y=x+(x@input_factor)@output_factor+bias`。参数未改动LLM。
- results/metrics.csv：所有层、池化、seed、方法、split、变体的实际评估。
- results/grid.csv：每个秩/正则化的训练bootstrap样本误差与dev误差。
- results/choices.csv：仅由dev选择的配置；rank方法各自用dev选择alpha。
- results/probes.csv、surface.csv：隐藏状态探针及长度/词汇模板探针。
- results/spectra.csv：仅训练集差分的未中心化SVD与中心化PCA。
- results/primary_pairs.csv、paired_cluster_ci.csv：预指定层/池化逐样本结果与聚类区间。
- results/challenge_seed*.json、removal.csv：明确为探索性诊断。
- figures/：层曲线、秩曲线、奇异值谱、探针曲线、泛化差距。
- results/validation.json、extraction_checks.json：分组、闭式解、padding检查。

三seed为对训练原命题组的有放回重采样；不是三个独立LLM，也不是独立测试集。置信区间按topic×action聚类并先平均seed；仅反映这批模板内的不确定性。报告中的小数来自运行输出，不含模拟研究结果。check.py中的随机矩阵只用于数学实现单元检验。

报告生成脚本含本次冻结pilot的中文解释，并检查关键数值；若更换模型、数据或划分，必须另建结果目录并重新撰写解释，不能沿用本报告结论。ARTIFACT_MANIFEST.json记录交付时文件哈希。

## Git 仓库收录范围

Git 收录报告、代码、配置、数据与划分、全部结果表、图表、运行日志和文本 token 审计。约462 MB的可再生成二进制表示缓存及算子文件（`cache/*.npy`、`cache/operator_*.npz`）留在原机器，由`.gitignore`排除；克隆后先执行表示提取和分析以重建这些文件，不需要Git LFS。模型权重和Python环境不随仓库分发。

`ARTIFACT_MANIFEST.json`是首次本地交付的完整快照清单，含未纳入Git的缓存，并保留当时文件的哈希（发布时更新了README及个别代码空行）；`PUBLICATION_MANIFEST.json`记录本次Git提交文件的大小和SHA256（不包含清单自身）。两者用途不同。
