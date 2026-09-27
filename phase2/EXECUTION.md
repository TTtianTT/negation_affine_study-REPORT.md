# 实际执行与关系记录

第一阶段全部已跟踪文件及cache通过逐文件SHA256检查保持原样，见results/validation.json；phase2没有写入第一阶段路径。

- 首先读第一阶段REPORT、PROTOCOL、实际CSV及核心代码，冻结第二阶段协议并保存hash。
- 初稿人名固定划分改为轮转划分，在测试推理前记录修订；384命题及96例C确认集随后冻结。
- GPU job1632：sbatch分配1卡，srun运行extract.py，完成24条开发文本pilot、1632例token解析恒等式检查及4568条新文本提取；54秒。
- 依赖afterok:1632提交因完成作业记录已清除而失败。sacct核验COMPLETED后直接提交无GPU作业1634。
- CPU job1634：srun run_c.py（旧结果重现、新C、C到B迁移）及run_ab.py（A/B全网格）；14分03秒。
- 仅查看token匹配统计后，发现原A三编辑无完全共同支持。独立冻结384条过去标记非否定补充，主B/C与主终点保持不变，补充明确为探索性。
- GPU job1637：1卡、34秒，只提取384条新目标；与1632无重叠。模型/源缓存/环境复用。
- CPU job1638：matched.py、supplement.py、diagnostics.py、summarize.py，完成严格匹配补充、挑战/候选池诊断、分组区间及原始图表。
- 报告及版式脚本在统计完成后编写，用本地CPU轻量渲染，不训练新模型、不改变选择；最终finish.slurm现已串接presentation.py、report.py、validate.py供重现。原1638实际执行的四步与这些后续渲染步骤通过日志区分。
- CPU job1639：最终validate.py核验全部输出数量、原始文件/缓存、冻结数据、候选正例标签、一对多损失恒等式、主效应独立重聚合，全部通过。
- prepare.py最后将清单条目限定为原五个源数据文件，避免重现时误把后续补充/候选文件加入原清单；extract审核门槛允许真实human-reviewed状态。属于复现路径整理，未改变数据或已运行结果。

所有GPU分配总88秒、最大并发1卡；其他作业未申请GPU。详细资源与最终状态见logs/slurm_accounting.psv。没有下载模型、安装依赖、购买算力或调用付费API。

缺少独立真人审核。19组主数据和5组匹配补充是AI助手检查，有单独记录；未伪称人工金标准。没有运行的白化、新模型、生成/因果干预均未填数值。
