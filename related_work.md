# 相关工作核查
检索日期：2026-09-28。下列均为作者论文、会议原文或官方模型卡；不是系统综述，不宣称新方向。

- [Hewitt & Manning 2019，A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419/)：学习几何变换以恢复句法距离/深度，说明结构假设需要明确指标；不证明肯否句之间有共享算子。
- [Hewitt & Liang 2019，Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/)：探针容量/记忆可能影响解释，本实验独立报告分类并加乱标签对照。
- [Belrose et al. 2023，LEACE](https://arxiv.org/abs/2306.03819)：在其形式化条件下闭式消除线性可读概念；不是删除全部非线性信息的保证，也不是句子关系算子存在性的证据。
- [Park, Choe & Veitch，The Linear Representation Hypothesis and the Geometry of Large Language Models](https://arxiv.org/abs/2311.03658)：区分表示/度量/干预几何；概念方向与整个句子的肯否仿射映射不同。
- [Kassner & Schütze 2020，Negated and Misprimed Probes for Pretrained Language Models](https://aclanthology.org/2020.acl-main.698/)：说明语言模型的事实探测对否定存在困难；不能从文本相近推断语义极性已建模。
- [Polarity inversion operators in PLM，CoNLL 2025](https://aclanthology.org/2025.conll-1.20/)：直接研究BERT上下文token的否定/肯定算子，训练外部MLP，递归应用未能充分恢复原极性；与本题直接相关。本文是中文指令LLM句子池化与受限仿射算子的有限探索，不能宣称首次研究否定变换。
- [CONDAQA，EMNLP 2022](https://aclanthology.org/2022.emnlp-main.598/)：通过对比编辑评估否定推理，支持范围/改写对照的重要性；本实验没有下载或使用该英语数据集。
- [Qwen2.5-7B-Instruct官方模型卡](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)：7.61B、28层，支持中文；本地配置hidden_size3584。选择基于已有缓存、中文能力和资源适配。官方推荐chat输入，本研究固定裸文本来明确文本边界，结论局限于该输入条件。

来源只用于方法定位；本研究数值均来自本地实际运行，不取自上述论文。论文关于其他模型或token层面的发现不能直接外推到本研究。
