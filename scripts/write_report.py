import csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def load(n):return list(csv.DictReader((R/'results'/n).open()))
m=load('summary.csv');ci=load('paired_cluster_ci.csv');probe=load('probes.csv');surface=load('surface_operator.csv')
def row(method,split,variant='canonical',layer='14',pool='mean'):
 return next(r for r in m if r['method']==method and r['split']==split and r['variant']==variant and r['layer']==layer and r['pool']==pool)
def val(r,k):return float(r[k])
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(z) for z in r)+'|' for r in rows])
assert abs(val(row('lowrank','joint'),'mse')-72.75243239900762)<1e-3, 'Narrative is specific to the frozen pilot; revise it for changed experiments.'
main=[]
for sp in ['iid','topic','template','expression','joint']:
 a,b,c=row('identity',sp),row('shift',sp),row('lowrank',sp);t=next(x for x in ci if x['split']==sp and x['variant']=='canonical')
 main.append([sp,a['n'],f"{val(a,'mse'):.4f}",f"{val(b,'mse'):.4f}",f"{val(c,'mse'):.4f}",f"{100*val(t,'improvement_over_shift'):.2f}% [{100*val(t,'improvement_ci_low'):.2f}, {100*val(t,'improvement_ci_high'):.2f}]",f"{val(b,'recall1'):.3f}/{val(c,'recall1'):.3f}",f"{val(b,'mrr'):.3f}/{val(c,'mrr'):.3f}"])
main_table=table(['测试集','n','恒等MSE','平移MSE','低秩MSE','低秩相对平移误差下降及95%区间','平移/低秩R@1','平移/低秩MRR'],main)
control=[]
for method in ['identity','shift','rank1','lowrank','full_delta','full_affine','shuffled','random_subspace']:
 a,b,c=row(method,'dev'),row(method,'topic'),row(method,'joint')
 control.append([method,f"{val(a,'mse'):.4f}",f"{val(b,'mse'):.4f}",f"{val(c,'mse'):.4f}",f"{val(c,'recall1'):.3f}"])
control_table=table(['方法','dev MSE','topic MSE','joint MSE','joint R@1'],control)
para=[]
for sp in ['iid','topic','template','expression','joint']:
 a,b=row('shift',sp,'paraphrase'),row('lowrank',sp,'paraphrase')
 para.append([sp,f"{val(a,'mse'):.4f}/{val(b,'mse'):.4f}",f"{val(a,'recall1'):.3f}/{val(b,'recall1'):.3f}",f"{val(a,'mrr'):.3f}/{val(b,'mrr'):.3f}"])
para_table=table(['改写测试','平移/低秩MSE','平移/低秩R@1','平移/低秩MRR'],para)
layers=[]
for pool in ['mean','last']:
 for l in ['0','5','9','14','19','23','28']:
  a,b,c=row('lowrank','topic',layer=l,pool=pool),row('lowrank','joint',layer=l,pool=pool),row('full_affine','joint',layer=l,pool=pool)
  pa=np.mean([float(z['accuracy']) for z in probe if z['layer']==l and z['pool']==pool and z['split']=='joint' and z['variant']=='canonical' and z['control']=='real'])
  layers.append([pool,l,f"{val(a,'relative_improvement'):.3f}",f"{val(b,'relative_improvement'):.3f}",f"{val(b,'recall1'):.3f}",f"{val(c,'recall1'):.3f}",f'{pa:.3f}'])
layer_table=table(['池化','层索引','低秩topic误差较恒等下降','低秩joint误差较恒等下降','低秩joint R@1','完整仿射joint R@1','joint否定探针准确率'],layers)
st=[]
for sp in ['iid','topic','template','expression','joint']:
 z=[r for r in surface if r['split']==sp and r['variant']=='canonical' and r['mode']=='length_only'];st.append([sp,f"{val(row('shift',sp),'mse'):.4f}",f"{np.mean([float(r['mse']) for r in z]):.4f}",f"{val(row('lowrank',sp),'mse'):.4f}"])
stable=table(['测试集','固定平移MSE','仅输入长度条件平移MSE','低秩MSE'],st)
report=f'''# 现代LLM隐藏表示中的中文否定：低秩仿射关系探索报告

研究日期：2026-09-28（Asia/Singapore）。本地实际运行；模型为Qwen2.5-7B-Instruct。状态：480对受控中文句子的pilot完成，7个层索引×2种池化×3种训练样本扰动；没有基础模型对照、独立真人审核或自然语料确认实验。

## 结论先行

**低秩修正在本次受控“已经……了→没有……”关系中，确实比固定平移更准确；其优势可迁移到两个未见主题和一种未见话题化句式。但没有获得跨否定表达、稳定保持语义范围的通用否定算子的证据。**

预先指定的第14层均值表示、联合留出测试中，低秩MSE比平移低6.24%，条件聚类95%区间为[3.30%, 9.42%]，支持狭义主假设；然而二者Recall@1均为0，低秩MSE仍比恒等映射高83.29%。这是一项“比一个失效基线少错一些”的统计优势，不能称为否定关系预测成功。

已知“没有”形式、未见音乐/体育主题时，低秩相对平移减少77.15%误差，秩1已经取得大部分收益。但纯embedding层均值也有类似优势，长度条件平移解释了相当部分收益；这些观察支持表面措辞/池化几何是重要因素，**并不证明优势全部来自语义否定，也不证明全是长度所致**。

## 资源、版本与实际完成范围

检查了工作区README、父目录约定、模型缓存、依赖及Slurm。登录节点无本地CUDA，约187GiB主存；现有B300q可运行单卡作业。优先复用本地中文模型，未修改相邻项目或模型。官方模型卡说明其为支持中文的7.61B、28层指令模型；本地hidden_size=3584。[官方模型卡](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

模型revision：`a09a35458c702b33eeacc393d103063234e8bc28`，权重文件逐个SHA256保存在[environment.json](results/environment.json)。固定裸文本输入；不使用chat模板，不添加BOS/EOS，不生成助手前缀。研究对象是冻结LLM隐藏状态，**不是经过检索训练的sentence embedding**。

最初估算：额外下载0、缓存约414MB、单GPU上限1小时，CPU分析几分钟到数十分钟。实际最终缓存约395MiB，最终FP32提取含模型文件哈希约36.1秒、峰值GPU分配28.86GB；CPU主拟合循环每层/池化约9.5–12.7秒，总约3分钟（另有写表等开销）。这些是脚本计时/显存分配，非采购费用或完整集群排队计时。GPU为NVIDIA B300 SXM6 AC。无付费API、无新下载。

初始BF16在批次一致性检查中出现2.66%末层相对差异，因此最终改用FP32、禁用TF32；最终42项单句/混合长度批次对比最大相对差异1.32e-5。不是提高容差后强行通过。Python3.12.3，PyTorch2.11.0+cu128，Transformers4.57.6，NumPy2.3.5，SciPy1.18.1，scikit-learn1.9.1，Matplotlib3.11.2，均只读复用现有环境。

没有同系列基础模型缓存；没有为可选对照下载额外约15GB模型。没有继续扩充同样的模板数量或取全部层：完成pilot后，独立数据质量与语义覆盖比增加重复模板更重要。可执行代码与配置已留存，但未运行的基础模型、chat输入、自然语料、人类确认均没有填写数值。

## 相关工作与问题定位

结构探针通过受限变换检验表示中的结构，不等于肯否句之间存在共享映射。[Hewitt & Manning 2019](https://aclanthology.org/N19-1419/) 探针还可能受容量和控制任务影响。[Hewitt & Liang 2019](https://aclanthology.org/D19-1275/)

LEACE在其形式化条件下消除线性可读概念，不能推导出非线性信息全部消失，或本实验的否定算子存在。[Belrose et al.](https://arxiv.org/abs/2306.03819) 线性表示几何研究区分概念方向、度量和干预，本研究检验的是整个句子池化后的关系预测。[Park et al.](https://arxiv.org/abs/2311.03658)

否定研究已有事实探针、范围推理及显式算子工作。尤其CoNLL2025的Polarity inversion operators in PLM研究BERT上下文token的外部否定/肯定MLP算子，与本题直接相关，不能宣称首次探索。[否定事实探针](https://aclanthology.org/2020.acl-main.698/)、[否定算子](https://aclanthology.org/2025.conll-1.20/)、[CONDAQA](https://aclanthology.org/2022.emnlp-main.598/)。更完整的核查记录见[related_work.md](related_work.md)。本次未使用这些论文的数据作为实际评估集。

## 数据与抽查

10个主题：科研、教学、医疗、物流、餐饮、园艺、音乐、体育、考古、出版。每主题12个事件动作×4个主语描述，共480个独立原命题组。肯否句、肯否改写与相关候选留在同组；原命题组无跨split重复，训练/测试的四种主句文本也无重复。数据由AI助手手写词表与规则生成，没有调用另一个LLM生成API，仍存在设计者偏差。

- train168、dev48来自前6个主题的普通陈述/“昨天”/“上午”句式。
- iid24：新主语—动作组合，但可共享动作与句式，**不能当作严格模板隔离**。
- topic96：完整留出音乐、体育；保留熟悉的否定形式与句式。
- template24：留出宾语话题化句式“关于X，某人已经V了”，原命题组不进训练。
- expression24：完整命题引述后接“并非事实”；训练不含此否定形式。
- joint96：考古、出版两主题，同时使用留出句式与“并非事实”。

各拆分是轴向诊断，不声称每个split同时在所有轴上互斥。训练中的动作语义会以不同主语出现在部分dev/iid/template/expression组中；只有topic/joint完全隔离主题。全部preprocessing、正则化、秩选择只使用train/dev。

抽查18组的四种句子，记录见[audit_sample.json](data/audit_sample.json)。**这是AI助手逐条检查，尚无独立真人审核，未满足真人确认的数据金标准。** 初次抽查发现“并非上午，……”会产生错误范围，统计结果检查前已改成明确否定完整引述命题，并重新提取全部表示；修订与失败日志保留。最终引述否定仍引入长度、引号和元语言结构，因此跨表达失败不能单独归因于某个否定词。

肯定改写用“确实”替代或增补“已经”，否定改写用“并没有”或“这一说法不成立”。它们意图保持事件真假，但时体/语用强调不完全相同，且仍是规则改写，不能当作丰富自然同义语料。

困难检索每例5个候选：正确否定、原肯定、同主题同split另一命题的肯定/否定、量词范围不同的“并非所有人都……”。候选池固定，不用于拟合；不同范围候选是语义不同而不必与目标逻辑互斥。对改写输入，正确候选与原肯定也换成各自改写，其余候选保持固定，因此这是成对改写压力测试，不是全候选统一改写。候选仅5个，常规集出现天花板效应；不能外推到开放库检索。

另外7个挑战例明确区分命题否定、量词、情态、范围、双重否定近等价、反义、情感对立、提及“否定”这个词。后四类的contrast不作为命题否定标签，也不把7例混算一个分类准确率。

## 表示与估计方法

提取索引0、5、9、14、19、23、28。0是token embedding；1–27对应相应block的隐藏输出；28为最终归一化后的输出，尺度与中间层不同。last为最后有效文本token，本数据通常是句号；mean只平均attention_mask内文本token。没有padding、特殊token或助手前缀混入。层0-last因共同句号退化，保留为负对照，零恒等误差的相对改善记为NaN。

采用每坐标MSE `mean((prediction-target)^2)`、相对恒等误差下降、余弦、五候选Recall@1/MRR、正确否定与原肯定的余弦比较。相似度相等按悲观名次处理。不同层尺度不同，跨层优先比较相对恒等改善及检索指标，不直接比较原始MSE。

对行向量，记D=Y−X，训练均值中心化。比较恒等、平均差平移、`x+(xV)Uᵀ+b`低秩修正、对A−I正则的完整修正`full_delta`，以及直接对A正则的`full_affine`。后两者容量相同、先验不同；不能把差异都归因于秩。

低秩求解的是带Frobenius惩罚的精确降秩回归：在增广设计对应的拟合响应上做SVD选择输出子空间，随后构造因子；不是任意截断完整系数。`lambda=alpha*trace(Xc Xcᵀ)/n`，alpha=0.01/0.1/1/10，rank=1/2/4/8/16/32，仅dev MSE选择。闭式解与小维度原始ridge数值一致到3.34e-14；阶数增加的惩罚目标单调下降。

名义参数量：恒等0；平移3584；秩1为10752；秩32为232960；完整仿射12848640。因子参数有旋转冗余，名义数未扣掉r²自由度。固定随机输出子空间只有d*r+d可训练参数。训练样本只有168，完整矩阵的高维能力主要受ridge与样本张成空间约束；其数值表现相近不等于总体高维算子已识别。

三个seed17/29/43对训练原命题组有放回重采样，固定dev/test，量化训练样本扰动；不是三个独立LLM训练。全部三个seed在主位置选中r=32、alpha=.01，均位于网格边界，因此**只能称本次候选中的最优，不能声称发现了内在秩32**。权重因子、选择与完整网格均保存。

主假设事前固定为14层mean、joint上的低秩MSE优于平移。置信区间先平均seed，再对topic×action组进行2000次配对bootstrap。joint含24个动作组，但只有2个主题、1种留出句式：这些区间条件于所构造主题/模板，不是对独立主题总体的置信保证。三seed不作为三个独立测试样本，也不做挑最优层后的显著性宣称。

## 实际主结果与统计证据

下表为三seed均值。区间为低秩相对平移的误差下降，正值较好；其余测试区间为探索性、未校正多重比较。

{main_table}

主要配对MSE差（低秩减平移）为−4.8428，95%条件聚类区间[−7.0387,−2.6399]。joint平移MSE的seed标准差1.1464，低秩1.5556；固定seed重采样结果相近，但没有独立跨主题重复验证。expression差为+0.1002、区间跨零，未观察到可靠改善。

常规topic/template上，平移与低秩的R@1已都达到1；低秩的向量误差改善是真实观察，但不意味着这些候选下的检索更优。joint中正确否定胜过原肯定的比例两者均为0；平均余弦仍分别约0.999658与0.999654，展示了只报告高余弦会严重误导。甚至低秩MSE较小而余弦略低，指标并不等价。

完整对照：

{control_table}

低秩与完整delta很接近。打乱配对后误差大幅增加，表明真实配对含有可学习结构；随机同秩子空间只接近平移，说明学习的方向并非任意方向。不过打乱配对仍可能在常规小候选池中获得不错R@1，它保留了平均负向变化，所以不能以单一检索分数宣称对照彻底失效。

训练原168例上的主位置MSE为平移1.1566、低秩0.2016、完整delta0.2005；dev分别0.8323、0.3228、0.3221。具体bootstrap袋内训练误差见grid.csv，和对原168例的train汇总不同。训练误差优势并未转化为跨表达成功，主要问题包含分布变化，而不只是训练/验证差距。

## 秩、层与可读出信息

在topic上，r=1的MSE0.2826，r=32为0.2766，完整delta0.2762；秩1已经接近完整对照。template上r=1为0.0984、r=32为0.0966。继续增秩收益很小，即使dev选择了32。

训练差分的未中心化SVD第一方向解释99.71%平方奇异值能量；中心化后第一主成分仍解释98.41%变异。这说明该受控任务中共同方向以及该方向的幅度变化都很强；**低维差分并不自动意味着能从X预测该差分，更不自动意味着这个方向是抽象语义否定**。具体谱只使用训练数据。

![秩—性能曲线](figures/rank_performance.png)

![差分SVD与中心化PCA](figures/singular_spectrum.png)

下表完整展示预指定层网格，不挑选一个“获胜层”代替主位置。误差下降为相对恒等，负值代表更差；层0-last的NaN是零基线误差而非遗漏实验。

{layer_table}

描述性观察：mean在熟悉表达的topic上，很多层都很好，甚至无上下文embedding层也达到高性能；上下文层中23层的相对误差下降较大，但不是独立确认的“最优层”。last在早中层对熟悉形式的关系预测较好；对joint，最后层低秩只取得约2.43% R@1，完整仿射达到44.44%，说明容量/正则化先验与评估指标会改变排序。后者是值得复核的探索性发现，不能将所有完整算子都概括为完全无效，也不能称已经可靠泛化。

**分类探针与算子明显分离**：last的9/14/19/23层在joint分类均为100%，但14层last低秩joint R@1为0，MSE甚至略差于平移。这直接说明“可线性读出否定标签”不等于“能够线性预测相应否定句表示”。mean在熟悉形式分类100%，在新表达却为50%。分类同样可能使用句尾“并非事实”等线索，100%不是理解范围的证据。

![层位置—预测性能](figures/layer_performance.png)

![独立线性探针](figures/linear_probe.png)

## 措辞、长度与语义的区分

词汇/模板/长度逻辑回归在熟悉否定形式的iid、topic、template上均100%，仅长度在topic约55.73%。14层mean真实探针topic为100%，乱训练标签对照平均63.54%（不是人为写成50%）；少seed、共同模板和验证选择使乱标签控制存在较大偶然波动，不能假定每次恰为机会水平。新表达mean真实探针50%，缺乏强泛化。

观察主表后追加的**事后探索性**条件平移仅用输入token长度、倒数和字符长度，不用目标长度或目标表示特征；仍只在train拟合、dev选alpha：

{stable}

长度条件平移已解释不少固定平移的不足，且在joint误差甚至低于低秩算子。这支持“均值池化和长度规律可能解释部分收益”的判断，但没有严格分解因果贡献。纯embedding均值层也能拟合关系，更削弱将几何低秩直接解释为高层语义的论证。尚未运行长度完全匹配、否定词位置/频率平衡、丰富自然改写、非否定伪编辑等足以排除所有混淆的实验。

同义改写压力测试：

{para_table}

topic改写下低秩R@1由原句的100%降至76.39%，但仍优于平移54.17%。template改写下低秩MSE比平移更差，检索反而更好，提示欧氏几何与候选排序的评价目标分离。跨表达及联合改写检索仍为0。这些结果不能支持“跨措辞完全不变”的关系。

## 具体成功与失败

以下固定使用seed17、14层mean；按split顺序选择首个符合成功/失败条件的实例，只作说明，不作为抽样准确率。

- 成功（topic，06-00）：“那位乐团成员已经整理了演出乐谱。”→“那位乐团成员没有整理演出乐谱。”低秩正确否定排第1，MSE0.0684。
- 改写失败（topic，06-11）：“上午，那位乐团成员确实清点了舞台器材。”→“上午，那位乐团成员并没有清点舞台器材。”正确否定排第2，首选原肯定，MSE2.4374。
- 指标分离（template，00-40）：“异常数据，这位研究员确实已经标记了。”→“异常数据，这位研究员并没有标记。”检索排第1，但MSE11.7965，说明正确排序并不保证准确预测向量。
- 联合失败（08-00）：“关于测绘设备，负责此事的考古队员已经检查了。”→“‘关于测绘设备，负责此事的考古队员已经检查了’并非事实。”正确否定排第3，首选原肯定，MSE49.3483。

挑战集seed17低秩对7个示例均最接近原肯定候选；不把这个小样本当作总体正确率。“所有学生都交了作业”的否定应允许部分未交，而不是“所有学生都没有交”；“必须参加”的否定是“不必参加”，不是“必须不参加”；“并非没有完成”接近原肯定，不能贴上负标签。实际算子未在这些例子上可靠找到正确命题否定候选。反义词“冷/热”、情感“喜欢/讨厌”和提及“否定”一词都单独记录，未混进训练标签。

## 子空间移除诊断

仅对14层mean的缓存表示做正交投影，未改变LLM或执行生成干预。去掉训练未中心化差分的首方向，topic冻结探针从100%降至50%，重新拟合后约65.63%；去掉8方向后重新拟合约53.13%。肯定改写到同split原肯定句的内容检索仍约100%。同维随机移除结果见removal.csv。

这说明所选方向与本任务的分类信号相关，并且在这批短模板内内容检索较稳健。**不构成严格因果证据，不说明全部否定信息已删除。** 事实上某些新表达在移除首方向、重拟合后准确率反而提高（expression约89.58%），说明读出、正则化与分布变化之间的关系并不简单。该诊断用完整168例训练，和主实验bootstrap不同。

## 局限与下一步

1. 数据为单一完成事件骨架及规则变体；训练核心否定是“没有”。双重否定、量词、情态只有7例挑战中的少量实例，不支持广泛语言学结论。AI抽查不替代独立真人审核。
2. 新表达包含引述和句长变化，joint还同时变换主题与句式；当前只能说明对这种复合变化的边界，不能定位失败的唯一因素。输入裸文本，未比较chat模板；仅一个指令模型，无基础模型/其他规模证据。
3. 句子可有多个同义否定表达；固定从一个肯定表示预测某一特定否定措辞的隐藏表示，本身含有“目标措辞选择”问题。一个统一算子无法凭空知道要求哪种等价表达，这个不可识别性必须与语义否定能力分开。
4. 候选池小且常规测试饱和，训练样本小而维度高，rank/alpha选择在网格边界；有限seed、2个新主题和共享模板限制统计外推。没有独立确认集，不应将多层多指标探索转述成已确认最优层。
5. 没有测试生成后的真实性、推理表现、算子反复作用是否回到原句、双向一致性或组合性；表示拟合不代表改变模型信念，也不代表实际推理增强。

优先下一步是独立审核自然中文命题及范围，构建词汇/长度/位置平衡的非否定编辑对照；让训练覆盖多种否定表达，并按语义模板族、主题、表达三轴做多折留出。以肯否各多种改写的等价类进行多正例检索，隔离语义与目标措辞。冻结本次发现后，再用新主题确认集和同系列基础模型复核，尤其验证最后层完整仿射的检索优势。不宜只增加同一规则下的句对数量。

## 交付与复现

[README运行说明](README.md)给出从数据生成到GPU提取、CPU分析和图表的命令。[PROTOCOL](PROTOCOL.md)保留事前主假设与修订；[执行记录](EXECUTION_NOTES.md)披露数据/数值/输出字段故障与修复。

完整结果：[逐seed指标6552行](results/metrics.csv)、[汇总表](RESULTS_TABLES.md)、[参数网格1176行](results/grid.csv)、[配对区间](results/paired_cluster_ci.csv)、[探针1176行](results/probes.csv)、[表面分类](results/surface.csv)、[长度条件算子](results/surface_operator.csv)、[移除实验](results/removal.csv)。所有层、两种池化和失败结果均保留；没有填充未运行结果。[数值/分组验证](results/validation.json)、[完整性核验](results/completeness.json)。

本次最有依据的结论是：**低秩、甚至秩1的输入依赖修正能描述一类受控肯否措辞变化，比固定平移更精确；现有证据不足以将它认定为跨主题、跨句式、跨表达的通用语义否定算子。**
'''
(R/'REPORT.md').write_text(report)
print('Wrote REPORT.md',len(report))
