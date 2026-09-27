import json,random,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def save(name,x): (R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
# Authored controlled Chinese events; no external dataset and no generator-model API.
topics={
'科研':('研究员','整理:实验记录 核对:测量数据 提交:研究报告 校准:测量仪器 检查:实验设备 保存:实验样本 更新:分析程序 标记:异常数据 复核:统计结果 记录:实验温度 清理:实验台面 备份:原始数据'),
'教学':('教师','批改:数学作业 整理:课堂笔记 检查:考试试卷 更新:课程安排 提交:教学计划 打印:复习资料 记录:学生成绩 准备:课堂练习 核对:学生名单 收集:学习反馈 修改:课程讲义 保存:教学录像'),
'医疗':('医护人员','核对:患者姓名 记录:体温数据 整理:检查报告 检查:医疗设备 更新:护理记录 提交:值班报告 清点:医疗用品 打印:检查单据 保存:检测样本 标记:采样时间 确认:预约信息 清洁:检查台面'),
'物流':('工作人员','核对:收货地址 整理:快递包裹 检查:运输车辆 更新:配送路线 提交:运输单据 打印:快递标签 记录:发货时间 清点:仓库库存 确认:订单信息 保存:签收记录 标记:易碎货物 清理:装货区域'),
'餐饮':('厨师','清洗:新鲜蔬菜 检查:厨房设备 整理:采购清单 更新:当天菜单 记录:冷藏温度 准备:午餐食材 核对:预订信息 清点:厨房餐具 保存:采购票据 清理:厨房台面 标记:食材日期 提交:采购申请'),
'园艺':('园艺师','修剪:花园树枝 检查:灌溉设备 整理:园艺工具 记录:土壤湿度 清理:花坛杂草 准备:育苗材料 标记:植物名称 更新:养护记录 核对:种子数量 提交:种植计划 保存:土壤样本 清洗:育苗容器'),
'音乐':('乐团成员','整理:演出乐谱 检查:录音设备 记录:排练时间 更新:演出安排 核对:曲目清单 准备:演奏乐器 提交:演出申请 保存:排练录音 清点:舞台器材 打印:音乐节目单 标记:乐谱页码 清理:排练场地'),
'体育':('教练','整理:训练器材 检查:场地设施 记录:比赛成绩 更新:训练计划 核对:参赛名单 准备:训练用球 提交:参赛申请 保存:比赛录像 清点:运动装备 打印:比赛日程 标记:跑道位置 清理:训练场地'),
'考古':('考古队员','整理:发掘记录 检查:测绘设备 记录:地层深度 更新:遗址地图 核对:文物编号 准备:发掘工具 提交:考察报告 保存:陶器碎片 清点:出土文物 打印:遗址图纸 标记:发掘位置 清理:工作台面'),
'出版':('编辑','整理:作者稿件 检查:排版文件 记录:修改意见 更新:出版计划 核对:参考文献 准备:校对材料 提交:审稿报告 保存:原始插图 清点:样书数量 打印:书稿目录 标记:校对位置 清理:文件目录')}
rows=[]
for ti,(topic,(role,actions)) in enumerate(topics.items()):
    combos=[(a,k) for a in actions.split() for k in range(4)]
    random.Random(20260928+ti).shuffle(combos)
    for j,(action,k) in enumerate(combos):
        v,o=action.split(':'); s=['这位','那位','新来的','负责此事的'][k]+role
        split=('train' if j<28 else 'dev' if j<36 else 'iid' if j<40 else 'template' if j<44 else 'expression') if ti<6 else ('topic' if ti<8 else 'joint')
        t=j%3 if split not in ['template','joint'] else 3
        prefix=['','昨天，','上午，'][t] if t<3 else ''
        p=prefix+s+'已经'+v+'了'+o+'。'; n=prefix+s+'没有'+v+o+'。'
        pp=prefix+s+'确实'+v+'了'+o+'。'; nn=prefix+s+'并没有'+v+o+'。'
        if t==3:
            p=f'关于{o}，{s}已经{v}了。'; n=f'关于{o}，{s}没有{v}。'
            pp=f'{o}，{s}确实已经{v}了。'; nn=f'{o}，{s}并没有{v}。'
        if split in ['expression','joint']: n=f'“{p[:-1]}”并非事实。'; nn=f'“{pp[:-1]}”这一说法不成立。'
        rows.append(dict(id=f'{ti:02d}-{j:02d}',group=f'{ti}-{action}-{k}',topic=topic,action=action,template=t,expression='并非' if split in ['expression','joint'] else '没有',split=split,pos=p,neg=n,pos_para=pp,neg_para=nn,scope=f'并非所有{role}都{v}了{o}。',wrong=f'{s}没有{v}另一份{o}。'))
# wrong-object distractors use another natural object from same domain, avoiding classifier mismatches.
for r in rows:
    same=[q for q in rows if q['topic']==r['topic'] and q['action']!=r['action'] and q['split']==r['split']]
    r['other_id']=same[0]['id']; r['wrong']=same[0]['neg']
assert len({r['group'] for r in rows})==len(rows)
save('data/pairs.json',rows)
challenge=[
('量词','所有学生都交了作业。','并非所有学生都交了作业。','所有学生都没有交作业。'),
('情态','小林必须参加会议。','小林不必参加会议。','小林必须不参加会议。'),
('范围','小林只阅读了报告。','小林并非只阅读了报告。','小林没有阅读报告。'),
('双重否定','小林完成了任务。','小林并非没有完成任务。','小林没有完成任务。'),
('反义词','今天很冷。','今天很热。','今天不冷。'),
('情感','小林喜欢这部电影。','小林讨厌这部电影。','小林不喜欢这部电影。'),
('否定词非命题否定','小林阅读了关于否定的论文。','小林阅读了关于肯定的论文。','小林没有阅读关于否定的论文。')]
save('data/challenge.json',[dict(category=a,pos=b,contrast=c,alternative=d,contrast_relation=('propositional_negation' if i<3 else ['double_negation_near_equivalence','antonym_not_complement','sentiment_opposition_not_complement','topic_word_substitution'][i-3]),gold_negative_key='contrast' if i<3 else 'alternative') for i,(a,b,c,d) in enumerate(challenge)])
texts=sorted({r[k] for r in rows for k in ['pos','neg','pos_para','neg_para','scope','wrong']}|{r[k] for r in json.loads((R/'data/challenge.json').read_text()) for k in ['pos','contrast','alternative']})
save('data/texts.json',texts)
save('config.json',dict(model='/dataset1/zailong/models/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28',dtype='float32',layers=[0,5,9,14,19,23,28],pooling=['last','mean'],ranks=[1,2,4,8,16,32],alphas=[0.01,0.1,1,10],seeds=[17,29,43],primary=dict(layer=14,pool='mean',split='joint',metric='mse_lowrank_minus_shift'),input='raw text; add_special_tokens=False; no chat template',data_seed=20260928))
save('data/manifest.json',dict(source='Controlled templates authored by AI assistant; no external corpus or generation API',n_pairs=len(rows),n_texts=len(texts),splits={s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},sha256=hashlib.sha256((R/'data/pairs.json').read_bytes()).hexdigest(),human_review='No independent human reviewer available; assistant audit recorded separately'))
print((R/'data/manifest.json').read_text())
