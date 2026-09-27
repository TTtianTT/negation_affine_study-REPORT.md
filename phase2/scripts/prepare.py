"""Deterministic controlled data. No model generation, no test-output-driven edits."""
import json,hashlib,random,sys,datetime
from pathlib import Path
import difflib
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1];P=R.parent
cfg0=json.loads((P/'config.json').read_text());tok=AutoTokenizer.from_pretrained(cfg0['model'],local_files_only=True)
def save(name,x): (R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
# All topics and content nouns are newly authored for phase2.
TOPICS={
'天文观测':'望远镜镜面 星表坐标 观测日志 星光强度 星云位置 穹顶轨道 跟踪电机 滤光镜片',
'船舶维修':'船体焊缝 航行参数 维修档案 舱内湿度 漏水位置 甲板积水 舱底水泵 船舱阀门',
'矿山巡检':'矿井支架 作业许可 巡检台账 巷道风速 裂缝位置 矿道碎石 通风设备 安全绳索',
'广播制作':'混音设备 播出时刻 采访录音 收音电平 剪辑位置 录音室杂物 监听音箱 话筒电池',
'档案修复':'纸张纤维 馆藏编号 修复档案 纸张酸度 破损位置 装裱台面 恒湿设备 防护薄膜',
'铁路调度':'道岔状态 列车班次 调度日志 轨面温度 限速区段 站台积雪 信号装置 转辙电机',
'水厂运维':'滤池结构 供水参数 运维档案 出水浊度 渗漏位置 沉淀池淤泥 加药设备 过滤组件',
'航空地勤':'起落架轮胎 停机位编号 交接单据 胎压数值 维修位置 机坪杂物 牵引装置 应急灯具',
'桥梁检测':'桥墩表面 荷载数据 检测档案 桥面位移 裂纹位置 排水口杂物 位移传感器 护栏组件',
'展览布置':'展柜玻璃 展品编号 布展文件 展厅照度 展板位置 展台杂物 定向射灯 展柜锁具',
'海洋监测':'浮标外壳 采样坐标 监测档案 海水盐度 采样位置 船舷附着物 水下探头 供电模块',
'纺织生产':'纱线质量 布料批次 生产台账 织机转速 疵点位置 机架棉絮 断纱传感器 织机梭芯'}
VERBS=['检查','核对','整理','记录','标记','清理','安装','更换'];NAMES=['林岚','周宁','许澄','陈遥']
rows=[]
for ti,(topic,objects) in enumerate(TOPICS.items()):
 for vi,o in enumerate(objects.split()):
  for ai,s in enumerate(NAMES):
   v=VERBS[vi]; t='周一当天';base=f'{t}，{s}{v}了{o}。'
   fold=(ai+ti+vi)%4
   split=('train' if fold<2 else 'dev' if fold==2 else 'iid') if ti<8 and vi<6 else ('template' if ti<8 else 'topic' if vi<6 else 'joint')
   pos=[base,f'{s}在{t}{v}过{o}。']
   neg=[f'{t}，{s}没有{v}{o}。',f'{t}，{s}未{v}{o}。',f'“{t}，{s}{v}了{o}”这一说法不属实。',f'{t}，{s}不曾{v}{o}。',f'{t}，{s}从未{v}{o}。']
   other_o=list(TOPICS.values())[(ti+1)%len(TOPICS)].split()[vi]
   hard=[f'{t}，另一位同事没有{v}{o}。',f'{t}，{s}不必{v}{o}。',f'{t}，{s}并非只{v}了{o}。',f'{t}，{s}没有{v}{other_o}。']
   rows.append(dict(id=f'B{ti:02}-{vi}-{ai}',topic=topic,topic_id=ti,verb=v,template=f'predicate_{v}',object=o,actor=s,split=split,pos=pos,neg=neg,neg_families=['absence_没有','absence_未','proposition_untrue','experiential_unseen','experiential_unseen'],hard=hard,hard_categories=['different_subject','modality','scope','different_object'],truth_condition=f'exists event({s},{v},{o}) during 周一当天; all negatives deny existence within same interval'))
save('data/b_propositions.json',rows)
def editinfo(x,y):
 ix=tok.encode(x,add_special_tokens=False);iy=tok.encode(y,add_special_tokens=False);ops=difflib.SequenceMatcher(a=ix,b=iy,autojunk=False).get_opcodes();d=[];a=[];positions=[]
 for tag,i,j,k,l in ops:
  if tag!='equal':d+=ix[i:j];a+=iy[k:l];positions.append(i/max(1,len(ix)))
 return dict(n=len(ix),m=len(iy),delta_length=len(iy)-len(ix),deleted=len(d),added=len(a),edited=len(d)+len(a),position=min(positions) if positions else 0,deleted_ids=d,added_ids=a,opcodes=ops)
edits=[]
for r in rows:
 x=r['pos'][0];neg=r['neg'][0];ref=editinfo(x,neg)
 # Candidates chosen only by tokenizer statistics, before any model evaluation.
 opts={'time':[x.replace('周一当天',t) for t in ['周二当天','周三当天','上周一当天']], 'emphasis':[x.replace(r['actor']+r['verb'],r['actor']+a+r['verb']) for a in ['确实','的确','真的','已经']]}
 targets={'negation':neg}
 for kind,tt in opts.items():
  def cost(y):
   z=editinfo(x,y);return abs(z['delta_length']-ref['delta_length'])+abs(z['edited']-ref['edited'])+2*abs(z['position']-ref['position'])
  targets[kind]=min(tt,key=lambda y:(cost(y),y))
 for kind,y in targets.items():
  info=editinfo(x,y);cost=abs(info['delta_length']-ref['delta_length'])+abs(info['edited']-ref['edited'])+2*abs(info['position']-ref['position'])
  edits.append(dict(id=r['id']+'-'+kind,base_id=r['id'],topic=r['topic'],template=r['template'],split=r['split'],kind=kind,source=x,target=y,matching_cost=cost,**info))
save('data/a_edits.json',edits)
# Independent C confirmation: same two phase1 joint topics, entirely new verb/object events.
C={
'考古':('考古队员','测量:墓室尺寸 描绘:壁画轮廓 登记:石器重量 拍摄:遗迹剖面 修复:青铜饰件 包装:骨骼标本 筛分:填土沉积物 拼接:陶片纹样 加固:墓道支撑 翻译:碑刻文字 搭建:遮雨棚架 封存:探方档案'),
'出版':('编辑','审定:封面配色 编排:索引条目 录入:版权信息 签署:印刷合同 发送:作者样书 绘制:装帧草图 裁切:校样纸张 核算:印制成本 装订:试印样本 编写:发行通知 收录:勘误条目 登记:稿酬金额')}
c=[]
for ti,(topic,(role,aa)) in enumerate(C.items()):
 for vi,a in enumerate(aa.split()):
  v,o=a.split(':')
  for ai,pref in enumerate(['这位','那位','新来的','负责此事的']):
   s=pref+role;p=f'关于{o}，{s}已经{v}了。';n=f'“{p[:-1]}”并非事实。'
   c.append(dict(id=f'C{ti}-{vi:02}-{ai}',topic=topic,action=a,pos=p,neg=n,scope=f'并非所有{role}都{v}了{o}。'))
for r in c:r['other_id']=next(q['id'] for q in c if q['topic']==r['topic'] and q['action']!=r['action'])
old=json.loads((P/'data/pairs.json').read_text());assert not {r['action'] for r in c}&{r['action'] for r in old}
save('data/c_confirmation.json',c)
challenge=[
 dict(id='Q_all',pos='所有值班员都参加了演练。',neg=['并非所有值班员都参加了演练。','至少一名值班员没有参加演练。'],wrong=['所有值班员都没有参加演练。','所有值班员都不必参加演练。'],category='universal_scope',assumption='nonempty finite on-duty group'),
 dict(id='Q_some',pos='至少一名值班员参加了演练。',neg=['没有任何值班员参加演练。','所有值班员都没有参加演练。'],wrong=['并非所有值班员都参加了演练。','至少一名值班员没有参加演练。'],category='existential_scope',assumption='same group/event'),
 dict(id='M_must',pos='按规定，林岚必须佩戴护目镜。',neg=['按规定，林岚不必佩戴护目镜。','规定并未要求林岚佩戴护目镜。'],wrong=['按规定，林岚必须不佩戴护目镜。','林岚没有佩戴护目镜。'],category='necessity',assumption='same deontic source, not factual event'),
 dict(id='M_can',pos='按规定，周宁可以使用这台设备。',neg=['按规定，周宁不可以使用这台设备。','规定不允许周宁使用这台设备。'],wrong=['周宁没有使用这台设备。','周宁不必使用这台设备。'],category='permission',assumption='permission, not ability'),
 dict(id='lexical',pos='周宁在周一阅读了题为《没有》的小说。',neg=['周宁在周一没有阅读题为《没有》的小说。','周宁在周一未读过题为《没有》的小说。'],wrong=['周宁在周一阅读了题为《已经》的小说。','周宁在周一并非没有阅读题为《没有》的小说。'],category='mention_and_double_negation',assumption='novel title is lexical mention'),
 dict(id='sentiment',pos='林岚喜欢这幅画。',neg=['林岚不喜欢这幅画。','林岚并不喜欢这幅画。'],wrong=['林岚讨厌这幅画。','林岚喜欢另一幅画。'],category='sentiment_not_complement',assumption='not-liking includes neutral'),
 dict(id='antonym',pos='这个房间很明亮。',neg=['这个房间并不很明亮。','这个房间算不上很明亮。'],wrong=['这个房间很黑暗。','这个房间不算黑暗。'],category='antonym_not_complement',assumption='threshold predicate; antonym is not complement')]
save('data/challenge.json',challenge)
texts=sorted({s for r in rows for k in ['pos','neg','hard'] for s in r[k]}|{r[k] for r in edits for k in ['source','target']}|{r[k] for r in c for k in ['pos','neg','scope']}|{r['pos'] for r in challenge}|{s for r in challenge for k in ['neg','wrong'] for s in r[k]})
assert len(texts)<=8000
assert not set(texts)&set(json.loads((P/'data/texts.json').read_text()))
save('data/texts.json',texts)
cfg=dict(model=cfg0['model'],revision=cfg0['revision'],dtype='float32',representations=[[0,'mean'],[14,'mean'],[14,'last'],[28,'last']],seeds=[17,29,43],ranks=[1,2,4,8,16,32],alphas=[.01,.1,1,10],seen_forms=[0,1,2],unseen_forms=[3,4],input='raw text, no special tokens, no chat template',data_seed=20260928,primary=dict(representation='l14_mean',split='iid',candidate_forms='unseen',metric='multi_positive_recall1',comparison='lowrank minus dev-selected surface baseline'))
save('config.json',cfg)
summary=dict(version='phase2_v1',source='AI-authored controlled data; no external corpus or model generation API',n_base=len(rows),n_C=len(c),n_edits=len(edits),n_texts=len(texts),splits={s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},human_review='not available; semantic runs exploratory',proposition_uniqueness=len({(r['topic'],r['verb'],r['actor'],r['object']) for r in rows}),data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'data'/n for n in ['a_edits.json','b_propositions.json','c_confirmation.json','challenge.json','texts.json']]})
save('data/manifest.json',summary)
# Review development examples only; all families reviewed before model outputs.
sample=[next(r for r in rows if r['verb']==v and r['split']==sp) for v in VERBS[:6] for sp in ['train','dev']]+[next(r for r in rows if r['topic']==t and r['split']=='dev') for t in list(TOPICS)[1:8]]
save('data/review_pending.json',[dict(id=r['id'],split=r['split'],pos=r['pos'],neg=r['neg'],hard=r['hard'],reviewer='pending',status='pending') for r in sample if r['split'] in ['train','dev']])
print(json.dumps(summary,ensure_ascii=False,indent=2))
