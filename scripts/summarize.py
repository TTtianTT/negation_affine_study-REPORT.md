import json,csv,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def load(name): return list(csv.DictReader((R/'results'/name).open()))
def save(name,rr):
 with (R/'results'/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def avg(rr,k):return float(np.mean([float(r[k]) for r in rr]))
m=load('metrics.csv');p=load('probes.csv');s=load('spectra.csv');pairs=load('primary_pairs.csv')
keys=['layer','pool','method','split','variant'];groups=collections.defaultdict(list)
for r in m:groups[tuple(r[k] for k in keys)].append(r)
agg=[]
for key,rr in groups.items():
 a=dict(zip(keys,key));a.update(n=rr[0]['n'])
 for k in ['mse','relative_improvement','cosine','recall1','mrr','neg_over_pos']:
  a[k]=avg(rr,k);a[k+'_seed_sd']=float(np.std([float(r[k]) for r in rr],ddof=1))
 agg.append(a)
save('summary.csv',agg)
# Average repeated training perturbations before paired cluster bootstrap: seeds are not independent test units.
ci=[];rng=np.random.default_rng(20260928)
for split in ['iid','topic','template','expression','joint']:
 for variant in ['canonical','paraphrase']:
  sub=[r for r in pairs if r['split']==split and r['variant']==variant]
  ids=sorted({r['id'] for r in sub});by=[]
  for ident in ids:
   a=[r for r in sub if r['id']==ident];low=[r for r in a if r['method']=='lowrank'];sh=[r for r in a if r['method']=='shift'];identr=[r for r in a if r['method']=='identity']
   by.append(dict(id=ident,cluster=a[0]['cluster'],low=avg(low,'mse'),shift=avg(sh,'mse'),identity=avg(identr,'mse'),recall_diff=np.mean([int(r['rank'])==1 for r in low])-np.mean([int(r['rank'])==1 for r in sh])))
  clusters=sorted({r['cluster'] for r in by});blocks=[np.array([[r['low'],r['shift'],r['identity'],r['recall_diff']] for r in by if r['cluster']==g]) for g in clusters]
  allv=np.concatenate(blocks);boots=[]
  for _ in range(2000):
   a=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]);v=a.mean(0);boots.append([v[0]-v[1],1-v[0]/v[1],v[3]])
  bounds=np.quantile(boots,[.025,.975],axis=0);v=allv.mean(0)
  ci.append(dict(split=split,variant=variant,n=len(ids),clusters=len(clusters),mse_difference=v[0]-v[1],difference_ci_low=bounds[0,0],difference_ci_high=bounds[1,0],improvement_over_shift=1-v[0]/v[1],improvement_ci_low=bounds[0,1],improvement_ci_high=bounds[1,1],recall_difference=v[3],recall_ci_low=bounds[0,2],recall_ci_high=bounds[1,2]))
save('paired_cluster_ci.csv',ci)
# Plot all prespecified layers, never choose layers using test.
plt.rcParams.update({'font.size':10,'figure.dpi':140})
fig,ax=plt.subplots(2,2,figsize=(11,7),sharex=True)
for j,pool in enumerate(['last','mean']):
 for split,style in [('topic','-'),('joint','--')]:
  for method,color in [('identity','gray'),('shift','tab:blue'),('lowrank','tab:orange'),('full_delta','tab:green')]:
   rr=sorted([r for r in agg if r['pool']==pool and r['split']==split and r['variant']=='canonical' and r['method']==method],key=lambda r:int(r['layer']))
   ax[0,j].plot([int(r['layer']) for r in rr],[1-r['relative_improvement'] for r in rr],style,color=color,label=method+'/'+split)
   ax[1,j].plot([int(r['layer']) for r in rr],[r['recall1'] for r in rr],style,color=color)
 ax[0,j].set_title(pool);ax[0,j].set_ylabel('MSE / identity MSE');ax[1,j].set_ylabel('Recall@1');ax[1,j].set_xlabel('Hidden-state index');ax[0,j].axvline(14,color='black',alpha=.2)
ax[0,0].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(R/'figures/layer_performance.png');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4))
for split in ['dev','iid','topic','template','expression','joint']:
 rr=[next(r for r in agg if r['layer']=='14' and r['pool']=='mean' and r['method']==f'rank{k}' and r['split']==split and r['variant']=='canonical') for k in [1,2,4,8,16,32]]
 ax[0].plot([1,2,4,8,16,32],[1-r['relative_improvement'] for r in rr],marker='o',label=split)
 ax[1].plot([1,2,4,8,16,32],[r['recall1'] for r in rr],marker='o',label=split)
for a in ax:a.set_xscale('log',base=2);a.set_xlabel('Rank (alpha selected on dev)');a.legend(fontsize=8)
ax[0].set_ylabel('MSE / identity MSE');ax[1].set_ylabel('Recall@1');fig.suptitle('Prespecified layer 14, mean');fig.tight_layout();fig.savefig(R/'figures/rank_performance.png');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4))
for centered in ['False','True']:
 for pool in ['last','mean']:
  rr=[r for r in s if r['layer']=='14' and r['pool']==pool and r['centered']==centered];xx=[int(r['component']) for r in rr]
  ax[0].plot(xx,[float(r['singular_value']) for r in rr],label=f'{pool}, centered={centered}')
  ax[1].plot(xx,[float(r['cumulative']) for r in rr],label=f'{pool}, centered={centered}')
ax[0].set_yscale('log');ax[0].set_ylabel('Singular value');ax[1].set_ylabel('Cumulative squared-singular-value energy')
for a in ax:a.set_xlabel('Component');a.legend(fontsize=8)
fig.suptitle('Training differences only, layer 14');fig.tight_layout();fig.savefig(R/'figures/singular_spectrum.png');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4),sharey=True)
for j,pool in enumerate(['last','mean']):
 for split in ['topic','template','expression','joint']:
  yy=[avg([r for r in p if r['layer']==str(l) and r['pool']==pool and r['split']==split and r['variant']=='canonical' and r['control']=='real'],'accuracy') for l in [0,5,9,14,19,23,28]]
  ax[j].plot([0,5,9,14,19,23,28],yy,marker='o',label=split)
 ax[j].axhline(.5,color='gray',ls='--');ax[j].set_title(pool);ax[j].set_xlabel('Hidden-state index');ax[j].legend();ax[j].set_ylabel('Linear classification accuracy')
fig.tight_layout();fig.savefig(R/'figures/linear_probe.png');plt.close(fig)
# Separate training-fit and generalization evidence.
fig,ax=plt.subplots(figsize=(7,4))
for method in ['shift','lowrank','full_delta','full_affine']:
 yy=[next(r['mse'] for r in agg if r['layer']=='14' and r['pool']=='mean' and r['variant']=='canonical' and r['split']==sp and r['method']==method) for sp in ['train','dev','iid','topic','template','expression','joint']]
 ax.plot(range(7),yy,marker='o',label=method)
ax.set_xticks(range(7),['train','dev','iid','topic','template','expression','joint']);ax.set_ylabel('MSE');ax.legend();fig.tight_layout();fig.savefig(R/'figures/generalization_gap.png');plt.close(fig)
# Complete human-readable summary table, CSV retains all seed-level observations.
lines=['# 完整汇总表','三种训练bootstrap种子取均值。置信区间见paired_cluster_ci.csv，seed标准差见summary.csv。','|层|池化|方法|split|变体|MSE|相对恒等改善|R@1|MRR|正确否定优于原句|','|---|---|---|---|---|---|---|---|---|---|']
for r in agg:lines.append('|'+ '|'.join(str(r[k]) if k in keys else f'{r[k]:.5f}' for k in keys+['mse','relative_improvement','recall1','mrr','neg_over_pos'])+'|')
(R/'RESULTS_TABLES.md').write_text('\n'.join(lines))
print(json.dumps(ci,ensure_ascii=False,indent=2))
