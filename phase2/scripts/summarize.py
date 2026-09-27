import csv,json,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import R,savecsv

def read(n):return list(csv.DictReader((R/'results'/n).open()))
def agg(rows,keys,values):
 g=collections.defaultdict(list)
 for r in rows:g[tuple(r[k] for k in keys)].append(r)
 out=[]
 for key,rr in g.items():
  z=dict(zip(keys,key))
  for k in values:
   v=np.array([float(r[k]) for r in rr]);z[k]=float(v.mean());z[k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.
  out.append(z)
 return out
metrics=read('ab_metrics.csv');summary=agg(metrics,['experiment','representation','method','split','mode'],['mse','recall1','mrr','margin','candidate_n','positive_n','chance']);savecsv('ab_summary.csv',summary)
c=read('c_metrics.csv');cs=agg(c,['branch','geometry','method'],['mse','recall1','mrr','margin','pred_norm']);savecsv('c_summary.csv',cs)
# Load only main/secondary relevant per-query results for uncertainty, average within base before resampling.
primary=[];comparison=[]
for r in csv.DictReader((R/'results/b_per_query.csv').open()):
 if r['split'] in ['iid','topic','template','joint'] and r['mode'] in ['seen','unseen'] and r['method'] in ['lowrank','surface_selected','rank1','shift']:
  primary.append(r)
ci=[]
def interval(rows,method_a,method_b,label,representation,split,mode,metric='recall1',clusterkey='cluster',reps=2000):
 # rows contain repeated source variants and training bootstrap seeds, neither is an independent test unit.
 d=collections.defaultdict(lambda:collections.defaultdict(list));meta={}
 for r in rows:
  if r['method'] in [method_a,method_b]:d[r['id']][r['method']].append(float(r[metric]));meta[r['id']]=r
 vals=[];groups=[]
 for ident,rr in d.items():
  if method_a in rr and method_b in rr:vals.append(np.mean(rr[method_a])-np.mean(rr[method_b]));groups.append(meta[ident][clusterkey])
 vals=np.array(vals);gg=sorted(set(groups));blocks=[vals[np.array(groups)==g] for g in gg];rng=np.random.default_rng(20260928);boot=[]
 for _ in range(reps):boot.append(np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]).mean())
 q=np.quantile(boot,[.0125,.025,.975,.9875]);return dict(comparison=label,representation=representation,split=split,mode=mode,metric=metric,cluster_unit=clusterkey,n_base=len(vals),n_clusters=len(gg),difference=float(vals.mean()),ci95_low=float(q[1]),ci95_high=float(q[2]),ci975_low=float(q[0]),ci975_high=float(q[3]))
for rep in ['l0_mean','l14_mean','l14_last','l28_last']:
 for split,mode in [('iid','unseen'),('iid','seen'),('topic','seen'),('topic','unseen'),('template','seen'),('joint','unseen')]:
  rr=[r for r in primary if r['representation']==rep and r['split']==split and r['mode']==mode]
  for unit in ['cluster','topic']:ci.append(interval(rr,'lowrank','surface_selected','lowrank_minus_surface',rep,split,mode,clusterkey=unit))
# Context-vs-embedding comparison uses identical candidate semantics but each layer's own space.
for split,mode in [('iid','unseen'),('topic','unseen'),('joint','unseen')]:
 rr=[]
 for r in primary:
  if r['method']=='lowrank' and r['split']==split and r['mode']==mode and r['representation'] in ['l0_mean','l14_mean']:
   rr.append(dict(r,method=r['representation']))
 for unit in ['cluster','topic']:ci.append(interval(rr,'l14_mean','l0_mean','context_minus_embedding','cross_layer',split,mode,clusterkey=unit))
cc=[r for r in read('c_per_item.csv') if r['branch']=='new_matched_confirmation' and r['geometry']=='raw']
for unit in ['cluster','topic']:ci.append(interval(cc,'full_affine','lowrank','C_full_affine_minus_lowrank','l28_last','new_matched','five',clusterkey=unit))
savecsv('paired_intervals.csv',ci)
# Summaries of candidate-category movement, needed to interpret full affine success.
cat=agg(read('c_per_item.csv'),['branch','geometry','method'],['correct_neg_cos','source_pos_cos','other_pos_cos','other_neg_cos','scope_cos','correct_neg_change_vs_identity','source_pos_change_vs_identity','other_pos_change_vs_identity','other_neg_change_vs_identity','scope_change_vs_identity']);savecsv('c_category_summary.csv',cat)
cb=agg(read('c_frozen_b_per_query.csv'),['method','split','mode'],['recall1','mrr','margin']);savecsv('c_frozen_b_summary.csv',cb)
plt.rcParams.update({'font.size':10,'figure.dpi':140})
fig,axes=plt.subplots(1,2,figsize=(11,4))
sp=read('a_spectra.csv')
for centered,ax in zip(['False','True'],axes):
 for kind in ['negation','time','emphasis']:
  rr=[r for r in sp if r['representation']=='l14_mean' and r['kind']==kind and r['centered']==centered];ax.plot([int(r['component']) for r in rr],[float(r['cumulative']) for r in rr],label=kind)
 ax.set_title('Centered PCA' if centered=='True' else 'Uncentered difference SVD');ax.set_xlabel('Components');ax.set_ylabel('Cumulative energy');ax.legend()
fig.tight_layout();fig.savefig(R/'figures/a_difference_spectrum.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,split in zip(axes,['iid','topic']):
 labels=['negation','time','emphasis'];xx=np.arange(3)
 for j,method in enumerate(['length_shift','rank1','lowrank','full_affine']):
  yy=[]
  for kind in labels:
   rr=next(r for r in summary if r['experiment']=='A_'+kind and r['representation']=='l14_mean' and r['split']==split and r['method']==method);sh=next(r for r in summary if r['experiment']=='A_'+kind and r['representation']=='l14_mean' and r['split']==split and r['method']=='shift');yy.append(rr['mse']/sh['mse'])
  ax.bar(xx+(j-1.5)*.18,yy,.18,label=method)
 ax.set_xticks(xx,labels);ax.set_title(split);ax.set_ylabel('MSE / shift MSE');ax.axhline(1,color='gray',ls='--')
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(R/'figures/a_edit_comparison.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,mode in zip(axes,['seen','unseen']):
 xx=np.arange(4)
 for j,method in enumerate(['shift','surface_selected','lowrank','full_affine']):
  yy=[next(r['recall1'] for r in summary if r['experiment']=='B_shared' and r['representation']==rep and r['split']=='iid' and r['mode']==mode and r['method']==method) for rep in ['l0_mean','l14_mean','l14_last','l28_last']];ax.bar(xx+(j-1.5)*.18,yy,.18,label=method)
 ax.set_xticks(xx,['0 mean','14 mean','14 last','28 last']);ax.set_ylim(0,1);ax.set_title('iid / '+mode);ax.set_ylabel('Multi-positive Recall@1')
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(R/'figures/b_semantic_retrieval.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
methods=['identity','shift','rank1','lowrank','full_delta','full_affine'];xx=np.arange(len(methods))
for j,branch in enumerate(['phase1_reproduction','new_matched_confirmation']):
 yy=[next(r['recall1'] for r in cs if r['branch']==branch and r['geometry']=='raw' and r['method']==m) for m in methods];axes[0].bar(xx+(j-.5)*.35,yy,.35,label=branch)
axes[0].set_xticks(xx,methods,rotation=25,ha='right');axes[0].set_ylabel('Recall@1');axes[0].legend(fontsize=7)
for method in ['shift','lowrank','full_affine']:
 r=next(r for r in cat if r['branch']=='new_matched_confirmation' and r['geometry']=='raw' and r['method']==method);axes[1].plot(['correct neg','original pos','other pos','other neg','scope'],[r[k+'_change_vs_identity'] for k in ['correct_neg','source_pos','other_pos','other_neg','scope']],marker='o',label=method)
axes[1].set_ylabel('Cosine change vs identity');axes[1].tick_params(axis='x',rotation=20);axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(R/'figures/c_replication_and_margins.png');plt.close(fig)
# Complete machine readable tables plus compact markdown index.
lines=['# Phase2 complete summary','Full seed-level and per-item data are in results/.','|experiment|representation|method|split|mode|MSE|R@1|MRR|','|---|---|---|---|---|---|---|---|']
for r in summary:lines.append('|'+ '|'.join(str(r[k]) for k in ['experiment','representation','method','split','mode'])+'|'+ '|'.join(f'{r[k]:.6g}' for k in ['mse','recall1','mrr'])+'|')
(R/'RESULTS_TABLES.md').write_text('\n'.join(lines)+'\n');print('Summary complete; intervals:',json.dumps([r for r in ci if r['representation']=='l14_mean' and r['split']=='iid' and r['mode']=='unseen'],indent=2),flush=True)
