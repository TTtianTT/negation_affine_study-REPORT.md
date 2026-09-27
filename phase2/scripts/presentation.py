import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import R

def read(n):return list(csv.DictReader((R/'results'/n).open()))
s=read('ab_summary.csv');m=read('matched_metrics.csv');sp=read('a_spectra.csv');mp=read('matched_spectrum_l14_mean.csv');plt.rcParams.update({'font.size':10,'figure.dpi':140})
fig,axes=plt.subplots(1,2,figsize=(11,4.4));methods=['shift','length_shift','source_token_ridge','rank1','lowrank'];xx=np.arange(len(methods))
for j,kind in enumerate(['negation','matched_past']):
 yy=[]
 for method in methods:
  if kind=='negation':y=float(next(r['mse'] for r in s if r['experiment']=='A_negation' and r['representation']=='l14_mean' and r['split']=='iid' and r['method']==method))
  else:y=np.mean([float(r['mse']) for r in m if r['representation']=='l14_mean' and r['split']=='iid' and r['method']==method])
  yy.append(y)
 axes[0].bar(xx+(j-.5)*.36,np.array(yy)/yy[0],.36,label=kind)
axes[0].set_xticks(xx,['shift','length','tokens','rank1','lowrank'],rotation=20);axes[0].set_ylabel('MSE / own shift MSE');axes[0].set_title('14 mean, iid; same token edit geometry');axes[0].legend()
for rr,name in [(sp,'negation'),(mp,'matched_past')]:
 rr=[r for r in rr if r['representation']=='l14_mean' and r['centered']=='True' and (name=='matched_past' or r['kind']=='negation')];axes[1].plot([int(r['component']) for r in rr],[float(r['cumulative']) for r in rr],label=name)
axes[1].set_xlabel('Centered principal components');axes[1].set_ylabel('Cumulative difference energy');axes[1].legend();fig.tight_layout();fig.savefig(R/'figures/a_strictly_matched.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,mode in zip(axes,['seen','unseen']):
 xx=np.arange(4)
 for j,method in enumerate(['shift','surface_selected','lowrank','full_affine']):
  yy=[float(next(r['recall1'] for r in s if r['experiment']=='B_shared' and r['representation']==rep and r['split']=='iid' and r['mode']==mode and r['method']==method)) for rep in ['l0_mean','l14_mean','l14_last','l28_last']];ax.bar(xx+(j-1.5)*.18,yy,.18,label=method)
 ax.set_xticks(xx,['0 mean','14 mean','14 last','28 last']);ax.set_ylim(0,1.05);ax.set_title('iid / '+mode);ax.set_ylabel('Multi-positive Recall@1')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,fontsize=9);fig.tight_layout(rect=(0,0,1,.9));fig.savefig(R/'figures/b_semantic_retrieval.png');plt.close(fig)
c=read('c_summary.csv');e=read('c_expanded_candidates.csv');fig,ax=plt.subplots(figsize=(8,4));methods=['identity','shift','rank1','lowrank','full_delta','full_affine'];xx=np.arange(len(methods))
for j,size in enumerate([5,97]):
 yy=[float(next(r['recall1'] for r in c if r['branch']=='new_matched_confirmation' and r['geometry']=='raw' and r['method']==k)) if size==5 else np.mean([float(r['recall1']) for r in e if r['method']==k]) for k in methods];ax.bar(xx+(j-.5)*.35,yy,.35,label=f'{size} candidates');ax.axhline(1/size,ls='--',alpha=.5,label=f'1/{size} chance')
ax.set_xticks(xx,methods,rotation=20);ax.set_ylabel('Recall@1');ax.set_title('Same 96 fresh propositions, frozen phase1 models');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(R/'figures/c_candidate_size.png');plt.close(fig)
