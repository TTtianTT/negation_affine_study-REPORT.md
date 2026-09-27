"""Exploratory projection diagnostic, not a causal intervention on the LLM."""
import json,csv
from pathlib import Path
import numpy as np
from scipy.linalg import svd
from linear import Design
R=Path(__file__).resolve().parents[1];rows=json.loads((R/'data/pairs.json').read_text());texts=json.loads((R/'data/texts.json').read_text());lookup={s:i for i,s in enumerate(texts)};H=np.load(R/'cache/l14_mean.npy').astype(float)
ids={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in sorted({r['split'] for r in rows})}
X=H[[lookup[r['pos']] for r in rows]];Y=H[[lookup[r['neg']] for r in rows]];P=H[[lookup[r['pos_para']] for r in rows]];tr=ids['train'];dv=ids['dev'];alphas=[.01,.1,1,10]
def train(x,y):
 a=np.concatenate([x[tr],y[tr]]);labels=np.r_[np.zeros(len(tr)),np.ones(len(tr))];d=Design(a);fits={al:d.fit(labels[:,None],al) for al in alphas};v=np.concatenate([x[dv],y[dv]]);vl=np.r_[np.zeros(len(dv)),np.ones(len(dv))]
 al=max(alphas,key=lambda z:np.mean((d.predict(v,fits[z])[:,0]>.5)==vl));return d,fits[al]
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
base,fit=train(X,Y);_,_,vt=svd(Y[tr]-X[tr],full_matrices=False);out=[]
for method in ['none','difference_svd','random']:
 for rank in ([0] if method=='none' else [1,8]):
  for seed in ([17,29,43] if method=='random' else [17]):
   w=np.zeros((X.shape[1],0)) if method=='none' else vt[:rank].T if method=='difference_svd' else np.linalg.qr(np.random.default_rng(seed).normal(size=(X.shape[1],rank)))[0]
   xx=X-(X@w)@w.T;yy=Y-(Y@w)@w.T;pp=P-(P@w)@w.T;d,rf=train(xx,yy)
   for split in ['iid','topic','template','expression','joint']:
    ii=ids[split];q=np.concatenate([xx[ii],yy[ii]]);lab=np.r_[np.zeros(len(ii)),np.ones(len(ii))]
    sims=norm(pp[ii])@norm(xx[ii]).T;target=np.diag(sims);content_rank=np.sum(sims>=target[:,None]-1e-12,axis=1)
    out.append(dict(method=method,rank=rank,seed=seed,split=split,refit_probe_accuracy=float(np.mean((d.predict(q,rf)[:,0]>.5)==lab)),frozen_probe_accuracy=float(np.mean((base.predict(q,fit)[:,0]>.5)==lab)),content_recall1=float(np.mean(content_rank==1)),content_mrr=float(np.mean(1/content_rank))))
with (R/'results/removal.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
