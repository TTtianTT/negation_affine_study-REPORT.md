"""Post-hoc surface diagnostic: input-length-conditioned translation (no target lengths)."""
import json,csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
R=Path(__file__).resolve().parents[1];rows=json.loads((R/'data/pairs.json').read_text());texts=json.loads((R/'data/texts.json').read_text());lookup={s:i for i,s in enumerate(texts)};H=np.load(R/'cache/l14_mean.npy').astype(float);ln={r['text']:r['length'] for r in json.loads((R/'cache/token_audit.json').read_text())};idx={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in sorted({r['split'] for r in rows})};X=H[[lookup[r['pos']] for r in rows]];Y=H[[lookup[r['neg']] for r in rows]]
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def feature(s,mode):
 a=[ln[s],1/ln[s],len(s)]
 if mode=='length_template':a += [float(s.startswith(v)) for v in ['昨天','上午','关于']]
 return a
out=[]
for mode in ['length_only','length_template']:
 F=np.array([feature(r['pos'],mode) for r in rows])
 for seed in [17,29,43]:
  bt=np.random.default_rng(seed).choice(idx['train'],len(idx['train']),replace=True);sc=StandardScaler().fit(F[bt]);fits={a:Ridge(alpha=a).fit(sc.transform(F[bt]),Y[bt]-X[bt]) for a in [.01,.1,1,10,100]};dv=idx['dev'];a=min(fits,key=lambda a:np.mean((X[dv]+fits[a].predict(sc.transform(F[dv]))-Y[dv])**2));m=fits[a]
  for variant in ['canonical','paraphrase']:
   for split,ii in idx.items():
    if variant=='paraphrase' and split in ['train','dev']:continue
    pk,nk=('pos','neg') if variant=='canonical' else ('pos_para','neg_para')
    xx=H[[lookup[rows[i][pk]] for i in ii]];yy=H[[lookup[rows[i][nk]] for i in ii]];ff=np.array([feature(rows[i][pk],mode) for i in ii]);pred=xx+m.predict(sc.transform(ff));cand=[]
    for i in ii:
     r=rows[i];o=next(q for q in rows if q['id']==r['other_id']);cand.append([lookup[z] for z in [r[nk],r[pk],o['pos'],o['neg'],r['scope']]])
    sims=np.einsum('nd,nkd->nk',norm(pred),norm(H[np.array(cand)]));rank=1+np.sum(sims[:,1:]>=sims[:,:1]-1e-12,axis=1)
    out.append(dict(mode=mode,seed=seed,alpha=a,split=split,variant=variant,mse=float(np.mean((pred-yy)**2)),recall1=float(np.mean(rank==1)),mrr=float(np.mean(1/rank))))
with (R/'results/surface_operator.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
