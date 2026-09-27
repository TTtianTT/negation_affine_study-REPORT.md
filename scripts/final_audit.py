"""Audit completeness and correct fixed-random-basis parameter metadata."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
expected={'metrics.csv':6552,'grid.csv':1176,'choices.csv':42,'probes.csv':1176,'primary_pairs.csv':15624,'removal.csv':45,'surface_operator.csv':72}
out={}
for name,n in expected.items():
 p=R/'results'/name;rows=list(csv.DictReader(p.open()));assert len(rows)==n,(name,len(rows),n)
 if name in ['metrics.csv','primary_pairs.csv']:
  for r in rows:
   if r['method']=='random_subspace':r['parameters']=str(3584*int(r.get('operator_rank',r.get('rank')))+3584)
  with p.open('w') as f:
   w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 out[name]=len(rows)
rows=json.loads((R/'data/pairs.json').read_text());txt=json.loads((R/'data/texts.json').read_text());cfg=json.loads((R/'config.json').read_text());env=json.loads((R/'results/environment.json').read_text())
assert cfg==env['config'];assert env['data_sha256']==hashlib.sha256((R/'data/texts.json').read_bytes()).hexdigest()
for l in cfg['layers']:
 for p in cfg['pooling']:
  a=np.load(R/f'cache/l{l}_{p}.npy');assert a.shape==(len(txt),3584) and np.isfinite(a).all()
  for seed in cfg['seeds']:
   z=np.load(R/f'cache/operator_l{l}_{p}_s{seed}.npz');assert z['input_factor'].shape==(3584,int(z['rank']))
# Reloaded factorization must reproduce the exact stored primary per-sample MSE.
lookup={s:i for i,s in enumerate(txt)}; rr={r['id']:r for r in rows}
x=np.load(R/'cache/l14_mean.npy').astype(float)
pp=list(csv.DictReader((R/'results/primary_pairs.csv').open()))
errs=[]
for seed in cfg['seeds']:
 z=np.load(R/f'cache/operator_l14_mean_s{seed}.npz')
 for p in [a for a in pp if a['seed']==str(seed) and a['method']=='lowrank']:
  d=rr[p['id']];pk,nk=('pos','neg') if p['variant']=='canonical' else ('pos_para','neg_para')
  xx=x[lookup[d[pk]]];yy=x[lookup[d[nk]]];pred=xx+(xx@z['input_factor'])@z['output_factor']+z['bias']
  errs.append(abs(float(np.mean((pred-yy)**2))-float(p['mse'])))
out['max_reloaded_operator_mse_difference']=max(errs)
assert max(errs)<1e-8
out['all_shapes_and_hashes_valid']=True
(R/'results/completeness.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
