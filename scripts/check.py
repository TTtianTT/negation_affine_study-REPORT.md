import json,hashlib
from pathlib import Path
import numpy as np
from linear import Design
R=Path(__file__).resolve().parents[1];checks={};rng=np.random.default_rng(0);x=rng.normal(size=(60,10));y=rng.normal(size=(60,8));de=Design(x);fit=de.fit(y,.1)
x0=x-x.mean(0);y0=y-y.mean(0);lam=.1*np.trace(x0@x0.T)/len(x)
b=np.linalg.solve(x0.T@x0+lam*np.eye(10),x0.T@y0)
checks['ridge_vs_primal_maxerr']=float(np.max(abs(de.predict(x,fit)-(x0@b+y.mean(0)))))
assert checks['ridge_vs_primal_maxerr']<1e-10
objectives=[]
for r in [1,2,4,8]:
    w=fit[2][:r].T;br=b@w@w.T
    objectives.append(float(np.sum((y0-x0@br)**2)+lam*np.sum(br**2)))
assert all(a>=b-1e-9 for a,b in zip(objectives,objectives[1:]));checks['rank_objectives']=objectives
rows=json.loads((R/'data/pairs.json').read_text());assert len({r['group'] for r in rows})==len(rows);checks['unique_proposition_groups']=len(rows)
train=[r for r in rows if r['split']=='train']
for split,axis in [('topic','topic'),('template','template'),('expression','expression')]:
    assert not {r[axis] for r in train}&{r[axis] for r in rows if r['split']==split}
checks['heldout_axes_disjoint']=True
for split in ['dev','iid','template','expression','topic','joint']:
    assert not {r[k] for r in train for k in ['pos','neg','pos_para','neg_para']}&{r[k] for r in rows if r['split']==split for k in ['pos','neg','pos_para','neg_para']}
checks['no_cross_split_sentence_duplicates']=True
if (R/'results/environment.json').exists():
    env=json.loads((R/'results/environment.json').read_text());assert env['data_sha256']==hashlib.sha256((R/'data/texts.json').read_bytes()).hexdigest()
    checks['cache_hash_matches']=True
    pad=json.loads((R/'results/extraction_checks.json').read_text());checks['max_padding_relative_error']=max(r['relative_error'] for r in pad);assert checks['max_padding_relative_error']<1e-4
(R/'results/validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
