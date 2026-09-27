"""Surface-only baselines; no hidden state or test-fitted preprocessing."""
import json,csv,re
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
R=Path(__file__).resolve().parents[1];rows=json.loads((R/'data/pairs.json').read_text());out=[]
def f(s,mode):
    a=[len(s)]
    if mode!='length':a += [s.count(w) for w in ['没有','并非','不','已经','确实','昨天','上午','关于','“','。','，']]
    return a
for mode in ['length','lexical_template_length']:
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    def xy(rr,para=False):
        ks=['pos_para','neg_para'] if para else ['pos','neg']
        return np.array([f(r[k],mode) for r in rr for k in ks]),np.tile([0,1],len(rr))
    x,y=xy(train); dx,dy=xy(dev); scaler=StandardScaler().fit(x)
    models=[LogisticRegression(C=c,max_iter=1000).fit(scaler.transform(x),y) for c in [.01,.1,1,10]]
    m=max(models,key=lambda m:m.score(scaler.transform(dx),dy))
    for split in sorted({r['split'] for r in rows}):
        for para in [False,True]:
            tx,ty=xy([r for r in rows if r['split']==split],para)
            out.append(dict(mode=mode,split=split,variant='paraphrase' if para else 'canonical',accuracy=m.score(scaler.transform(tx),ty)))
with (R/'results/surface.csv').open('w') as fp:
    w=csv.DictWriter(fp,fieldnames=out[0]);w.writeheader();w.writerows(out)
# Pre-existing challenge items have relation-specific gold annotations, not one conflated label.
review=[]
for split in ['train','dev','topic','template','expression','joint']:
    rr=[r for r in rows if r['split']==split]
    for i in np.random.default_rng(20).choice(len(rr),3,replace=False):
        r=rr[i];review.append(dict(id=r['id'],split=split,pos=r['pos'],neg=r['neg'],pos_para=r['pos_para'],neg_para=r['neg_para'],reviewer='AI assistant (not independent human)',status='pending assistant inspection'))
(R/'data/audit_sample.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
