"""Cached-only, exploratory scope challenges and candidate-composition ablation."""
import json,csv
import numpy as np
from core import *
# Qualitative scope / modal / sentiment challenges; never merged into the main binary label.
base_tr=np.array([i for i,r in enumerate(B) if r['split']=='train']);base_dv=np.array([i for i,r in enumerate(B) if r['split']=='dev']);tr=(base_tr[:,None]*2+np.arange(2)).ravel();dv=(base_dv[:,None]*2+np.arange(2)).ravel();bid=np.repeat(np.arange(len(B)),2);src=[s for r in B for s in r['pos']];fl,ft,_=features(src,tr);challenge=json.loads((R/'data/challenge.json').read_text());out=[]
# Predictors requiring surface features on novel challenge strings need the same train vocabulary.
audit={r['text']:r for r in json.loads((R/'cache/token_audit.json').read_text())};_,_,vocab=features(src,tr);vi={v:i for i,v in enumerate(vocab)}
cfl=[];cft=[]
for r in challenge:
 ids=audit[r['pos']]['ids'];length=[len(ids),1/len(ids),len(r['pos'])];v=np.zeros(len(vocab))
 for t in ids:
  if t in vi:v[vi[t]]+=1
 cfl.append(length);cft.append(np.r_[length,v])
cfl=np.array(cfl);cft=np.array(cft)
for rep in ['l14_mean','l28_last']:
 H=loadh(rep);X=H[[LOOKUP[s] for s in src]];Y=np.repeat(H[np.array([[LOOKUP[s] for s in r['neg'][:3]] for r in B])].mean(1),2,axis=0);cx=H[[LOOKUP[r['pos']] for r in challenge]]
 for seed in CFG['seeds']:
  btbase=np.random.default_rng(seed).choice(base_tr,len(base_tr),replace=True);bt=(btbase[:,None]*2+np.arange(2)).ravel()
  def score(p):
   rr=retrieval(p,H,bid[dv],'seen');return (-np.mean([r['recall1'] for r in rr]),-np.mean([r['mrr'] for r in rr]),float(mse(p,Y[dv]).mean()))
  mm,_,cc=fit_suite(X,Y,fl,ft,bt,dv,seed,score,shuffle_unit=2)
  for method in ['identity','shift','rank1','lowrank','full_affine','length_shift','source_token_ridge']:
   pred=mm[method](cx,cfl,cft)
   for i,r in enumerate(challenge):
    cand=r['neg']+[r['pos']]+r['wrong'];sims=normalize(H[[LOOKUP[s] for s in cand]])@normalize(pred[i]);rank=1+np.sum(sims[2:]>=sims[:2].max()-1e-12)
    out.append(dict(representation=rep,seed=seed,method=method,id=r['id'],category=r['category'],source=r['pos'],winner=cand[int(np.argmax(sims))],correct_negatives=r['neg'],rank=int(rank),recall1=float(rank==1),cosines=sims.tolist()))
(R/'results/challenge_outputs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
# C candidate-composition ablation: same independent examples, same frozen operator, 97 instead of 5 candidates.
old=json.loads((P/'data/pairs.json').read_text());texts=json.loads((P/'data/texts.json').read_text());ol={s:i for i,s in enumerate(texts)};OH=np.load(P/'cache/l28_last.npy').astype(float);H=loadh('l28_last');C=json.loads((R/'data/c_confirmation.json').read_text());ot=np.array([i for i,r in enumerate(old) if r['split']=='train']);X=OH[[ol[r['pos']] for r in old]];Y=OH[[ol[r['neg']] for r in old]];ch=list(csv.DictReader((P/'results/choices.csv').open()));gg=list(csv.DictReader((P/'results/grid.csv').open()));cx=H[[LOOKUP[r['pos']] for r in C]];out=[]
for seed in CFG['seeds']:
 bt=np.random.default_rng(seed).choice(ot,len(ot),replace=True);de=Design(X[bt]);d=Y[bt]-X[bt];cc=next(r for r in ch if r['layer']=='28' and r['pool']=='last' and r['seed']==str(seed));rank=int(cc['rank']);alpha=float(cc['alpha']);ra=float(min([r for r in gg if r['layer']=='28' and r['pool']=='last' and r['seed']==str(seed) and r['rank']=='1'],key=lambda r:float(r['dev_mse']))['alpha'])
 mm={'identity':Predictor('identity'),'shift':Predictor('shift',shift=d.mean(0)),'rank1':Predictor('delta',de,de.fit(d,ra),rank=1),'lowrank':Predictor('delta',de,de.fit(d,alpha),rank=rank),'full_delta':Predictor('delta',de,de.fit(d,float(cc['full_delta_alpha']))),'full_affine':Predictor('affine',de,de.fit(Y[bt],float(cc['full_affine_alpha'])))}
 for method,model in mm.items():
  pred=model(cx)
  for i,r in enumerate(C):
   others=[o for o in C if o['topic']==r['topic'] and o['id']!=r['id']];tt=[r['neg'],r['pos']]+[o['pos'] for o in others]+[o['neg'] for o in others]+[r['scope']];cats=['correct_neg','source_pos']+['other_pos']*len(others)+['other_neg']*len(others)+['scope'];sim=normalize(H[[LOOKUP[s] for s in tt]])@normalize(pred[i]);rank=1+np.sum(sim[1:]>=sim[0]-1e-12)
   out.append(dict(seed=seed,method=method,id=r['id'],topic=r['topic'],cluster=r['topic']+'|'+r['action'],candidate_n=len(tt),chance=1/len(tt),recall1=float(rank==1),mrr=float(1/rank),margin=float(sim[0]-sim[1]),winner_category=cats[int(np.argmax(sim))],correct_cos=float(sim[0]),source_cos=float(sim[1]),other_neg_max_cos=float(sim[2+len(others):-1].max())))
savecsv('c_expanded_candidates.csv',out);print('Supplement complete',flush=True)
