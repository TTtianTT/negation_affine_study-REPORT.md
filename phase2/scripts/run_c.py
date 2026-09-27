"""Frozen phase1 operators: exact reproduction -> fresh matched confirmation -> strict B transfer."""
import json,csv,time
import numpy as np
from scipy.linalg import svd
from core import *
old=json.loads((P/'data/pairs.json').read_text());ot=json.loads((P/'data/texts.json').read_text());ol={s:i for i,s in enumerate(ot)};OH=np.load(P/'cache/l28_last.npy').astype(float);NH=loadh('l28_last');new=json.loads((R/'data/c_confirmation.json').read_text());tr=np.array([i for i,r in enumerate(old) if r['split']=='train']);te=np.array([i for i,r in enumerate(old) if r['split']=='joint']);X=OH[[ol[r['pos']] for r in old]];Y=OH[[ol[r['neg']] for r in old]]
orig_metrics=list(csv.DictReader((P/'results/metrics.csv').open()));orig_choices=list(csv.DictReader((P/'results/choices.csv').open()));orig_grid=list(csv.DictReader((P/'results/grid.csv').open()))
mean=np.concatenate([X[tr],Y[tr]]).mean(0);_,_,vt=svd(np.concatenate([X[tr],Y[tr]])-mean,full_matrices=False);pc=vt[0]
metrics=[];items=[];bitems=[];reproduction=[]
def geom(a,kind):
 if kind=='raw':return a
 z=a-mean
 return z if kind=='train_centered' else z-(z@pc)[...,None]*pc

def evaluate(pred,source,targets,H,cands,base_rows,branch,geometry,seed,method):
 for i,(pr,x,y,c,r) in enumerate(zip(pred,source,targets,cands,base_rows)):
  transformed=geom(np.array(c),geometry);sim=normalize(transformed)@normalize(geom(pr,geometry));si=normalize(transformed)@normalize(geom(x,geometry));best=sim[0];rank=1+np.sum(sim[1:]>=best-1e-12)
  d=dict(branch=branch,geometry=geometry,seed=seed,method=method,id=r['id'],topic=r['topic'],cluster=r['topic']+'|'+r.get('action',r['id']),mse=float(mse(pr,y)),recall1=float(rank==1),mrr=float(1/rank),margin=float(sim[0]-sim[1]),pred_norm=float(np.linalg.norm(pr)),source_norm=float(np.linalg.norm(x)),target_norm=float(np.linalg.norm(y)),winner=int(np.argmax(sim)),candidate_n=len(sim),chance=1/len(sim))
  for j,cat in enumerate(['correct_neg','source_pos','other_pos','other_neg','scope']):d[cat+'_cos']=float(sim[j]);d[cat+'_change_vs_identity']=float(sim[j]-si[j])
  items.append(d)
 for g in [items[-len(base_rows):]]:
  metrics.append(dict(branch=branch,geometry=geometry,seed=seed,method=method,n=len(g),**{k:float(np.mean([r[k] for r in g])) for k in ['mse','recall1','mrr','margin','pred_norm','source_norm','target_norm','chance']}))

def five(rows,H,look):
 by={r['id']:r for r in rows};out=[]
 for r in rows:
  o=by[r['other_id']];out.append(H[[look[t] for t in [r['neg'],r['pos'],o['pos'],o['neg'],r['scope']]]])
 return out
oldc=five(old,OH,ol);newc=five(new,NH,LOOKUP);NX=NH[[LOOKUP[r['pos']] for r in new]];NY=NH[[LOOKUP[r['neg']] for r in new]]
for seed in CFG['seeds']:
 bt=np.random.default_rng(seed).choice(tr,len(tr),replace=True);des=Design(X[bt]);d=Y[bt]-X[bt];ch=next(r for r in orig_choices if r['layer']=='28' and r['pool']=='last' and r['seed']==str(seed));lr=int(ch['rank']);la=float(ch['alpha']);fa=float(ch['full_affine_alpha']);da=float(ch['full_delta_alpha']);rr=[r for r in orig_grid if r['layer']=='28' and r['pool']=='last' and r['seed']==str(seed) and r['rank']=='1'];ra=float(min(rr,key=lambda r:float(r['dev_mse']))['alpha'])
 models={'identity':Predictor('identity'),'shift':Predictor('shift',shift=d.mean(0)),'rank1':Predictor('delta',des,des.fit(d,ra),rank=1),'lowrank':Predictor('delta',des,des.fit(d,la),rank=lr),'full_delta':Predictor('delta',des,des.fit(d,da)),'full_affine':Predictor('affine',des,des.fit(Y[bt],fa))}
 for name,m in models.items():
  # All original choices frozen; no phase2 labels used here.
  oldname='rank1' if name=='rank1' else name
  evaluate(m(X[te]),X[te],Y[te],OH,[oldc[i] for i in te],[old[i] for i in te],'phase1_reproduction','raw',seed,name)
  actual=metrics[-1];ref=next(r for r in orig_metrics if r['layer']=='28' and r['pool']=='last' and r['seed']==str(seed) and r['method']==oldname and r['split']=='joint' and r['variant']=='canonical')
  errors={k:abs(actual[k]-float(ref[k])) for k in ['mse','recall1','mrr']};assert max(errors.values())<1e-7,(name,seed,errors);reproduction.append(dict(seed=seed,method=name,errors=errors))
  npred=m(NX)
  for geometry in ['raw','train_centered','train_pc1_removed']:evaluate(npred,NX,NY,NH,newc,new,'new_matched_confirmation',geometry,seed,name)
  # Frozen operator transferred into larger multi-positive B pools; no retraining.
  src=[s for r in B for s in r['pos']];bx=NH[[LOOKUP[s] for s in src]];base_ids=np.repeat(np.arange(len(B)),2);bp=m(bx)
  for mode in ['seen','unseen','all']:
   out=retrieval(bp,NH,base_ids,mode,detail=True)
   for j,r in enumerate(out):
    b=B[base_ids[j]]
    bitems.append(dict(seed=seed,method=name,mode=mode,id=b['id'],topic=b['topic'],cluster=b['topic']+'|'+b['template'],split=b['split'],source_variant=j%2,**r))
 print('C seed',seed,'complete',flush=True)
savecsv('c_metrics.csv',metrics);savecsv('c_per_item.csv',items);savecsv('c_frozen_b_per_query.csv',bitems)
(R/'results/phase1_reproduction.json').write_text(json.dumps(reproduction,indent=2))
# Quantify original rank1 gap recovered by the already-run source-length baseline.
p1s=list(csv.DictReader((P/'results/summary.csv').open()));p1surface=list(csv.DictReader((P/'results/surface_operator.csv').open()));out=[]
for split in ['iid','topic','template','expression','joint']:
 def get(m):return float(next(r['mse'] for r in p1s if r['layer']=='14' and r['pool']=='mean' and r['split']==split and r['variant']=='canonical' and r['method']==m))
 sh=get('shift');rk=get('rank1');length=np.mean([float(r['mse']) for r in p1surface if r['split']==split and r['variant']=='canonical' and r['mode']=='length_only']);out.append(dict(split=split,shift=sh,rank1=rk,length=float(length),fraction_of_rank1_mse_gain_recovered=(sh-length)/(sh-rk) if abs(sh-rk)>1e-12 else float('nan'),interpretation='descriptive ratio, not causal variance decomposition; can exceed 1 or be undefined'))
savecsv('phase1_surface_gain.csv',out)
