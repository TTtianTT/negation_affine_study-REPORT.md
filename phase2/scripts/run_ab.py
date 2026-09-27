import time,json
import numpy as np
from scipy.linalg import svd
from core import *
metrics=[];per=[];grid=[];choices=[];spectra=[];start=time.time()
(R/'results/a_per_item.jsonl').write_text('')
# Candidate membership fixed before fitting and identical for all methods.
(R/'data/candidates.json').write_text(json.dumps(CAND,ensure_ascii=False,indent=2))
base_tr=np.array([i for i,r in enumerate(B) if r['split']=='train']);base_dv=np.array([i for i,r in enumerate(B) if r['split']=='dev']);source=[s for r in B for s in r['pos']];tr=(base_tr[:,None]*2+np.arange(2)).ravel();dv=(base_dv[:,None]*2+np.arange(2)).ravel();base_ids=np.repeat(np.arange(len(B)),2)
FL,FT,vocab=features(source,tr);(R/'results/source_feature_vocabulary.json').write_text(json.dumps(dict(token_ids=vocab,input_features=len(vocab)+3,fit_split='train only'),indent=2))
for layer,pool in CFG['representations']:
 rep=f'l{layer}_{pool}';H=loadh(rep);X=H[[LOOKUP[s] for s in source]];Yneg=H[np.array([[LOOKUP[s] for s in r['neg']] for r in B])];Y=np.repeat(Yneg[:,:3].mean(1),2,axis=0)
 # Fixed negation task, shared multi-target mean and known-form conditional branches.
 for seed in CFG['seeds']:
  btbase=np.random.default_rng(seed).choice(base_tr,len(base_tr),replace=True);bt=(btbase[:,None]*2+np.arange(2)).ravel()
  def score(p):
   rr=retrieval(p,H,base_ids[dv],'seen');return (-np.mean([r['recall1'] for r in rr]),-np.mean([r['mrr'] for r in rr]),float(mse(p,Y[dv]).mean()))
  methods,gg,cc=fit_suite(X,Y,FL,FT,bt,dv,seed,score,shuffle_unit=2)
  meta=dict(experiment='B_shared',representation=rep,seed=seed)
  grid += [dict(**meta,**r) for r in gg];choices += [dict(**meta,**r) for r in cc]
  # Surface baseline is selected on dev/seen only; preserve the selected name.
  surface=min(['length_shift','source_token_ridge'],key=lambda name:score(methods[name](X[dv],FL[dv],FT[dv])))
  choices.append(dict(**meta,method='surface_selected',rank=0,alpha=-1,selected_name=surface))
  # Make choice rows homogeneous, added field for all existing entries.
  for r in choices:r.setdefault('selected_name','')
  methods['surface_selected']=methods[surface]
  for name,predict in methods.items():
   pred=predict(X,FL,FT)
   for mode in ['seen','unseen','all']:
    rr=retrieval(pred,H,base_ids,mode,detail=True);forms={'seen':[0,1,2],'unseen':[3,4],'all':list(range(5))}[mode]
    for j,z in enumerate(rr):
     b=B[base_ids[j]];loss=float(np.mean((pred[j]-Yneg[base_ids[j],forms])**2));per.append(dict(**meta,method=name,split=b['split'],mode=mode,id=b['id'],topic=b['topic'],cluster=b['topic']+'|'+b['template'],source_variant=j%2,mse_to_text_set=loss,**z))
    for split in sorted({r['split'] for r in B}):
     ii=[j for j,b in enumerate(base_ids) if B[b]['split']==split];sel=[rr[j] for j in ii]
     metrics.append(dict(**meta,method=name,split=split,mode=mode,n_base=len(ii)//2,mse=float(np.mean([np.mean((pred[j]-Yneg[base_ids[j],forms])**2) for j in ii])),**{k:float(np.mean([r[k] for r in sel])) for k in ['recall1','mrr','margin','candidate_n','positive_n','chance']}))
  # Per-form predictors use only seen form targets; no fitted head for unseen forms.
  for f in CFG['seen_forms']:
   yf=np.repeat(Yneg[:,f],2,axis=0);mm,gg,cc=fit_suite(X,yf,FL,FT,bt,dv,seed,shuffle_unit=2);meta2=dict(experiment=f'B_conditional_f{f}',representation=rep,seed=seed)
   grid += [dict(**meta2,**r) for r in gg];choices += [dict(**meta2,**r,selected_name='') for r in cc]
   for name,model in mm.items():
    pred=model(X,FL,FT)
    for split in sorted({r['split'] for r in B}):
     ii=[j for j,b in enumerate(base_ids) if B[b]['split']==split]
     metrics.append(dict(**meta2,method=name,split=split,mode=f'specified_f{f}',n_base=len(ii)//2,mse=float(mse(pred[ii],yf[ii]).mean()),recall1=float('nan'),mrr=float('nan'),margin=float('nan'),candidate_n=0.,positive_n=0.,chance=float('nan')))
  print('B',rep,seed,'seconds',round(time.time()-start,1),flush=True)
 # A paired edit diagnostics, with exactly the same source for each kind.
 edits=json.loads((R/'data/a_edits.json').read_text())
 for kind in ['negation','time','emphasis']:
  ee=[r for r in edits if r['kind']==kind];xa=H[[LOOKUP[r['source']] for r in ee]];ya=H[[LOOKUP[r['target']] for r in ee]];fl,ft,_=features([r['source'] for r in ee],base_tr)
  dd=ya[base_tr]-xa[base_tr]
  for centered in [False,True]:
   ss=svd(dd-dd.mean(0) if centered else dd,full_matrices=False,compute_uv=False);tot=max(float(np.sum(ss**2)),1e-30)
   for j,v in enumerate(ss):spectra.append(dict(representation=rep,kind=kind,centered=centered,component=j+1,singular_value=float(v),energy=float(v*v/tot),cumulative=float(np.sum(ss[:j+1]**2)/tot)))
  for seed in CFG['seeds']:
   bt=np.random.default_rng(seed).choice(base_tr,len(base_tr),replace=True);mm,gg,cc=fit_suite(xa,ya,fl,ft,bt,base_dv,seed);meta=dict(experiment='A_'+kind,representation=rep,seed=seed)
   grid += [dict(**meta,**r) for r in gg];choices += [dict(**meta,**r,selected_name='') for r in cc]
   for name,model in mm.items():
    pred=model(xa,fl,ft);loss=mse(pred,ya)
    for split in sorted({r['split'] for r in ee}):
     ii=[j for j,r in enumerate(ee) if r['split']==split]
     metrics.append(dict(**meta,method=name,split=split,mode='specified_edit',n_base=len(ii),mse=float(loss[ii].mean()),recall1=float('nan'),mrr=float('nan'),margin=float('nan'),candidate_n=0.,positive_n=0.,chance=float('nan')))
    # A strata persisted separately to avoid mixing semantic retrieval units.
    with (R/'results/a_per_item.jsonl').open('a') as fp:
     for j,r in enumerate(ee):fp.write(json.dumps(dict(**meta,method=name,id=r['id'],base_id=r['base_id'],split=r['split'],topic=r['topic'],cluster=r['topic']+'|'+r['template'],mse=float(loss[j]),identity_mse=float(mse(xa[j],ya[j])),n=r['n'],m=r['m'],delta_length=r['delta_length'],edited=r['edited'],position=r['position'],matching_cost=r['matching_cost']))+'\n')
  print('A',rep,kind,'seconds',round(time.time()-start,1),flush=True)
 for name,rr in [('ab_metrics.csv',metrics),('b_per_query.csv',per),('selection_grid.csv',grid),('choices.csv',choices),('a_spectra.csv',spectra)]:savecsv(name,rr)
print('AB complete',time.time()-start,flush=True)
