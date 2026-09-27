"""Exact ridge reduced-rank regression, held-out selection, independent readout probes."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ.setdefault(k,'4')
import json,csv,time,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,svd
R=Path(__file__).resolve().parents[1]; cfg=json.loads((R/'config.json').read_text()); rows=json.loads((R/'data/pairs.json').read_text()); texts=json.loads((R/'data/texts.json').read_text()); lookup={s:i for i,s in enumerate(texts)}
idx={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in sorted({r['split'] for r in rows})}
assert json.loads((R/'results/environment.json').read_text())['data_sha256']==hashlib.sha256((R/'data/texts.json').read_bytes()).hexdigest(), 'stale representation cache'
metrics=[]; grid=[]; spectra=[]; probes=[]; pairs=[]; choices=[]
def writecsv(name,rr):
    if not rr:return
    with (R/'results'/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def norm(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def err(x,y): return np.mean((x-y)**2,axis=1)
def evaluate(pred,ii,variant,tag,meta):
    x=XP[ii] if variant=='paraphrase' else X[ii]; y=YP[ii] if variant=='paraphrase' else Y[ii]
    candidates=[]
    for j in ii:
        r=rows[j]; other=next(z for z in rows if z['id']==r['other_id'])
        keys=[r['neg_para'] if variant=='paraphrase' else r['neg'], r['pos_para'] if variant=='paraphrase' else r['pos'],other['pos'],other['neg'],r['scope']]
        assert len(set(keys))==len(keys)
        candidates.append([lookup[z] for z in keys])
    c=H[np.array(candidates)]; sims=np.einsum('nd,nkd->nk',norm(pred),norm(c)); rank=1+np.sum(sims[:,1:]>=sims[:,:1]-1e-12,axis=1) # pessimistic ties
    losses=err(pred,y); baseline=err(x,y); margin=sims[:,0]-sims[:,1]
    result=dict(**meta,split=tag,variant=variant,n=len(ii),mse=float(losses.mean()),relative_improvement=float(1-losses.mean()/baseline.mean()) if baseline.mean()>1e-20 else float('nan'),cosine=float(np.sum(norm(pred)*norm(y),axis=1).mean()),recall1=float(np.mean(rank==1)),mrr=float(np.mean(1/rank)),neg_over_pos=float(np.mean(margin>1e-12)))
    metrics.append(result)
    if meta['layer']==14 and meta['pool']=='mean' and not meta['method'].startswith('rank'):
        for j,l,b,rr,ma,ss in zip(ii,losses,baseline,rank,margin,sims):pairs.append(dict(**{k:v for k,v in meta.items() if k!='rank'},operator_rank=meta['rank'],split=tag,variant=variant,id=rows[j]['id'],cluster=rows[j]['topic']+'|'+rows[j]['action'],mse=float(l),identity_mse=float(b),rank=int(rr),margin=float(ma),winner=int(np.argmax(ss))))

from linear import Design

for layer in cfg['layers']:
 for pool in cfg['pooling']:
    started=time.time();H=np.load(R/f'cache/l{layer}_{pool}.npy').astype(np.float64)
    X=H[[lookup[r['pos']] for r in rows]];Y=H[[lookup[r['neg']] for r in rows]]; XP=H[[lookup[r['pos_para']] for r in rows]];YP=H[[lookup[r['neg_para']] for r in rows]]
    tr=idx['train'];dv=idx['dev'];D=Y[tr]-X[tr]
    for centered in [False,True]:
        z=D-D.mean(0) if centered else D;s=svd(z,full_matrices=False,compute_uv=False,check_finite=False);total=max(float((s*s).sum()),1e-20)
        for k,val in enumerate(s):spectra.append(dict(layer=layer,pool=pool,centered=centered,component=k+1,singular_value=float(val),energy=float(val*val/total),cumulative=float((s[:k+1]**2).sum()/total)))
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed);bt=rng.choice(tr,len(tr),replace=True); design=Design(X[bt]); delta=Y[bt]-X[bt]; base=dict(layer=layer,pool=pool,seed=seed)
        fits={a:design.fit(delta,a) for a in cfg['alphas']}
        selected={}
        for rank in cfg['ranks']+[None]:
            losses=[]
            for a,fit in fits.items():
                vp=X[dv]+design.predict(X[dv],fit,rank);tp=X[bt]+design.predict(X[bt],fit,rank)
                vl=float(err(vp,Y[dv]).mean());tl=float(err(tp,Y[bt]).mean());losses.append((vl,a))
                grid.append(dict(**base,rank=rank or 'full',alpha=a,dev_mse=vl,train_mse=tl))
            vl,a=min(losses);selected[rank]=(vl,a)
        rr=min(cfg['ranks'],key=lambda r:selected[r][0]);aa=selected[rr][1]
        mu,coef,vt=fits[aa]; basis=vt[:rr].T; left=design.x.T@coef@basis
        np.savez_compressed(R/f'cache/operator_l{layer}_{pool}_s{seed}.npz',input_factor=left,output_factor=basis.T,bias=mu-design.mean@left@basis.T,rank=rr,alpha=aa)
        afits={a:design.fit(Y[bt],a) for a in cfg['alphas']}; af=min(afits,key=lambda a:err(design.predict(X[dv],afits[a]),Y[dv]).mean())
        shuffled=Y[bt][rng.permutation(len(bt))]-X[bt];sfit=design.fit(shuffled,aa)
        rb=np.linalg.qr(rng.normal(size=(H.shape[1],rr)))[0]
        # Same output rank but random orientation. Alpha separately selected on dev.
        ra=min(fits,key=lambda a:err(X[dv]+design.predict(X[dv],fits[a],random_basis=rb),Y[dv]).mean())
        methods={
        'identity':lambda x:x,
        'shift':lambda x:x+delta.mean(0),
        'lowrank':lambda x:x+design.predict(x,fits[aa],rr),
        'full_delta':lambda x:x+design.predict(x,fits[selected[None][1]]),
        'full_affine':lambda x:design.predict(x,afits[af]),
        'shuffled':lambda x:x+design.predict(x,sfit,rr),
        'random_subspace':lambda x:x+design.predict(x,fits[ra],random_basis=rb)}
        for rank in cfg['ranks']:methods[f'rank{rank}']=lambda x,r=rank:x+design.predict(x,fits[selected[r][1]],r)
        for method,predict in methods.items():
            rnk=rr if method in ['lowrank','shuffled','random_subspace'] else (int(method[4:]) if method.startswith('rank') else 0)
            parameters=(2*H.shape[1]*rnk+H.shape[1]) if rnk else (H.shape[1]**2+H.shape[1] if method.startswith('full') else H.shape[1] if method=='shift' else 0)
            if method=='random_subspace': parameters=H.shape[1]*rnk+H.shape[1]
            meta=dict(**base,method=method,rank=rnk,parameters=parameters)
            for split,ii in idx.items(): evaluate(predict(X[ii]),ii,'canonical',split,meta)
            for split in ['iid','topic','template','expression','joint']:evaluate(predict(XP[idx[split]]),idx[split],'paraphrase',split,meta)
        choices.append(dict(**base,rank=rr,alpha=aa,dev_mse=selected[rr][0],full_delta_alpha=selected[None][1],full_affine_alpha=af,random_alpha=ra))
        # Independent linear ridge classification, features and regularization learned on train/dev only.
        px=np.concatenate([X[bt],Y[bt]]); labels=np.r_[np.zeros(len(bt)),np.ones(len(bt))]; pd=Design(px)
        for control in ['real','shuffled_labels']:
            lab=labels if control=='real' else labels[rng.permutation(len(labels))]
            pf={a:pd.fit(lab[:,None],a) for a in cfg['alphas']}
            vx=np.concatenate([X[dv],Y[dv]]);vy=np.r_[np.zeros(len(dv)),np.ones(len(dv))]
            pa=max(cfg['alphas'],key=lambda a:np.mean((pd.predict(vx,pf[a])[:,0]>.5)==vy))
            for split,ii in idx.items():
                for variant,zx,zy in [('canonical',X,Y),('paraphrase',XP,YP)]:
                    tx=np.concatenate([zx[ii],zy[ii]]);ty=np.r_[np.zeros(len(ii)),np.ones(len(ii))]
                    score=float(np.mean((pd.predict(tx,pf[pa])[:,0]>.5)==ty))
                    probes.append(dict(**base,control=control,split=split,variant=variant,accuracy=score,alpha=pa))
        # Challenge set: only clear quantifier/modal/scope examples are valid positive -> negative mappings.
        if layer==14 and pool=='mean':
            challenges=json.loads((R/'data/challenge.json').read_text()); output=[]
            for ch in challenges:
                xx=H[[lookup[ch['pos']]]]; yy=H[[lookup[ch['contrast']],lookup[ch['alternative']],lookup[ch['pos']]]]
                for name in ['identity','shift','lowrank']:
                    sims=(norm(methods[name](xx))@norm(yy).T)[0]
                    output.append(dict(seed=seed,method=name,**ch,scores=sims.tolist(),winner=int(np.argmax(sims))))
            (R/f'results/challenge_seed{seed}.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(f'analyzed layer={layer} pool={pool} {time.time()-started:.1f}s',flush=True)
    for filename,rr in [('metrics.csv',metrics),('grid.csv',grid),('spectra.csv',spectra),('probes.csv',probes),('choices.csv',choices),('primary_pairs.csv',pairs)]:writecsv(filename,rr)
print('DONE',flush=True)
