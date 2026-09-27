import sys,json,csv,collections,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];P=R.parent
sys.path.insert(0,str(P/'scripts'))
from linear import Design
CFG=json.loads((R/'config.json').read_text());B=json.loads((R/'data/b_propositions.json').read_text());TEXTS=json.loads((R/'data/texts.json').read_text());LOOKUP={s:i for i,s in enumerate(TEXTS)}
def savecsv(name,rows):
 if not rows:return
 with (R/'results'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-15)
def mse(x,y):return np.mean((x-y)**2,axis=-1)
def loadh(rep):
 e=json.loads((R/'results/environment.json').read_text());assert e['data_sha256']==hashlib.sha256((R/'data/texts.json').read_bytes()).hexdigest()
 return np.load(R/f'cache/{rep}.npy').astype(np.float64)
def features(source_texts,train_rows):
 audit={r['text']:r for r in json.loads((R/'cache/token_audit.json').read_text())};counts=collections.Counter(i for j in train_rows for i in audit[source_texts[j]]['ids']);vocab=sorted(counts,key=lambda i:(-counts[i],i))[:512];vi={u:i for i,u in enumerate(vocab)}
 lens=[];bag=[]
 for s in source_texts:
  ids=audit[s]['ids'];n=len(ids);lens.append([n,1/n,len(s)]);v=np.zeros(len(vocab))
  for i in ids:
   if i in vi:v[vi[i]]+=1
  bag.append(v)
 fl=np.asarray(lens,float);ft=np.column_stack([fl,np.array(bag)])
 return fl,ft,vocab

def candidates(mode):
 forms={'seen':[0,1,2],'unseen':[3,4],'all':[0,1,2,3,4]}[mode];out=[]
 for r in B:
  others=[z for z in B if z['topic']==r['topic'] and z['split']==r['split'] and z['id']!=r['id']][:11]
  tt=[r['neg'][i] for i in forms]+r['pos'];cat=['correct_neg']*len(forms)+['source_pos']*2
  for o in others:tt+=o['pos']+[o['neg'][i] for i in forms];cat+=['other_pos']*2+['other_neg']*len(forms)
  tt+=r['hard'];cat+=r['hard_categories'];assert len(tt)==len(set(tt))
  out.append(dict(base_id=r['id'],ids=[LOOKUP[t] for t in tt],categories=cat,positive_count=len(forms),chance=len(forms)/len(tt)))
 return out
CAND={m:candidates(m) for m in ['seen','unseen','all']}
_NORMALIZED={}
def retrieval(pred,H,base_ids,mode,detail=False):
 if id(H) not in _NORMALIZED:_NORMALIZED[id(H)]=(H,normalize(H))
 hn=_NORMALIZED[id(H)][1];pn=normalize(pred);out=[]
 for j,bi in enumerate(base_ids):
  c=CAND[mode][int(bi)];sim=hn[c['ids']]@pn[j];k=c['positive_count'];best=sim[:k].max();wrong=sim[k:];rank=1+np.sum(wrong>=best-1e-12);posmax=sim[k:k+2].max();winner=int(np.argmax(sim));d=dict(recall1=float(rank==1),mrr=float(1/rank),margin=float(best-posmax),candidate_n=len(sim),positive_n=k,chance=c['chance'])
  if detail:
   d.update(winner_text=TEXTS[c['ids'][winner]],winner_category=c['categories'][winner],best_correct_cos=float(best),source_cos=float(posmax),pred_norm=float(np.linalg.norm(pred[j])),rank=int(rank))
   for cc in ['other_pos','other_neg','different_subject','modality','scope','different_object']:
    ss=[v for v,t in zip(sim,c['categories']) if t==cc];d[cc+'_max_cos']=float(max(ss)) if ss else float('nan')
  out.append(d)
 return out
class Predictor:
 def __init__(self,kind,design=None,fit=None,rank=None,basis=None,shift=None,feature=None):
  self.kind=kind;self.design=design;self.fit=fit;self.feature=feature;self.shift=shift;self.basis=basis
  if rank is not None:self.basis=fit[2][:rank].T
  self.left=design.x.T@fit[1]@self.basis if self.basis is not None else None
 def __call__(self,x,fl=None,ft=None):
  if self.kind=='identity':return x
  if self.kind=='shift':return x+self.shift
  z=x
  if self.feature:
   src,mu,sd=self.feature;z=((fl if src=='length' else ft)-mu)/sd
  if self.basis is not None:v=((z-self.design.mean)@self.left)@self.basis.T+self.fit[0]
  else:v=self.design.predict(z,self.fit)
  return v if self.kind=='affine' else x+v

def fit_suite(X,Y,FL,FT,bt,dv,seed,score=None,shuffle_unit=1):
 """score uses dev only; default MSE. Return all controls, independently dev-selected."""
 des=Design(X[bt]);delta=Y[bt]-X[bt];rng=np.random.default_rng(seed);grids=[];choices=[]
 if score is None:score=lambda pred:(float(mse(pred,Y[dv]).mean()),)
 def select(method,options):
  scored=[]
  for rank,a,p in options:
   vp=p(X[dv],FL[dv],FT[dv]);sc=tuple(score(vp));loss=float(mse(vp,Y[dv]).mean());tr=float(mse(p(X[bt],FL[bt],FT[bt]),Y[bt]).mean())
   grids.append(dict(method=method,rank=rank or 0,alpha=a,dev_mse=loss,train_mse=tr,selection_score=json.dumps(sc)))
   scored.append((sc+(rank or 0,-a),rank,a,p))
  _,r,a,p=min(scored,key=lambda z:z[0]);choices.append(dict(method=method,rank=r or 0,alpha=a));return p,r,a
 fits={a:des.fit(delta,a) for a in CFG['alphas']}
 methods={'identity':Predictor('identity'),'shift':Predictor('shift',shift=delta.mean(0))}
 for name,ranks in [('rank1',[1]),('lowrank',CFG['ranks']),('full_delta',[None])]:
  methods[name],rank,alpha=select(name,[(r,a,Predictor('delta',des,f,rank=r)) for a,f in fits.items() for r in ranks])
  if name=='lowrank':chosen_rank=rank
 af={a:des.fit(Y[bt],a) for a in CFG['alphas']}
 methods['full_affine'],_,_=select('full_affine',[(None,a,Predictor('affine',des,f)) for a,f in af.items()])
 for name,source,feature in [('length_shift','length',FL),('source_token_ridge','token',FT)]:
  mu=feature[bt].mean(0);sd=feature[bt].std(0);sd[sd<1e-12]=1;fd=Design((feature[bt]-mu)/sd);ff={a:fd.fit(delta,a) for a in CFG['alphas']}
  methods[name],_,_=select(name,[(None,a,Predictor('delta',fd,f,feature=(source,mu,sd))) for a,f in ff.items()])
 group_perm=rng.permutation(len(bt)//shuffle_unit);perm=(group_perm[:,None]*shuffle_unit+np.arange(shuffle_unit)).ravel();shdelta=Y[bt][perm]-X[bt];shfits={a:des.fit(shdelta,a) for a in CFG['alphas']}
 methods['shuffled'],_,_=select('shuffled',[(r,a,Predictor('delta',des,f,rank=r)) for a,f in shfits.items() for r in CFG['ranks']])
 basis=np.linalg.qr(rng.normal(size=(X.shape[1],chosen_rank)))[0]
 methods['random_subspace'],_,_=select('random_subspace',[(chosen_rank,a,Predictor('delta',des,f,basis=basis)) for a,f in fits.items()])
 return methods,grids,choices
