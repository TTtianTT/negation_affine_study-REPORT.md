import json,hashlib,time,sys,os
import numpy as np
from core import *
rows=json.loads((R/'data/a_matched_supplement.json').read_text());target=[r['target'] for r in rows]
if '--extract' in sys.argv:
 import torch
 from transformers import AutoModel,AutoTokenizer
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.device_count()==1
 tok=AutoTokenizer.from_pretrained(CFG['model'],local_files_only=True);tok.padding_side='right';t0=time.time();model=AutoModel.from_pretrained(CFG['model'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval().cuda();model.requires_grad_(False)
 arrays={l:[] for l in [0,14]}
 @torch.inference_mode()
 def batch(tt):
  bb=tok(tt,add_special_tokens=False,padding=True,return_tensors='pt');bb={k:v.cuda() for k,v in bb.items()};mask=bb['attention_mask'];hh=model(**bb,output_hidden_states=True,use_cache=False).hidden_states
  return {l:((hh[l].float()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).cpu().numpy() for l in arrays}
 for start in range(0,len(target),32):
  out=batch(target[start:start+32])
  for l,v in out.items():arrays[l].append(v)
 checks=[]
 for l in arrays:arrays[l]=np.concatenate(arrays[l])
 for j in [0,31,383]:
  out=batch([target[j]])
  for l in arrays:
   e=float(np.linalg.norm(out[l][0]-arrays[l][j])/max(np.linalg.norm(out[l][0]),1e-12));assert e<1e-4;checks.append(dict(layer=l,index=j,relative_error=e))
 for l,v in arrays.items():np.save(R/f'cache/matched_l{l}_mean.npy',v)
 (R/'results/matched_extraction.json').write_text(json.dumps(dict(seconds=time.time()-t0,job_id=os.getenv('SLURM_JOB_ID'),gpu_count=1,checks=checks,data_sha256=hashlib.sha256((R/'data/a_matched_supplement.json').read_bytes()).hexdigest()),indent=2))
 print('Matched extraction completed',flush=True)
else:
 tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);dv=np.array([i for i,r in enumerate(rows) if r['split']=='dev']);src=[r['source'] for r in rows];fl,ft,_=features(src,tr);metrics=[];grid=[];per=[]
 for rep in ['l0_mean','l14_mean']:
  H=loadh(rep);X=H[[LOOKUP[s] for s in src]];Y=np.load(R/f'cache/matched_{rep}.npy').astype(float)
  for seed in CFG['seeds']:
   bt=np.random.default_rng(seed).choice(tr,len(tr),replace=True);mm,gg,cc=fit_suite(X,Y,fl,ft,bt,dv,seed);grid += [dict(representation=rep,seed=seed,**r) for r in gg]
   for name,m in mm.items():
    pred=m(X,fl,ft);loss=mse(pred,Y)
    for split in sorted({r['split'] for r in rows}):
     ii=[i for i,r in enumerate(rows) if r['split']==split];metrics.append(dict(representation=rep,seed=seed,method=name,split=split,n=len(ii),mse=float(loss[ii].mean())))
    for i,r in enumerate(rows):per.append(dict(representation=rep,seed=seed,method=name,id=r['id'],split=r['split'],topic=r['topic'],cluster=r['topic']+'|'+r['template'],mse=float(loss[i]),exact_matched=r['exact_token_matching']))
  from scipy.linalg import svd
  ds=Y[tr]-X[tr];spect=[]
  for cent in [False,True]:
   ss=svd(ds-ds.mean(0) if cent else ds,full_matrices=False,compute_uv=False);ssq=ss*ss
   for i,z in enumerate(ss):spect.append(dict(representation=rep,centered=cent,component=i+1,singular_value=float(z),cumulative=float(ssq[:i+1].sum()/ssq.sum())))
  savecsv(f'matched_spectrum_{rep}.csv',spect)
 savecsv('matched_metrics.csv',metrics);savecsv('matched_grid.csv',grid);savecsv('matched_per_item.csv',per)
 print('Matched analysis complete',flush=True)
