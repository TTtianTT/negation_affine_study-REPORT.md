import os,json,sys,time,hashlib,datetime,difflib,platform,importlib.metadata
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer,AutoModel
R=Path(__file__).resolve().parents[1];P=R.parent;cfg=json.loads((R/'config.json').read_text());texts=json.loads((R/'data/texts.json').read_text());review=json.loads((R/'data/review.json').read_text());assert all(r['status'] in ['AI-reviewed','human-reviewed'] for r in review)
torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;t0=time.time()
assert torch.cuda.device_count()==1,'This task must see only one allocated GPU'
tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True);tok.padding_side='right'
model=AutoModel.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval().cuda();model.requires_grad_(False)
keys=[(l,p) for l,p in cfg['representations']];arrays={k:np.empty((len(texts),3584),np.float32) for k in keys}
@torch.inference_mode()
def batch(tt):
 b=tok(tt,padding=True,add_special_tokens=False,return_tensors='pt');assert b['input_ids'].shape[1]<128
 b={k:v.cuda() for k,v in b.items()};m=b['attention_mask'];ln=m.sum(1);hs=model(**b,output_hidden_states=True,use_cache=False).hidden_states;out={}
 for l,p in keys:
  h=hs[l].float();out[l,p]=(h[torch.arange(len(tt),device='cuda'),ln-1] if p=='last' else (h*m[:,:,None]).sum(1)/ln[:,None]).cpu().numpy()
 return out
# Pilot only development texts, sorted mixed lengths. No performance evaluation.
brows=json.loads((R/'data/b_propositions.json').read_text());pilot=sorted({s for r in brows if r['split'] in ['train','dev'] for s in r['pos']+r['neg'][:3]},key=len)[:12]+sorted({s for r in brows if r['split'] in ['train','dev'] for s in r['pos']+r['neg'][:3]},key=len)[-12:]
pout=batch(pilot);checks=[]
for i in [0,11,23]:
 one=batch([pilot[i]])
 for k in keys:
  err=float(np.linalg.norm(pout[k][i]-one[k][0])/max(np.linalg.norm(one[k][0]),1e-12));checks.append(dict(text=pilot[i],representation=f'l{k[0]}_{k[1]}',relative_error=err));assert err<1e-4,(k,err)
(R/'results/pilot_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
# Exact token embedding mechanism check, before expanding inference.
old=json.loads((P/'data/pairs.json').read_text());aa=json.loads((R/'data/a_edits.json').read_text());cases=[dict(id=r['id'],source=r['pos'],target=r['neg'],dataset='phase1',kind='negation') for r in old]+[dict(id=r['id'],source=r['source'],target=r['target'],dataset='phase2',kind=r['kind']) for r in aa]
alltexts=set(texts)|{r[k] for r in cases for k in ['source','target']};encoded={s:tok.encode(s,add_special_tokens=False) for s in alltexts};uids=sorted({i for ids in encoded.values() for i in ids})
with torch.inference_mode(): emb=model.get_input_embeddings()(torch.tensor(uids,device='cuda')).double().cpu().numpy()
ei={u:i for i,u in enumerate(uids)}
def em(ids):return emb[[ei[i] for i in ids]].sum(0) if ids else np.zeros(3584)
mechanism=[]
oldtext=json.loads((P/'data/texts.json').read_text());oldlookup={s:i for i,s in enumerate(oldtext)};oldh=np.load(P/'cache/l0_mean.npy')
for r in cases:
 ix=encoded[r['source']];iy=encoded[r['target']];n=len(ix);m=len(iy);a=[];c=[];kept=0;changed_blocks=0
 for tag,i,j,k,l in difflib.SequenceMatcher(a=ix,b=iy,autojunk=False).get_opcodes():
  if tag!='equal':a+=ix[i:j];c+=iy[k:l];changed_blocks+=1
  else:kept+=j-i
 x=em(ix)/n;y=em(iy)/m;formula=(n/m-1)*x+(em(c)-em(a))/m;err=np.max(abs((y-x)-formula));assert err<1e-12
 row=dict(id=r['id'],dataset=r['dataset'],kind=r['kind'],n=n,m=m,deleted=len(a),added=len(c),kept=kept,changed_blocks=changed_blocks,max_abs_error=float(err),oracle_target_required=True)
 if r['dataset']=='phase1':row['cached_source_mean_error']=float(np.max(abs(x-oldh[oldlookup[r['source']]])))
 mechanism.append(row)
(R/'results/a_embedding_identity.json').write_text(json.dumps(mechanism,indent=2))
np.savez_compressed(R/'cache/token_embeddings.npz',token_ids=np.array(uids),embeddings=emb)
print('Pilot and embedding identity passed; expanding to',len(texts),'texts',flush=True)
for start in range(0,len(texts),32):
 out=batch(texts[start:start+32])
 for k,v in out.items():arrays[k][start:start+len(v)]=v
 if start%640==0:print('extracted',start,'elapsed',round(time.time()-t0,1),flush=True)
for k,v in arrays.items():np.save(R/f'cache/l{k[0]}_{k[1]}.npy',v)
audit=[dict(text=s,ids=encoded[s],length=len(encoded[s]),last_id=encoded[s][-1]) for s in texts]
assert all(not set(r['ids'])&set(tok.all_special_ids) for r in audit)
(R/'cache/token_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
# Compare FP32 layer-0 cached means against direct embedding sum, not just algebra cancellation.
lookup={s:i for i,s in enumerate(texts)};maxerr=max(float(np.max(abs(arrays[0,'mean'][lookup[r['source']]]-em(encoded[r['source']])/len(encoded[r['source']])))) for r in aa)
meta=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.time()-t0,gpu=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),slurm_job_id=os.getenv('SLURM_JOB_ID'),peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,versions={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','scipy','scikit-learn','matplotlib']},config=cfg,data_sha256=hashlib.sha256((R/'data/texts.json').read_bytes()).hexdigest(),max_layer0_cache_abs_error=maxerr)
assert maxerr<1e-6
(R/'results/environment.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
