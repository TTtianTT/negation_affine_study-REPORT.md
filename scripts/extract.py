import os,json,time,hashlib,platform,importlib.metadata
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer,AutoModel
R=Path(__file__).resolve().parents[1]; cfg=json.loads((R/'config.json').read_text()); texts=json.loads((R/'data/texts.json').read_text())
torch.set_num_threads(4); torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
t0=time.time(); tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True); tok.padding_side='right'
model=AutoModel.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval().to('cuda')
model.requires_grad_(False)
arrays={(l,p):np.empty((len(texts),model.config.hidden_size),np.float32) for l in cfg['layers'] for p in cfg['pooling']}
audit=[]
@torch.inference_mode()
def batch(tt):
    b=tok(tt,padding=True,add_special_tokens=False,return_tensors='pt'); assert b['input_ids'].shape[1]<128
    b={k:v.to('cuda') for k,v in b.items()}; m=b['attention_mask']; lengths=m.sum(1)
    out=model(**b,output_hidden_states=True,use_cache=False).hidden_states
    result={}
    for l in cfg['layers']:
        h=out[l].float(); result[l,'last']=h[torch.arange(len(tt),device='cuda'),lengths-1].cpu().numpy(); result[l,'mean']=((h*m[:,:,None]).sum(1)/lengths[:,None]).cpu().numpy()
    return result,b,lengths
for start in range(0,len(texts),32):
    tt=texts[start:start+32]; out,b,ln=batch(tt)
    for k,v in out.items(): arrays[k][start:start+len(tt)]=v
    for j,s in enumerate(tt):
        ids=b['input_ids'][j,:ln[j]].tolist(); assert not set(ids)&set(tok.all_special_ids)
        audit.append(dict(text=s,length=len(ids),last_id=ids[-1],last_token=tok.convert_ids_to_tokens(ids[-1]),last_index=len(ids)-1))
    if start%320==0: print(f'extracted {start+len(tt)}/{len(texts)} elapsed={time.time()-t0:.1f}s',flush=True)
# Real padding-invariance check with different batch composition, allow BF16 numeric differences.
checks=[]
for j in [0,31,70]:
    one,_,_=batch([texts[j]])
    for k in arrays:
        a=arrays[k][j]; b=one[k][0]; err=float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-8)); checks.append(dict(index=j,layer=k[0],pool=k[1],relative_error=err)); assert err<0.0001,(j,k,err)
for (l,p),v in arrays.items(): np.save(R/f'cache/l{l}_{p}.npy',v)
(R/'cache/token_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
(R/'results/extraction_checks.json').write_text(json.dumps(checks,indent=2))
manifest=[]
for p in sorted(Path(cfg['model']).glob('*')):
    if p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
        manifest.append(dict(name=p.name,size=p.stat().st_size,sha256=h.hexdigest()))
meta=dict(seconds=time.time()-t0,gpu=torch.cuda.get_device_name(),peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,python=platform.python_version(),versions={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','scipy','scikit-learn','matplotlib']},model_files=manifest,data_sha256=hashlib.sha256((R/'data/texts.json').read_bytes()).hexdigest(),config=cfg)
(R/'results/environment.json').write_text(json.dumps(meta,indent=2)); print(json.dumps({k:v for k,v in meta.items() if k not in ['config','model_files']},indent=2),flush=True)
