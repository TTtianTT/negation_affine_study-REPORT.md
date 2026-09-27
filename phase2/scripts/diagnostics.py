"""Data-mechanism diagnostics; no performance-based sample selection."""
import json,csv,collections,hashlib
import numpy as np
from core import *
edits=json.loads((R/'data/a_edits.json').read_text());out=[]
for kind in ['negation','time','emphasis']:
 rr=[r for r in edits if r['kind']==kind]
 for split in sorted({r['split'] for r in rr}):
  aa=[r for r in rr if r['split']==split];d=dict(kind=kind,split=split,n_pairs=len(aa))
  for k in ['n','m','delta_length','deleted','added','edited','position','matching_cost']:
   z=[r[k] for r in aa];d[k+'_mean']=float(np.mean(z));d[k+'_min']=float(min(z));d[k+'_max']=float(max(z))
  train=[r for r in rr if r['split']=='train'] # token vocabulary is computed from audited source text below
  audit={x['text']:x['ids'] for x in json.loads((R/'cache/token_audit.json').read_text())};vocab={t for r in train for t in audit[r['source']]};testtokens=[t for r in aa for t in audit[r['source']]]
  d['source_token_seen_fraction']=sum(t in vocab for t in testtokens)/len(testtokens);d['predicate_seen_fraction']=np.mean([r['template'] in {z['template'] for z in train} for r in aa]);out.append(d)
savecsv('a_edit_matching.csv',out)
# Exact common support in length-change, total token edits, coarse edit-position bins.
for r in edits:r['stratum']=(r['delta_length'],r['edited'],min(3,int(r['position']*4)))
common=set.intersection(*[{r['stratum'] for r in edits if r['kind']==k} for k in ['negation','time','emphasis']]);support=dict(common_strata=[list(s) for s in sorted(common)],counts={k:sum(r['kind']==k and r['stratum'] in common for r in edits) for k in ['negation','time','emphasis']},interpretation='If common support is empty, no exact matched causal contrast is claimed.')
(R/'results/a_common_support.json').write_text(json.dumps(support,indent=2))
# Length-conditioned per-item results, with no pooling of seeds as independent observations.
if (R/'results/a_per_item.jsonl').exists():
 groups=collections.defaultdict(list)
 for line in (R/'results/a_per_item.jsonl').open():
  r=json.loads(line);key=tuple(r[k] for k in ['experiment','representation','method','split'])+(r['delta_length'],r['edited'],min(3,int(r['position']*4)))
  groups[key].append(r)
 strata=[]
 for key,rr in groups.items():
  d=dict(zip(['experiment','representation','method','split','delta_length','edited','position_quartile'],key));d.update(n_unique_base=len({r['base_id'] for r in rr}),mse=float(np.mean([r['mse'] for r in rr])),identity_mse=float(np.mean([r['identity_mse'] for r in rr])));strata.append(d)
 savecsv('a_stratified_metrics.csv',strata)
# Phase1 geometry: two terms of the exact identity from target-aware token statistics.
em=np.load(R/'cache/token_embeddings.npz');emb=em['embeddings'];ei={int(i):j for j,i in enumerate(em['token_ids'])}
from transformers import AutoTokenizer
import difflib
tok=AutoTokenizer.from_pretrained(CFG['model'],local_files_only=True);old=json.loads((P/'data/pairs.json').read_text());decomposition=[]
for r in old:
 a=tok.encode(r['pos'],add_special_tokens=False);b=tok.encode(r['neg'],add_special_tokens=False);x=emb[[ei[i] for i in a]].mean(0);y=emb[[ei[i] for i in b]].mean(0);d=y-x;length=(len(a)/len(b)-1)*x;edit=d-length
 decomposition.append(dict(id=r['id'],split=r['split'],delta_energy=float(d@d),length_term_energy=float(length@length),edit_term_energy=float(edit@edit),cross_term=float(2*length@edit),note='Terms are correlated, not additive independent explained variances.'))
savecsv('a_phase1_exact_terms.csv',decomposition)
# Read-only phase1 integrity and frozen-data audit.
assert all(hashlib.sha256((P/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in json.loads((R/'phase1_snapshot.json').read_text()))
freeze=json.loads((R/'data_freeze.json').read_text());assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in freeze['files'].items())
(R/'results/integrity.json').write_text(json.dumps(dict(phase1_unchanged=True,frozen_data_unchanged=True),indent=2))
print('Diagnostics complete, common strata:',support,flush=True)
