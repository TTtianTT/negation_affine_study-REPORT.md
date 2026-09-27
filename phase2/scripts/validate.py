"""Completeness, immutable inputs, grouping, and one-to-many loss identity checks."""
import csv,json,hashlib,re
import numpy as np
from core import *
expected={'ab_metrics.csv':6696,'b_per_query.csv':304128,'selection_grid.csv':6048,'c_metrics.csv':72,'c_per_item.csv':6912,'c_frozen_b_per_query.csv':41472,'matched_metrics.csv':360,'matched_per_item.csv':23040,'c_expanded_candidates.csv':1728}
checks={}
for name,n in expected.items():
 count=sum(1 for _ in csv.DictReader((R/'results'/name).open()));assert count==n,(name,count,n);checks[name]=count
assert sum(1 for _ in (R/'results/a_per_item.jsonl').open())==138240
checks['A_per_item_count']=138240
frozen=json.loads((R/'data_freeze.json').read_text());assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items());checks['frozen_data_unchanged']=True
assert hashlib.sha256((R/'PHASE2_PROTOCOL.md').read_bytes()).hexdigest()==json.loads((R/'protocol_freeze.json').read_text())['sha256'];checks['protocol_unchanged']=True
for r in json.loads((R/'phase1_snapshot.json').read_text()):assert hashlib.sha256((P/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
checks['phase1_tracked_files_unchanged']=True
for r in json.loads((P/'ARTIFACT_MANIFEST.json').read_text()):
 if r['path'].startswith('cache/'):
  hh=hashlib.sha256()
  with (P/r['path']).open('rb') as fp:
   for chunk in iter(lambda:fp.read(4*1024*1024),b''):hh.update(chunk)
  assert hh.hexdigest()==r['sha256'],r['path']
checks['phase1_all_caches_unchanged']=True
assert hashlib.sha256((R/'data/a_matched_supplement.json').read_bytes()).hexdigest()==json.loads((R/'a_matched_freeze.json').read_text())['data_sha256'];checks['matched_supplement_unchanged']=True
for mode,rows in CAND.items():
 for bi,r in enumerate(rows):
  assert len(r['ids'])==len(set(r['ids']));n=r['positive_count'];forms={'seen':[0,1,2],'unseen':[3,4],'all':[0,1,2,3,4]}[mode]
  assert [TEXTS[j] for j in r['ids'][:n]]==[B[bi]['neg'][f] for f in forms]
checks['candidate_labels_and_uniqueness']=True
# Equivalent one-to-many squared loss checked on actual cached vectors.
H=loadh('l14_mean');yy=H[[LOOKUP[s] for s in B[0]['neg'][:3]]];pred=H[LOOKUP[B[0]['pos'][0]]];left=np.mean((pred-yy)**2);right=np.mean((pred-yy.mean(0))**2)+np.mean((yy-yy.mean(0))**2);assert abs(left-right)<1e-10;checks['centroid_loss_identity_error']=float(abs(left-right))
assert all(float(r['recall1'])==0 for r in csv.DictReader((R/'results/b_per_query.csv').open()) if r['method']=='identity');checks['identity_never_labeled_positive']=True
# Independent reaggregation of main effect from per-query results.
groups={}
for r in csv.DictReader((R/'results/b_per_query.csv').open()):
 if r['representation']=='l14_mean' and r['split']=='iid' and r['mode']=='unseen' and r['method'] in ['lowrank','surface_selected']:
  groups.setdefault((r['id'],r['method']),[]).append(float(r['recall1']))
effect=np.mean([np.mean(groups[b['id'],'lowrank'])-np.mean(groups[b['id'],'surface_selected']) for b in B if b['split']=='iid']);ref=next(float(r['difference']) for r in csv.DictReader((R/'results/paired_intervals.csv').open()) if r['representation']=='l14_mean' and r['split']=='iid' and r['mode']=='unseen' and r['cluster_unit']=='cluster');assert abs(effect-ref)<1e-12;checks['primary_effect_reaggregated']=float(effect)
for doc in ['README.md','PHASE2_REPORT.md']:
 for link in re.findall(r'\]\(([^)]+)\)',(R/doc).read_text()):
  if not link.startswith(('http','#')):assert (R/link).exists(),(doc,link)
checks['report_links_exist']=True
(R/'results/validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
