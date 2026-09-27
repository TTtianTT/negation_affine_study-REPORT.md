"""Additional token-matched non-negation control, selected without performance access."""
import json,difflib,hashlib,datetime
from pathlib import Path
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1];cfg=json.loads((R/'config.json').read_text());tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
b=json.loads((R/'data/b_propositions.json').read_text());ed=json.loads((R/'data/a_edits.json').read_text());ref={r['base_id']:r for r in ed if r['kind']=='negation'}
def stat(s,t):
 x=tok.encode(s,add_special_tokens=False);y=tok.encode(t,add_special_tokens=False);ops=difflib.SequenceMatcher(a=x,b=y,autojunk=False).get_opcodes();change=[(i,j,k,l) for tag,i,j,k,l in ops if tag!='equal'];return (len(y)-len(x),sum(j-i+l-k for i,j,k,l in change),min(i for i,j,k,l in change)/len(x)),len(x),len(y)
rows=[]
for r in b:
 source=r['pos'][0];opts=[source.replace(r['actor']+r['verb']+'了',r['actor']+marker+r['verb']) for marker in ['曾','曾经']]
 z=ref[r['id']];target=min(opts,key=lambda t:abs(stat(source,t)[0][0]-z['delta_length'])+abs(stat(source,t)[0][1]-z['edited'])+2*abs(stat(source,t)[0][2]-z['position']))
 (d,e,p),n,m=stat(source,target);exact=(d==z['delta_length'] and e==z['edited'] and abs(p-z['position'])<1e-12)
 rows.append(dict(id=r['id'],topic=r['topic'],template=r['template'],split=r['split'],source=source,target=target,negative=z['target'],n=n,m=m,delta_length=d,edited=e,position=p,exact_token_matching=exact,semantic_relation='positive past-event marker substitution; not negation'))
p=R/'data/a_matched_supplement.json';p.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
meta=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),data_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),n=len(rows),exact_matched=sum(r['exact_token_matching'] for r in rows),selection='token statistics only; no phase2 performance read',status='exploratory supplement; does not modify primary B/C data or endpoint')
(R/'a_matched_freeze.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2));print(rows[0])
