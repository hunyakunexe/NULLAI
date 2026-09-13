import json
from pathlib import Path
from deta_kanri import update_dataset

p=Path('config.json')
c=json.loads(p.read_text(encoding='utf8'))
c.setdefault('dataset',{}).setdefault('wikipedia',{})['auto_fetch']=True
r=update_dataset(c)
print(json.dumps(r,ensure_ascii=False,indent=2))
