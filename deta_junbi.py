import json 
from pathlib import Path 
from deta_kanri import update_dataset 

p =Path ('config.json')
c =json .loads (p .read_text (encoding ='utf8'))
c .setdefault ('dataset',{}).setdefault ('wikipedia',{})['auto_fetch']=True 
r =update_dataset (c )
print (json .dumps (r ,ensure_ascii =False ,indent =2 ))
print ('事前学習コーパスの準備が完了しました。次に python gakushu.py を実行してください。')
