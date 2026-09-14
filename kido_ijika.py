import json 
from pathlib import Path 
from deta_kanri import update_dataset 
from keiken_core import ExperienceCore 

def run (config ):
    result =update_dataset (config )
    core =ExperienceCore ()
    exported =core .export (min_score =float (config .get ('experience',{}).get ('export_score',.45 )),min_conf =float (config .get ('experience',{}).get ('export_confidence',.65 )))
    discord_path =Path ('data/discord.jsonl');ckpt =Path ('data/moderu.pt');pre =Path ('data/pregakushued.pt')
    print (f"[startup] pretrain corpus={result .get ('chars',0 ):,} chars, added={result .get ('added',0 )}, failures={result .get ('failures',0 )}, curated_experiences={exported }")
    training =config .get ('training',{})
    newer =discord_path .exists ()and (not ckpt .exists ()or discord_path .stat ().st_mtime >ckpt .stat ().st_mtime )
    if training .get ('auto_train_on_startup',True )and (result .get ('added',0 )>0 or newer or not ckpt .exists ()or not pre .exists ()):
        from gakushu import train_once 
        train_once (incremental =ckpt .exists ())

if __name__ =='__main__':
    with open ('config.json',encoding ='utf8')as f :run (json .load (f ))
