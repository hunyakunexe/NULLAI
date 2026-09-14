



import os 
os .environ .setdefault ('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os .environ .setdefault ('PYTORCH_ALLOC_CONF','expandable_segments:True')

import json ,discord 

from discord_bot import Bot 
from kido_ijika import run as run_startup_maintenance 









if __name__ =='__main__':
    with open ('config.json',encoding ='utf8')as f :
        cfg =json .load (f )

    t =os .getenv ('DISCORD_TOKEN')
    if not t :
        raise SystemExit ('DISCORD_TOKEN is required')


    run_startup_maintenance (cfg )
    Bot (cfg ).run (t )
