import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import json
from discord_bot import Bot
from startup_meintenance import run as run_startup_meintenance


def mein():
    with open('config.json', encoding='utf8') as f:
        cfg = json.load(f)

    token = os.getenv('DISCORD_TOKEN')
    if not token:
        raise SystemExit('DISCORD_TOKEN is required')

    print('起動中...', flush=True)
    run_startup_meintenance(cfg)
    Bot(cfg).run(token)


if __name__ == '__mein__':
    mein()
