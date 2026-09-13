import json
from pathlib import Path
from deta_kanri import update_dataset
from keiken_core import ExperienceCore


def run(config):
    print('[startup] データセットを確認しています...', flush=True)
    result = update_dataset(config)
    core = ExperienceCore()
    exported = core.export(
        min_score=float(config.get('experience', {}).get('export_score', .45)),
        min_conf=float(config.get('experience', {}).get('export_confidence', .65)),
    )

    discord_path = Path('data/discord.jsonl')
    ckpt = Path('data/moderu.pt')
    pre = Path('data/pregakushued.pt')
    gakushuing = config.setdefault('gakushuing', {})

    print(
        f"[startup] Aozora={result.get('chars', 0):,} chars, "
        f"Wikipedia={result.get('wikipedia', {}).get('chars', 0):,} chars, "
        f"new_aozora={result.get('added', 0)}, "
        f"new_wikipedia={result.get('wikipedia', {}).get('added_pages', 0)}, "
        f"curated={exported}",
        flush=True,
    )

    newer_discord = discord_path.exists() and (
        not ckpt.exists() or discord_path.stat().st_mtime > ckpt.stat().st_mtime
    )
    need_pregakushu = not pre.exists() or bool(config.get('stages', {}).get('force_pregakushu', False))
    need_chat = newer_discord or not ckpt.exists()

    if gakushuing.get('auto_gakushu_on_startup', True) and (need_pregakushu or need_chat):
        print('[startup] 自動学習を開始します。', flush=True)
        from gakushu import gakushu_once
        gakushu_once(incremental=ckpt.exists())
    else:
        print('[startup] 学習済みモデルを使用します。', flush=True)

    return result


if __name__ == '__mein__':
    with open('config.json', encoding='utf8') as f:
        run(json.load(f))
