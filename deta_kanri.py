from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
import bz2
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_URL = "https://www.aozora.gr.jp"
INDEX_ZIP_URL = f"{BASE_URL}/index_pages/list_person_all_extended_utf8.zip"


def _download(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "DiscordExperienceAI/1.0 dataset updater"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _clean_text(raw: bytes) -> str:
    for enc in ("cp932", "shift_jis", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
                        
    if "---------------------------------------" in text:
        text = text.split("---------------------------------------", 1)[1]
    if "底本：" in text:
        text = text.split("底本：", 1)[0]
    if "底本：" not in text and "［＃底本" in text:
        text = text.split("［＃底本", 1)[0]

                              
    text = re.sub(r"｜([^《]+)《[^》]+》", r"\1", text)
    text = re.sub(r"《[^》]+》", "", text)
    text = re.sub(r"［＃[^］]*］", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_catalog(raw_zip: bytes):
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            text = f.read().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _work_key(row: dict) -> str:
    return str(row.get("作品ID", "")).strip()


def update_dataset(config: dict) -> dict:
    dc = config.get("dataset", {})
    root = Path("data/aozora")
    root.mkdir(parents=True, exist_ok=True)
    corpus = Path("data/pregakushu.txt")
    manifest_path = root / "manifest.json"
    catalog_path = root / "list_person_all_extended_utf8.zip"

    min_chars = int(dc.get("min_corpus_chars", 5_000_000))
    max_new = int(dc.get("max_new_works_per_start", 20))
    timeout = int(dc.get("download_timeout_seconds", 30))
    enabled = bool(dc.get("auto_fill", True))
    if not enabled:
        return {"added": 0, "chars": len(corpus.read_text(encoding="utf-8")) if corpus.exists() else 0, "skipped": True}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"works": {}}
    works = manifest.setdefault("works", {})

                                      
    try:
        catalog = _download(INDEX_ZIP_URL, timeout)
        catalog_path.write_bytes(catalog)
    except Exception as exc:
        print(f"[dataset] 青空文庫カタログ更新失敗: {exc}")
        catalog = catalog_path.read_bytes() if catalog_path.exists() else None
        if catalog is None:
            return {"added": 0, "chars": 0, "error": str(exc)}

    rows = _read_catalog(catalog)
    candidates = {}
    for row in rows:
        wid = _work_key(row)
        url = (row.get("テキストファイルURL") or "").strip()
        if not wid or not url or not url.startswith(BASE_URL):
            continue
        if row.get("作品著作権フラグ") != "なし" or row.get("人物著作権フラグ") != "なし":
            continue
        candidates[wid] = row

    current_chars = len(corpus.read_text(encoding="utf-8")) if corpus.exists() else 0
    added = 0
    failures = 0
                                 
    pending = [r for wid, r in candidates.items() if wid not in works]
    pending.sort(key=lambda r: (r.get("最終更新日", ""), r.get("作品ID", "")), reverse=True)

    with corpus.open("a", encoding="utf-8") as out:
        for row in pending:
            if current_chars >= min_chars or added >= max_new:
                break
            wid = _work_key(row)
            try:
                blob = _download(row["テキストファイルURL"], timeout)
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    txt_name = next(n for n in z.namelist() if not n.endswith("/"))
                    text = _clean_text(z.read(txt_name))
                if len(text) < 100:
                    raise ValueError("本文が短すぎます")
                out.write(f"\n\n<|document|>\n{text}\n<|end_document|>\n")
                current_chars += len(text)
                works[wid] = {
                    "title": row.get("作品名", ""),
                    "updated": row.get("最終更新日", ""),
                    "url": row.get("テキストファイルURL", ""),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "chars": len(text),
                }
                added += 1
                print(f"[dataset] 追加: {row.get('作品名', wid)} ({len(text):,} chars)")
            except Exception as exc:
                failures += 1
                print(f"[dataset] 取得失敗 {wid}: {exc}")
            time.sleep(float(dc.get("request_interval_seconds", 0.2)))

    manifest["catalog_url"] = INDEX_ZIP_URL
    manifest["last_check"] = time.time()
    manifest["target_chars"] = min_chars
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    wiki = update_wikipedia(config)
    return {"added": added, "chars": current_chars, "failures": failures, "target": min_chars, "wikipedia": wiki}


WIKI_BASE = "https://dumps.wikimedia.org/jawiki/latest/"


def _wiki_dump_url():
    import urllib.parse
    try:
        html = _download(WIKI_BASE, timeout=30).decode("utf-8", errors="ignore")
        names = re.findall(r'href=["\']([^"\']+pages-articles(?:-multistream)?\.xml\.bz2)["\']', html)
        names = [n for n in names if n.startswith("jawiki-latest-")]
        if names:
            name = sorted(names, key=lambda x: ("multistream" in x, len(x)), reverse=True)[0]
            return urllib.parse.urljoin(WIKI_BASE, name)
    except Exception:
        pass
    return WIKI_BASE + "jawiki-latest-pages-articles.xml.bz2"


def _clean_wiki_text(text: str) -> str:
                                                                                                    
    text = re.sub(r'<ref[^>]*>.*?</ref>', ' ', text, flags=re.S | re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\{\{.*?\}\}', ' ', text, flags=re.S)
    text = re.sub(r'\{\|.*?\|\}', ' ', text, flags=re.S)
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[https?://[^ ]+ ([^\]]+)\]', r'\1', text)
    text = re.sub(r'={2,6}\s*(.*?)\s*={2,6}', r'\1', text)
    text = re.sub(r'^\s*[*#:;].*$', ' ', text, flags=re.M)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def update_wikipedia(config: dict) -> dict:
    wc = config.get("dataset", {}).get("wikipedia", {})
    if not bool(wc.get("enabled", False)) or not bool(wc.get("auto_fetch", False)):
        return {"enabled": False, "added_pages": 0, "chars": 0}
    root = Path("data/wikipedia")
    root.mkdir(parents=True, exist_ok=True)
    corpus = Path("data/wikipedia.txt")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    url = _wiki_dump_url()
    timeout = int(wc.get("download_timeout_seconds", 120))
    max_pages = int(wc.get("max_pages_per_run", 5000))
    max_chars = int(wc.get("max_chars", 20_000_000))
                                                                                    
    existing = corpus.stat().st_size if corpus.exists() else 0
    if existing >= max_chars:
        return {"enabled": True, "added_pages": 0, "chars": existing, "skipped": True, "url": url}
    if manifest.get("source_url") == url and manifest.get("complete"):
        return {"enabled": True, "added_pages": 0, "chars": existing, "skipped": True, "url": url}

    print(f"[dataset] Wikipedia stream: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "DiscordExperienceAI/1.0 dataset updater"})
    added = 0
    chars = existing
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = bz2.BZ2File(resp)
            with corpus.open("a", encoding="utf-8") as out:
                                                                      
                for event, elem in ET.iterparse(raw, events=("end",)):
                    if elem.tag.endswith("}page") or elem.tag == "page":
                        ns = elem.find("{*}ns")
                        title = elem.findtext("{*}title") or ""
                        text_el = elem.find(".//{*}revision/{*}text")
                        text = text_el.text if text_el is not None and text_el.text else ""
                                                                                         
                        if (ns is None or (ns.text or "0") == "0") and text and not text.lstrip().upper().startswith("#REDIRECT"):
                            clean = _clean_wiki_text(text)
                            if len(clean) >= 300:
                                remein = max_chars - chars
                                if remein <= 0:
                                    break
                                clean = clean[:remein]
                                out.write(f"\n\n<|document|>\n[Wikipedia] {title}\n{clean}\n<|end_document|>\n")
                                chars += len(clean)
                                added += 1
                        elem.clear()
                        if added >= max_pages or chars >= max_chars:
                            break
        manifest.update({"source_url": url, "last_update": time.time(), "pages_added": added, "chars": chars, "complete": False})
    except Exception as exc:
        print(f"[dataset] Wikipedia stream failed: {exc}")
        manifest.update({"source_url": url, "last_error": str(exc), "last_update": time.time(), "pages_added": added, "chars": chars})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"enabled": True, "added_pages": added, "chars": chars, "url": url}
