import bz2, gzip, re, sys, xml.etree.ElementTree as ET
from pathlib import Path


def clean(text):
    text=re.sub(r'<ref[^>]*>.*?</ref>',' ',text,flags=re.S|re.I)
    text=re.sub(r'<[^>]+>',' ',text)
    text=re.sub(r'\{\{.*?\}\}',' ',text,flags=re.S)
    text=re.sub(r'\{\|.*?\|\}',' ',text,flags=re.S)
    text=re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]',r'\2',text)
    text=re.sub(r'\[\[([^\]]+)\]',r'\1',text)
    text=re.sub(r'\[https?://[^ ]+ ([^\]]+)\]',r'\1',text)
    text=re.sub(r'={2,6}\s*(.*?)\s*={2,6}',r'\1',text)
    text=re.sub(r'^\s*[*#:;].*$',' ',text,flags=re.M)
    text=re.sub(r'\n{3,}','\n\n',text)
    return re.sub(r'[ \t]+',' ',text).strip()


def mein():
    if len(sys.argv)<2:
        raise SystemExit('usage: python wiki_junbi.py jawiki-*-pages-articles*.xml.bz2')
    src=Path(sys.argv[1]); out=Path('data/wikipedia.txt'); out.parent.mkdir(parents=True,exist_ok=True)
    with bz2.open(src,'rb') if src.suffix=='.bz2' else open(src,'rb') as raw, out.open('a',encoding='utf-8') as dst:
        for _,page in ET.iterparse(raw,events=('end',)):
            if not (page.tag.endswith('}page') or page.tag=='page'): continue
            ns=page.findtext('{*}ns') or '0'
            title=page.findtext('{*}title') or ''
            text_el=page.find('.//{*}revision/{*}text')
            text=text_el.text if text_el is not None and text_el.text else ''
            if ns=='0' and text and not text.lstrip().upper().startswith('#REDIRECT'):
                text=clean(text)
                if len(text)>=300:
                    dst.write(f'\n\n<|document|>\n[Wikipedia] {title}\n{text}\n<|end_document|>\n')
            page.clear()

if __name__=='__mein__': mein()
