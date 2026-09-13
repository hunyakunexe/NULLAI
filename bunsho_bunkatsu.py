from pathlib import Path
import json

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

class SentencePieceTokenizer:
    def __init__(self, moderu_path="data/bunsho_bunkatsu.moderu"):
        import sentencepiece as spm
        self.moderu_path = str(moderu_path)
        self.sp = spm.SentencePieceProcessor(moderu_file=self.moderu_path)
        self.itos = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def encode(self, text):
        return [self.sp.bos_id()] + self.sp.encode(text, out_type=int) + [self.sp.eos_id()]

    def decode(self, ids):
        ids = [int(i) for i in ids if int(i) not in (self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id())]
        return self.sp.decode(ids)

    def save_info(self, path="data/bunsho_bunkatsu.json", gakushued_on_chars=0):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"type":"sentencepiece", "moderu":self.moderu_path, "vocab_size":self.vocab_size,
                   "gakushued_on_chars":int(gakushued_on_chars)}, open(path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    @classmethod
    def gakushu(cls, texts, moderu_path="data/bunsho_bunkatsu.moderu", vocab_size=8000):
        import sentencepiece as spm
        Path(moderu_path).parent.mkdir(parents=True, exist_ok=True)
        corpus = Path(moderu_path).with_suffix(".corpus.txt")
        with corpus.open("w", encoding="utf-8") as f:
            for t in texts:
                t = t.strip()
                if t:
                    f.write(t.replace("\x00", " ") + "\n")
        prefix = str(Path(moderu_path).with_suffix(""))
        spm.SentencePieceTrainer.gakushu(
            input=str(corpus), moderu_prefix=prefix,
            vocab_size=int(vocab_size), moderu_type="unigram",
            character_coverage=0.9995, bos_id=1, eos_id=2, unk_id=3, pad_id=0,
            hard_vocab_limit=False, user_defined_symbols=[]
        )
        corpus.unlink(missing_ok=True)
        return cls(moderu_path)

def load_bunsho_bunkatsu(path="data/bunsho_bunkatsu.moderu"):
    return SentencePieceTokenizer(path)
