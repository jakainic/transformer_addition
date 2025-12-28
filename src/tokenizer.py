from typing import List

# -----------------------
# Character-level tokenizer
# -----------------------
class CharacterTokenizer:
    def __init__(self):
        self.vocab = ["<pad>", "<bos>", "<eos>"] + list("0123456789 +=")
        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    def encode(self, s: str, add_bos=False, add_eos=False) -> List[int]:
        ids = []
        if add_bos: ids.append(self.bos_id)
        ids += [self.stoi[ch] for ch in s]
        if add_eos: ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i == self.eos_id: break
            if i in (self.pad_id, self.bos_id): continue
            out.append(self.itos[i])
        return "".join(out)

tok = CharacterTokenizer()
