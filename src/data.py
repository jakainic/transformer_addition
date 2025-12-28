import random

import torch
from torch.utils.data import Dataset

# -----------------------
# 2) Synthetic datasets
# -----------------------
def _choose_digits_given_carry(c_in: int, c_out: int, leading_dig: bool = False):
    """
    Sample (a_digit, b_digit) in [0..9]^2 such that:
      (a_digit + b_digit + c_in >= 10)  <=>  c_out == 1
    """
    assert c_in in (0, 1) and c_out in (0, 1)

    if c_out == 0:
        # need a+b+c_in <= 9  -> a+b <= 9 - c_in
        s_min = 0 if not leading_dig else 2
        s_max = 9 - c_in
    else:
        # need a+b+c_in >= 10 -> a+b >= 10 - c_in
        s_min, s_max = 10 - c_in, 18 - c_in

    s = random.randint(s_min, s_max)  # desired a+b
    a_min = max(0, s - 9) if not leading_dig else max(1, s-9)
    a_max = min(9, s) if not leading_dig else min(9, s-1)
    a = random.randint(a_min, a_max)
    b = s - a
    return a, b

def _digits_to_int(digits_lsd_first):
    n = 0
    for i, d in enumerate(digits_lsd_first):
        n += d * (10 ** i)
    return n

def _make_carry_pattern(k: int, mode: str):
    """
    Returns a list c_out of length k with entries in {0,1}.
    c_out[i] is carry-out from column i (i=0 is ones place).
    """
    if mode == "no_carry":
        return [0] * k
    if mode == "max_carry":
        return [1] * k
    if mode == "mid_carry":
        m = k // 2
        pattern = [0] * k
        for i in random.sample(range(k), m):
            pattern[i] = 1
        return pattern
    raise ValueError(f"Unknown carry mode: {mode}")

def sample_pair(k: int, mode="uniform"):
    """
    Sample two k-digit integers (a,b).

    modes:
      - "uniform": a,b uniform in [10^(k-1), 10^k - 1]
      - "no_carry": force 0 carries across k columns
      - "max_carry": force a carry out of every column
      - "mid_carry": force exactly floor(k/2) carry-outs (random columns)
    """
    lo, hi = 10**(k-1), 10**k - 1

    if mode == "uniform":
        return random.randint(lo, hi), random.randint(lo, hi)

    c_out = _make_carry_pattern(k, mode)

    a_digits = []
    b_digits = []
    c_in = 0

    for i in range(k):  # loop over columns
        leading_dig = (i == k-1)
        ai, bi = _choose_digits_given_carry(c_in=c_in, 
                                            c_out=c_out[i], 
                                            leading_dig=leading_dig)
            
        a_digits.append(ai)
        b_digits.append(bi)
        c_in = c_out[i]

    a = _digits_to_int(a_digits)
    b = _digits_to_int(b_digits)

    return a, b

def format_example(a: int, b: int, k: int):
    prompt = f"{a:0{k}d} + {b:0{k}d} ="
    ans = str(a + b)
    return prompt, ans

class AddDataset(Dataset):
    def __init__(self, n: int, k: int, mode="uniform", max_len=64):
        self.n = n
        self.k = k
        self.mode = mode
        self.max_len = max_len

    def __len__(self): return self.n

    def __getitem__(self, idx):
        a, b = sample_pair(self.k, self.mode)
        prompt, ans = format_example(a, b, self.k)

        # add a leading space before answer (optional but common)
        full = prompt + " " + ans
        input_ids = tok.encode(full, add_bos=True, add_eos=True)

        # labels: ignore prompt; supervise answer+eos
        prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
        labels = [-100] * len(prompt_ids)
        labels += input_ids[len(prompt_ids):]

        # pad to max_len
        input_ids = input_ids[:self.max_len]
        labels = labels[:self.max_len]

        attn = [1]*len(input_ids)
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [tok.pad_id]*pad_len
            labels += [-100]*pad_len
            attn += [0]*pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
