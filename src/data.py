# ------------------------------------------
# Synthetic datasets: generation + loading
# ------------------------------------------

import argparse, json
from pathlib import Path
import random
from typing import Tuple
import torch
from torch.utils.data import Dataset

from src.utils import seed_everything, make_rng

def _choose_digits_given_carry(rng: random.Random, c_in: int, c_out: int, leading_dig: bool = False):
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

    s = rng.randint(s_min, s_max)  # desired a+b
    a_min = max(0, s - 9) if not leading_dig else max(1, s-9)
    a_max = min(9, s) if not leading_dig else min(9, s-1)
    a = rng.randint(a_min, a_max)
    b = s - a
    return a, b

def _digits_to_int(digits_lsd_first):
    n = 0
    for i, d in enumerate(digits_lsd_first):
        n += d * (10 ** i)
    return n

def _make_carry_pattern(rng: random.Random, k: int, mode: str):
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
        for i in rng.sample(range(k), m):
            pattern[i] = 1
        return pattern
    raise ValueError(f"Unknown carry mode: {mode}")

def sample_pair(rng: random.Random, k: int, mode="uniform"):
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
        return rng.randint(lo, hi), rng.randint(lo, hi)

    c_out = _make_carry_pattern(rng, k, mode)

    a_digits = []
    b_digits = []
    c_in = 0

    for i in range(k):  # loop over columns
        leading_dig = (i == k-1)
        ai, bi = _choose_digits_given_carry(rng=rng,
                                            c_in=c_in, 
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

def write_dataset(path: Path, n: int, k: int, mode: str, seed: int):
    rng = make_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for _ in range(n):
            a, b = sample_pair(rng, k=k, mode=mode)
            prompt, answer = format_example(a, b, k)
            row = {
                "a": a, "b": b, "k": k, "mode": mode,
                "prompt": prompt,
                "answer": answer
            }
            f.write(json.dumps(row) + "\n")

class AdditionDataset(Dataset):
    def __init__(self, path: str, tok, max_len: int = 64):
        self.path = Path(path)
        self.tok = tok
        self.max_len = max_len
        self.rows = self.path.read_text().splitlines()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = json.loads(self.rows[idx])
        prompt = ex["prompt"]
        ans = ex["answer"]

        full = prompt + " " + ans
        input_ids = self.tok.encode(full, add_bos=True, add_eos=True)

        prompt_ids = self.tok.encode(prompt, add_bos=True, add_eos=False)
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        input_ids = input_ids[: self.max_len]
        labels = labels[: self.max_len]

        attn = [1] * len(input_ids)
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tok.pad_id] * pad_len
            labels += [-100] * pad_len
            attn += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--k_train", type=int, default=3)
    p.add_argument("--n_train", type=int, default=200_000)
    p.add_argument("--n_val", type=int, default=10_000)
    p.add_argument("--n_test", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)

    out = Path(args.out_dir)
    k = args.k_train

    # training/val
    write_dataset(out / f"k{k}_uniform_train.jsonl", args.n_train, k, "uniform", seed=args.seed + 0)
    write_dataset(out / f"k{k}_uniform_val.jsonl",   args.n_val,   k, "uniform", seed=args.seed + 1)

    # eval slices
    write_dataset(out / f"k{k}_uniform_test.jsonl",  args.n_test,  k,   "uniform",   seed=args.seed + 2)
    write_dataset(out / f"k{k+1}_uniform_test.jsonl",args.n_test,  k+1, "uniform",   seed=args.seed + 3)
    write_dataset(out / f"k{k}_maxcarry_test.jsonl", args.n_test,  k,   "max_carry", seed=args.seed + 4)

    print("Wrote datasets to", out)

if __name__ == "__main__":
    main()
