# -----------------
# Evaluate results
# -----------------
import argparse, json
import torch
from transformers import GPT2LMHeadModel

from src.tokenizer import CharacterTokenizer
from src.utils import seed_everything, make_rng

@torch.no_grad()
def predict_sum_batched(model, pairs, k, max_new_tokens, device="cuda"):
    model.eval()

    # Build prompts and encode
    prompts = [f"{a:0{k}d} + {b:0{k}d} =" for (a, b) in pairs]
    prompt_ids_list = [tok.encode(p, add_bos=True, add_eos=False) for p in prompts]

    # Pad prompts to same length for batching
    B = len(prompt_ids_list)
    max_prompt_len = max(len(x) for x in prompt_ids_list)
  
    input_ids = torch.full((B, max_prompt_len), tok.pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, max_prompt_len), dtype=torch.long, device=device)

    prompt_lens = [
    for i, ids in enumerate(prompt_ids_list):
      L = len(ids)
      prompt_lens.append(L)
      input_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
      attn[i, :L] = 1

    out = model.generate(
      input_ids=input_ids,
      max_new_tokens=max_new_tokens,
      do_sample=False,
      eos_token_id=tok.eos_id,
      pad_token_id=tok.pad_id,
      attention_mask=attn,
    )

    # only decode the generated continuation (not the prompt)
    preds = []
    out = out.tolist()
    for i in range(B):
      prompt_len = len(prompt_ids_list[i])
      gen_ids = out[i][prompt_len:]  # continuation only
      preds.append(tok.decode(gen_ids).strip())

    return preds

def digit_accuracy(pred: str, gold: str) -> float:
    """
    Digit-level accuracy aligned by least significant digit (right alignment).
    Returns fraction of digits correct in gold length.
    If pred contains non-digits or is empty, returns 0.0.
    """
    if (not pred) or (not pred.isdigit()):
        return 0.0

    # Right-align by place value
    L = len(gold)
    pred_aligned = pred.zfill(L)[-L:]   # keep last L digits
    gold_aligned = gold.zfill(L)

    correct = sum(p == g for p, g in zip(pred_aligned, gold_aligned))
    return correct / L

def evaluate_slice(model, tok, sample_pair_fn, k, n, mode, seed, device, batch_size=256):
    rng = make_rng(seed)

    if mode == "uniform":
        def sampler():
            lo, hi = 10**(k-1), 10**k - 1
            return rng.randint(lo, hi), rng.randint(lo, hi)
    else:
        # if you want: import from src.data sample_pair (carry mode)
        from src.data import sample_pair as carry_sample_pair
        def sampler():
            return carry_sample_pair(rng, k=k, mode=mode)

    max_new_tokens = k + 4
    em, da = 0, 0.0

    done = 0
    while done < n:
        bs = min(batch_size, n - done)
        pairs = [sampler() for _ in range(bs)]
        golds = [str(a + b) for a, b in pairs]
        preds = predict_sum_batched(model, tok, pairs, k=k, max_new_tokens=max_new_tokens, device=device)

        for pred, gold in zip(preds, golds):
            em += int(pred == gold)
            da += digit_accuracy(pred, gold)
        done += bs

    return {
        "k": k,
        "mode": mode,
        "n": n,
        "exact_match": em / n,
        "digit_acc": da / n,
        "seed": seed,
    }  

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to trained model dir or checkpoint dir")
    p.add_argument("--k_train", type=int, default=3)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default="results/results.json")
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = CharacterTokenizer()
    model = GPT2LMHeadModel.from_pretrained(args.ckpt).to(device)

    results = {}
    results["iid_uniform_k"] = evaluate_slice(model, tok, None,
        k=args.k_train, n=args.n, mode="uniform", seed=args.seed + 1, device=device, batch_size=args.batch_size
    )
    results["length_uniform_kplus1"] = evaluate_slice(model, tok, None,
        k=args.k_train + 1, n=args.n, mode="uniform", seed=args.seed + 2, device=device, batch_size=args.batch_size
    )
    results["shift_maxcarry_k"] = evaluate_slice(model, tok, None,
        k=args.k_train, n=args.n, mode="max_carry", seed=args.seed + 3, device=device, batch_size=args.batch_size
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", args.out)
    for k, r in results.items():
        print(k, r)

if __name__ == "__main__":
    from pathlib import Path
    main()
