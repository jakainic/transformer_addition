# -----------------
# Evaluate results
# -----------------
import argparse, json
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel

from src.tokenizer import CharacterTokenizer
from src.utils import seed_everything

def load_jsonl_examples(path: str):
    rows = Path(path).read_text().splitlines()
    exs = []
    for line in rows:
        r = json.loads(line)
        # expects fields written by data.py: a,b,k,answer
        exs.append((r["a"], r["b"], r["k"], r["answer"]))
    return exs

@torch.no_grad()
def predict_sum_batched(model, tok, pairs, k, max_new_tokens, device="cuda"):
    model.eval()

    # Build prompts and encode
    prompts = [f"{a:0{k}d} + {b:0{k}d} =" for (a, b) in pairs]
    prompt_ids_list = [tok.encode(p, add_bos=True, add_eos=False) for p in prompts]

    # Pad prompts to same length for batching
    B = len(prompt_ids_list)
    max_prompt_len = max(len(x) for x in prompt_ids_list)
  
    input_ids = torch.full((B, max_prompt_len), tok.pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, max_prompt_len), dtype=torch.long, device=device)

    prompt_lens = []
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
      gen_ids = out[i][prompt_lens[i]:]  # continuation only
      preds.append(tok.decode(gen_ids).strip())

    return preds

def digit_accuracy(pred: str, gold: str) -> float:
    if (not pred) or (not pred.isdigit()):
        return 0.0

    # Right-align by place value
    L = len(gold)
    pred_aligned = pred.zfill(L)[-L:]   # keep last L digits
    gold_aligned = gold.zfill(L)

    correct = sum(p == g for p, g in zip(pred_aligned, gold_aligned))
    return correct / L

def evaluate_examples(model, tok, examples, device="cuda", batch_size=256):
    # examples: list[(a,b,k,gold)]
    em = 0
    da = 0.0
    n = len(examples)

    k = examples[0][2]
    max_new_tokens = k + 4

    i = 0
    while i < n:
        chunk = examples[i:i+batch_size]
        pairs = [(a,b) for (a,b,_,_) in chunk]
        golds = [gold for (_,_,_,gold) in chunk]
        preds = predict_sum_batched(model, tok, pairs, k=k, max_new_tokens=max_new_tokens, device=device)
        for pred, gold in zip(preds, golds):
            em += int(pred == gold)
            da += digit_accuracy(pred, gold)
        i += batch_size

    return {"exact_match": em / n, "digit_acc": da / n, "n": n, "k": k} 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--iid_path", type=str, required=True)
    p.add_argument("--len_path", type=str, required=True)
    p.add_argument("--carry_path", type=str, required=True)
    p.add_argument("--out", type=str, default="results/results.json")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = CharacterTokenizer()
    model = GPT2LMHeadModel.from_pretrained(args.ckpt).to(device)

    iid = load_jsonl_examples(args.iid_path)
    lng = load_jsonl_examples(args.len_path)
    car = load_jsonl_examples(args.carry_path)

    results = {
        "iid_uniform_k": evaluate_examples(model, tok, iid, device=device, batch_size=args.batch_size),
        "length_uniform_kplus1": evaluate_examples(model, tok, lng, device=device, batch_size=args.batch_size),
        "shift_maxcarry_k": evaluate_examples(model, tok, car, device=device, batch_size=args.batch_size),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", args.out)
    for name, r in results.items():
        print(name, r)

if __name__ == "__main__":
    main()
