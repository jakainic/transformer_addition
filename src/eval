# -----------------
# Evaluate results
# -----------------

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

    for i, ids in enumerate(prompt_ids_list):
      L = len(ids)
      input_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
      attn[i, :L] = 1

    out = model.generate(
      input_ids=input_ids,
      max_new_tokens=max_new_tokens,
      do_sample=False,  # greedy
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

def evaluate_suite_batched(model, 
                           k, 
                           n, 
                           mode, 
                           device="cuda", 
                           max_new_tokens=None,
                           batch_size=256,
                           seed=0):
    """
    Batched evaluation: exact match + digit accuracy.
    """
    if max_new_tokens is None:
        max_new_tokens = (k + 4)
    
    rng = random.Random(seed)

    em_correct = 0
    digit_acc_sum = 0.0

    num_done = 0
    while num_done < n:
        bs = min(batch_size, n - num_done)

        state = random.getstate()
        random.setstate(rng.getstate())
        pairs = [sample_pair(k, mode) for _ in range(bs)]
        golds = [str(a + b) for a, b in pairs]

        rng.setstate(random.getstate())
        random.setstate(state)
        preds = predict_sum_batched(model, pairs, k, max_new_tokens, device)

        for pred, gold in zip(preds, golds):
            if pred == gold:
                em_correct += 1
            digit_acc_sum += digit_accuracy(pred, gold)
        num_done += bs

    return {
        "exact_match": em_correct / n,
        "digit_acc": digit_acc_sum / n,
        "n": n,
        "k": k,
        "mode": mode,
        "batch_size": batch_size
    }

def run_eval(model, k_train, n=5000, device="cuda", batch_size=256, seed=0):
    model.to(device)
    results = {}

    # 1) IID
    results["iid_uniform_k"] = evaluate_suite_batched(
        model, k=k_train, n=n, mode="uniform", device=device,
        max_new_tokens=k_train + 4, batch_size=batch_size, seed=seed + 1,
    )

    # 2) length generalization
    k_long = k_train + 1
    results["length_uniform_kplus1"] = evaluate_suite_batched(
        model, k=k_long, n=n, mode="uniform", device=device,
        max_new_tokens=k_long + 4, batch_size=batch_size, seed=seed + 2
    )

    # 3) carry shift (hard)
    results["shift_maxcarry_k"] = evaluate_suite_batched(
        model, k=k_train, n=n, mode="max_carry", device=device,
        max_new_tokens=k_train + 4, batch_size=batch_size, seed=seed + 3
    )

    return results    
  
