# Training and evaluating a small Transformer on multi-digit integer addition

This project studies what a small Transformer learns when trained to add two base-10 integers, and how that knowledge generalizes across input length and carry structure.

## 1. Task

We train a causal Transformer to map formulae of the form: "123 + 456 = " -> "579"

Key properties:
* Base-10 arithmetic
* Fixed-width inputs during training
* Autoregressive generation at test time
* No explicit supervision of carries or intermediate steps

The goal is not just accuracy, but to understand what the model learns and where it fails.

## 2. Model

* Architecture: GPT-2–style causal Transformer
* Layers: 6
* Hidden size: 256
* Attention heads: 8
* Context length: 64 tokens

## 3. Tokenization

Character-level tokenizer with vocabulary \<pad\>, \<bos\>, \<eos\>, 0-9, " ", +, =. 

## 4. Data

Training Distribution
* Support: k-digit integers (default k=3)
* Distribution: uniform over \[10^(k-1), 10^k-1\]

Test Distributions
* In distribution (uniform, k-digit)
* Length generalization (uniform, k+1-digit)
* Carry shift generalization (k-digit addition with carry at every column)

## 5. Training

* Optimizer: AdamW
* Learning rate: 3e-4
* Warmup: 200 steps
* Batch size: 256
* Loss: next-token cross-entropy (masked to answer tokens only)

## 6. Evaluation

Metrics:
* Exact match accuracy (entire sum correct)
* Digit accuracy (right-aligned per-digit correctness)

## 7. How to run
For convenience, this repository can be run end-to-end in [this colab notebook](https://colab.research.google.com/drive/1EmcUi9kOKgpqqpi5nEpk37CskjFoMcjv?usp=sharing)

