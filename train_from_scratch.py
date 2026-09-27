#!/usr/bin/env python3
"""
NEZ Young — Train a tiny decoder-only Transformer from scratch
(random initialization, no pretrained weights) on the tool-use corpus.

This is deliberately small so it can run on GitHub Actions CPU runners.
Architecture inspired by nanoGPT / GPT-2 but with random init only.
"""

import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------------------- Config (tiny for CPU) ---------------------------
class Config:
    # Data
    data_path = "data/nez_young_tooluse_corpus.json"
    out_dir = "checkpoints"
    # Model (very small)
    block_size = 256          # context length
    n_layer = 4
    n_head = 4
    n_embd = 128
    dropout = 0.1
    bias = False
    # Training
    batch_size = 16
    max_iters = 2000          # keep short for GH Actions (~few minutes on CPU)
    eval_interval = 200
    eval_iters = 50
    learning_rate = 3e-4
    weight_decay = 0.1
    grad_clip = 1.0
    warmup_iters = 100
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

cfg = Config()

# --------------------------- Data preparation ---------------------------
def format_example(ex: dict) -> str:
    """Turn one JSON example into a single training string."""
    tools = json.dumps(ex.get("tool_calls", []), ensure_ascii=False)
    return (
        f"### User\n{ex['user_prompt']}\n\n"
        f"### Thought\n{ex['thought_process']}\n\n"
        f"### Tools\n{tools}\n\n"
        f"### ToolOutputs\n{ex.get('tool_outputs', '')}\n\n"
        f"### Response\n{ex['final_response']}\n\n"
        f"<|endoftext|>\n"
    )

def load_and_tokenize(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_texts, val_texts = [], []
    for ex in data:
        text = format_example(ex)
        if ex.get("data_split") == "train":
            train_texts.append(text)
        else:
            # use validation + test as val for simplicity
            val_texts.append(text)
    return "\n".join(train_texts), "\n".join(val_texts)

print("Loading and formatting corpus...")
train_text, val_text = load_and_tokenize(cfg.data_path)
print(f"Train chars: {len(train_text):,}  |  Val chars: {len(val_text):,}")

# Character-level vocabulary (pure from-scratch, no external tokenizer)
chars = sorted(list(set(train_text + val_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print(f"Vocab size: {vocab_size}")

# Save tokenizer mapping for later inference
os.makedirs(cfg.out_dir, exist_ok=True)
with open(os.path.join(cfg.out_dir, "tokenizer.json"), "w") as f:
    json.dump({"chars": chars, "stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, f)

train_ids = torch.tensor(encode(train_text), dtype=torch.long)
val_ids = torch.tensor(encode(val_text), dtype=torch.long)

def get_batch(split: str):
    data = train_ids if split == "train" else val_ids
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + cfg.block_size] for i in ix])
    x, y = x.to(cfg.device), y.to(cfg.device)
    return x, y

# --------------------------- Model (from-scratch decoder-only) ---------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class NEZYoung(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight
        # random init (explicit)
        self.apply(self._init_weights)
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.3f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --------------------------- Training ---------------------------
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)

model = NEZYoung(cfg).to(cfg.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print(f"Training on {cfg.device} ...")
t0 = time.time()
best_val = float("inf")

for iter_num in range(cfg.max_iters + 1):
    # learning rate schedule (linear warmup then constant)
    if iter_num < cfg.warmup_iters:
        lr = cfg.learning_rate * (iter_num + 1) / cfg.warmup_iters
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(f"iter {iter_num:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s")
        if losses["val"] < best_val:
            best_val = losses["val"]
            ckpt = {
                "model": model.state_dict(),
                "config": {
                    "block_size": cfg.block_size,
                    "n_layer": cfg.n_layer,
                    "n_head": cfg.n_head,
                    "n_embd": cfg.n_embd,
                    "vocab_size": vocab_size,
                    "dropout": cfg.dropout,
                    "bias": cfg.bias,
                },
                "iter": iter_num,
                "best_val": best_val,
            }
            torch.save(ckpt, os.path.join(cfg.out_dir, "nez_young_best.pt"))
            print(f"  -> saved best checkpoint (val={best_val:.4f})")

    X, Y = get_batch("train")
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

# Final sample generation
print("\n--- Sample generation after training ---")
model.eval()
prompt = "### User\nWhat's the current price of Bitcoin?\n\n### Thought\n"
idx = torch.tensor([encode(prompt)], dtype=torch.long, device=cfg.device)
out = model.generate(idx, max_new_tokens=200, temperature=0.7, top_k=30)
print(decode(out[0].tolist()))

print("\nTraining complete. Best checkpoint saved to checkpoints/nez_young_best.pt")
print(f"Total time: {time.time() - t0:.1f}s")
