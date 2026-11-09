"""Train a Studio Orbis v2-style language model from scratch (PyTorch).

Produces a model2/ folder (config.json + orbis_weights.npz + tokenizer.json) that the
NumPy server in app.py / orbis_np.py can serve directly. Designed for a free Colab/Kaggle
GPU session; resumes from checkpoints; two stages: pretrain then optional SFT.

Examples (Colab, Runtime > Change runtime type > T4 GPU):
    # Stage 1: pretrain on ~150M tokens of FineWeb-Edu
    python train.py --size medium --tokens 150000000 --out runs/orbis_m

    # Stage 2: teach it to chat (continue from the pretrained weights)
    python train.py --size medium --sft chat_data.jsonl --tokens 20000000 \
        --init runs/orbis_m/ckpt.pt --out runs/orbis_m_sft --lr 1e-4
"""
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- model (must match orbis_np.py exactly) -------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def apply_rope(x, cos, sin):          # x: (B, H, T, hd), interleaved-pair convention
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = o1, o2
    return out

class Block(nn.Module):
    def __init__(self, d, H, ffn):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(d, 3 * d, bias=False)
        self.attn.proj = nn.Linear(d, d, bias=False)
        self.n2 = RMSNorm(d)
        self.mlp = nn.Module()
        self.mlp.w1 = nn.Linear(d, ffn, bias=False)
        self.mlp.w3 = nn.Linear(d, ffn, bias=False)
        self.mlp.w2 = nn.Linear(ffn, d, bias=False)
        self.H = H
    def forward(self, x, cos, sin):
        B, T, d = x.shape
        hd = d // self.H
        h = self.n1(x)
        q, k, v = self.attn.qkv(h).chunk(3, dim=-1)
        q = apply_rope(q.view(B, T, self.H, hd).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(B, T, self.H, hd).transpose(1, 2), cos, sin)
        v = v.view(B, T, self.H, hd).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + att.transpose(1, 2).reshape(B, T, d) @ self.attn.proj.weight.T
        h = self.n2(x)
        x = x + (F.silu(h @ self.mlp.w1.weight.T) * (h @ self.mlp.w3.weight.T)) @ self.mlp.w2.weight.T
        return x

class OrbisLM(nn.Module):
    def __init__(self, vocab, d_model, n_layer, n_head, ffn_hidden, block_size, rope_base=10000.0):
        super().__init__()
        self.cfg = dict(vocab_size=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head,
                        ffn_hidden=ffn_hidden, block_size=block_size, rope_base=rope_base)
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, ffn_hidden) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        hd = d_model // n_head
        inv = 1.0 / (rope_base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        fr = torch.outer(torch.arange(block_size, dtype=torch.float32), inv)
        self.register_buffer("cos", fr.cos(), persistent=False)
        self.register_buffer("sin", fr.sin(), persistent=False)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos, sin = self.cos[:T][None, None], self.sin[:T][None, None]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.norm_f(x) @ self.tok_emb.weight.T       # tied output head
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.reshape(-1), ignore_index=-100)

SIZES = {
    "small":  dict(d_model=256, n_layer=4, n_head=4, ffn_hidden=512,  block_size=192),
    "medium": dict(d_model=384, n_layer=6, n_head=6, ffn_hidden=1024, block_size=256),
    "large":  dict(d_model=512, n_layer=8, n_head=8, ffn_hidden=1360, block_size=320),
}

# --------------------------------- data ---------------------------------

def iter_pretrain_tokens(tok, limit):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    n = 0
    for ex in ds:
        ids = tok.encode(ex["text"]).ids
        yield from ids
        n += len(ids)
        if n >= limit:
            return

def chunked(stream, block, batch_tokens, eos_id):
    buf, rows = [], []
    need = batch_tokens // block + 1
    for t in stream:
        buf.append(t)
        if len(buf) == block + 1:
            rows.append(buf); buf = []
            if len(rows) == need:
                yield np.asarray(rows, dtype=np.int64); rows = []

def sft_rows(path, tok, block, eos_id):
    """jsonl of {"user": ..., "assistant": ...} -> loss only on the assistant part."""
    for line in open(path, encoding="utf-8"):
        ex = json.loads(line)
        prefix = tok.encode(f"User: {ex['user']}\nAssistant:").ids
        full = prefix + tok.encode(ex["assistant"]).ids + [eos_id]
        full = full[-block-1:]
        if len(full) < 4:
            continue
        x = np.asarray(full[:-1], dtype=np.int64)
        y = np.asarray(full[1:], dtype=np.int64)
        y[:len(prefix) - 1] = -100                              # mask the prompt
        yield x, y

# --------------------------------- train ---------------------------------

def export_npz(model, tok_path, out_dir, extra=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(model.cfg); cfg["eos_id"] = 1
    cfg.update(extra or {})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    sd = model.state_dict()
    np.savez(out / "orbis_weights.npz",
             **{k: v.detach().float().cpu().numpy() for k, v in sd.items()})
    import shutil
    shutil.copy(tok_path, out / "tokenizer.json")
    print(f"[export] wrote {out} ({sum(v.numel() for v in sd.values()):,} params)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="medium", choices=list(SIZES))
    ap.add_argument("--tokens", type=int, default=150_000_000)
    ap.add_argument("--batch-tokens", type=int, default=16_384)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=2_000)
    ap.add_argument("--val-every", type=int, default=2_000)
    ap.add_argument("--save-every", type=int, default=10_000)
    ap.add_argument("--out", default="runs/orbis")
    ap.add_argument("--init", default="", help="ckpt.pt to continue from (pretrain or SFT)")
    ap.add_argument("--sft", default="", help="jsonl with user/assistant turns; switches to SFT mode")
    ap.add_argument("--tokenizer", default="model2/tokenizer.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = 1

    model = OrbisLM(vocab=tok.get_vocab_size(), **SIZES[args.size]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params on {device}")

    if args.init:
        ck = torch.load(args.init, map_location=device)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"[init] loaded {args.init}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=(device == "cuda"))
    if args.init and args.sft:
        opt.load_state_dict(ck["opt"]) if "opt" in ck else None

    steps_per_epoch = max(1, args.tokens // args.batch_tokens)
    total_steps = steps_per_epoch if not args.sft else steps_per_epoch
    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0, step, losses = time.time(), 0, []
    model.train()

    if args.sft:
        rows = list(sft_rows(args.sft, tok, model.cfg["block_size"], eos_id))
        print(f"[sft] {len(rows):,} conversations")
        order = np.random.permutation(len(rows))
        while step < total_steps:
            for i in range(0, len(order), 32):
                batch = [rows[j] for j in order[i:i+32]]
                x = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
                y = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
                lr = lr_at(step)
                for g in opt.param_groups: g["lr"] = lr
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item()); step += 1
                if step % 100 == 0:
                    print(f"step {step}/{total_steps} loss {np.mean(losses[-100:]):.4f} "
                          f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)
                if step % args.save_every == 0 or step == total_steps:
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "step": step, "val_loss": float(np.mean(losses))},
                               out / "ckpt.pt")
                    export_npz(model, args.tokenizer, out, {"trained_steps": step,
                               "val_loss": float(np.mean(losses)), "sft": True})
    else:
        stream = iter_pretrain_tokens(tok, args.tokens + 2_000_000)
        for batch in chunked(stream, model.cfg["block_size"], args.batch_tokens, eos_id):
            x = torch.from_numpy(batch[:, :-1]).to(device)
            y = torch.from_numpy(batch[:, 1:]).to(device)
            lr = lr_at(step)
            for g in opt.param_groups: g["lr"] = lr
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item()); step += 1
            if step % 100 == 0:
                print(f"step {step}/{total_steps} loss {np.mean(losses[-100:]):.4f} "
                      f"lr {lr:.2e} ({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)
            if step % args.save_every == 0 or step == total_steps:
                vl = float(np.mean(losses))
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "val_loss": vl}, out / "ckpt.pt")
                export_npz(model, args.tokenizer, out, {"trained_steps": step, "val_loss": vl})
            if step >= total_steps:
                break

    print(f"[done] {step} steps in {(time.time()-t0)/60:.1f} min; final loss {np.mean(losses):.4f}")

if __name__ == "__main__":
    main()
