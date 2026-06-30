#!/usr/bin/env python3
"""
throughput_probe.py — Gate 0, Task 1: sustained training throughput on a single GPU.

Measures sustained (steady-state, NOT burst) training tokens/sec for several small
GPT-style decoder sizes, using synthetic random data so the number reflects the GPU,
not a dataloader. Runs long enough per size to surface thermal throttling on a laptop.

Outputs:
  - live per-window tok/s to stdout (so you can watch for throttle decay)
  - a JSON summary (throughput_results.json) with warmup vs. steady-state rates

Usage:
  python throughput_probe.py                  # defaults: 5M,15M,30M,60M @ seq512 bf16
  python throughput_probe.py --sizes 5 30     # just two sizes
  python throughput_probe.py --seconds 90     # longer per-size run (better throttle read)
  python throughput_probe.py --seq 256        # shorter context (child-text-appropriate)

Requires: torch (with CUDA). Nothing else.
"""

import argparse, json, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- model -----------------------------
# Minimal GPT-style decoder. Depth-tied dense family (Liu-style) so sizes are
# comparable. Uses PyTorch's scaled_dot_product_attention (Flash path when available).

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.fc1 = nn.Linear(dim, mlp_ratio * dim, bias=False)
        self.fc2 = nn.Linear(mlp_ratio * dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # causal flash/sdpa attention
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(a)
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, dim, depth, n_heads, seq):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.seq = seq

    def forward(self, idx, targets):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss


# Size presets: (dim, depth, heads) chosen to hit approx target non-embedding param counts.
# Param count is dominated by 12*dim^2*depth (attn+mlp). Embeddings excluded from the
# "model size" label since they scale with vocab and don't reflect compute the same way.
SIZE_PRESETS = {
    5:  dict(dim=256, depth=6,  heads=8),    # ~5M  non-embed
    15: dict(dim=384, depth=8,  heads=8),    # ~15M
    30: dict(dim=512, depth=10, heads=8),    # ~30M
    60: dict(dim=640, depth=14, heads=10),   # ~60M
}


def count_params(model, include_embed=False):
    total = sum(p.numel() for p in model.parameters())
    if include_embed:
        return total
    embed = model.tok.weight.numel() + model.pos.weight.numel()
    return total - embed


def _build_model(size_key, seq, vocab, device):
    cfg = SIZE_PRESETS[size_key]
    model = TinyGPT(vocab, cfg["dim"], cfg["depth"], cfg["heads"], seq).to(device)
    return model


def _try_step(size_key, seq, vocab, device, bs, dtype):
    model = _build_model(size_key, seq, vocab, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, vocab, (bs, seq), device=device)
    y = torch.randint(0, vocab, (bs, seq), device=device)
    with torch.autocast(device_type="cuda", dtype=dtype):
        loss = model(x, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    del model, opt, x, y, loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def find_batch(size_key, seq, vocab, device, dtype, start=64):
    bs = start
    while bs >= 1:
        try:
            _try_step(size_key, seq, vocab, device, bs, dtype)
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); bs //= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); bs //= 2
            else:
                raise
    return 1


def measure_size(size_key, seq, vocab, device, dtype, seconds, warmup_frac=0.3):
    """Run training steps for `seconds`, report warmup vs steady-state tok/s."""
    bs = find_batch(size_key, seq, vocab, device, dtype, start=64)
    model = _build_model(size_key, seq, vocab, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n_params = count_params(model)

    # Pre-allocate a few random batches and cycle them (we measure compute, not data IO).
    pool = [
        (torch.randint(0, vocab, (bs, seq), device=device),
         torch.randint(0, vocab, (bs, seq), device=device))
        for _ in range(8)
    ]

    tokens_per_step = bs * seq
    window_secs = 5.0
    windows = []           # (elapsed_at_window_end, tok/s_in_window)
    step = 0
    torch.cuda.synchronize()
    t0 = time.time()
    t_window = t0
    toks_window = 0

    while True:
        x, y = pool[step % len(pool)]
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        toks_window += tokens_per_step

        # only sync at window boundaries to avoid per-step sync overhead skewing the number
        now = time.time()
        if now - t_window >= window_secs:
            torch.cuda.synchronize()
            now = time.time()
            wtps = toks_window / (now - t_window)
            elapsed = now - t0
            windows.append((elapsed, wtps))
            print(f"  [{size_key:>3}M] t={elapsed:6.1f}s  window tok/s={wtps:11,.0f}  bs={bs}")
            t_window = now
            toks_window = 0
            if elapsed >= seconds:
                break

    # steady-state = mean of windows after warmup fraction
    cut = warmup_frac * windows[-1][0]
    steady = [w for (e, w) in windows if e >= cut]
    warmup = [w for (e, w) in windows if e < cut]
    steady_tps = sum(steady) / len(steady) if steady else windows[-1][1]
    warmup_tps = sum(warmup) / len(warmup) if warmup else steady_tps
    throttle_pct = 100 * (1 - steady_tps / warmup_tps) if warmup_tps > 0 else 0.0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    del model, opt, pool
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    return dict(
        size_label_M=size_key,
        non_embed_params=n_params,
        batch=bs, seq=seq,
        steady_tok_s=steady_tps,
        warmup_tok_s=warmup_tps,
        throttle_pct=throttle_pct,
        peak_mem_gb=peak_mem,
        n_windows=len(windows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 15, 30, 60],
                    help="param-size presets in millions (choose from 5 15 30 60)")
    ap.add_argument("--seq", type=int, default=512, help="sequence length")
    ap.add_argument("--vocab", type=int, default=32000, help="vocab size (BPE 32k default)")
    ap.add_argument("--seconds", type=int, default=60,
                    help="seconds per size (>=60 recommended to read thermal throttle)")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--compile", action="store_true", help="try torch.compile (may be slow to warm up)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This probe must run on the GPU machine.")
        sys.exit(1)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")
    print(f"Config: seq={args.seq} vocab={args.vocab} precision={args.precision} "
          f"seconds/size={args.seconds}")
    print(f"Sizes: {args.sizes}\n")
    print("NOTE: run plugged in, on a consistent power profile. Watch the window tok/s —")
    print("      if it decays over time, that's thermal throttling and 'steady' captures it.\n")

    results = []
    for s in args.sizes:
        if s not in SIZE_PRESETS:
            print(f"skip {s}M (no preset)"); continue
        print(f"=== {s}M params ===")
        try:
            r = measure_size(s, args.seq, args.vocab, device, dtype, args.seconds)
            results.append(r)
            print(f"  -> steady {r['steady_tok_s']:,.0f} tok/s  "
                  f"(warmup {r['warmup_tok_s']:,.0f}, throttle {r['throttle_pct']:.1f}%, "
                  f"peak {r['peak_mem_gb']:.2f} GB, bs {r['batch']})\n")
        except Exception as e:
            print(f"  FAILED at {s}M: {e}\n")

    out = dict(gpu=gpu, seq=args.seq, vocab=args.vocab, precision=args.precision,
               seconds_per_size=args.seconds, results=results)
    with open("throughput_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote throughput_results.json")

    # quick passes-per-day projection at a few word budgets (~1.3 tok/word assumed)
    if results:
        print("\n--- passes-per-24h projection (assumes ~1.3 tokens/word) ---")
        print("    (use the STEADY rate; recompute with your real tokens/word later)\n")
        for r in results:
            tps = r["steady_tok_s"]
            toks_day = tps * 86400
            print(f"  {r['size_label_M']:>3}M @ {tps:>11,.0f} tok/s -> {toks_day/1e9:5.1f}B tok/day")
            for words in (2_000_000, 5_000_000, 10_000_000):
                budget_tok = words * 1.3
                passes = toks_day / budget_tok
                print(f"        {words//1_000_000}M-word budget: {passes:6.0f} passes/day")
            print()


if __name__ == "__main__":
    main()