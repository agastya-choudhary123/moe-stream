#!/usr/bin/env python3
"""How does the union of routed experts grow with the number of tokens verified
together? This one number decides whether speculative decoding pays here.

Why it matters. Decode in this engine costs three things, and verifying t tokens
in ONE forward pass amortises all three:

  bytes off SSD   union(t) blobs per layer instead of t*top_k
  syncs           L round trips per FORWARD, not per token (HANDOFF: ~171 us of
                  every 210 us router invocation is the CPU-GPU round trip)
  read rate       burst size per layer goes 2.1 -> ~union(t)/layer, and
                  burst_io.py measured 2.73 GB/s at burst 1 vs 3.24 at burst 8

Under independent uniform routing MoESD gives union(t) = E*(1-(1-K/E)^t)
(arXiv 2505.19645). But routing here is NOT independent across tokens: HANDOFF
measured 8 consecutive tokens touching 17.9 distinct experts per layer on the
120b where independence predicts 28.7. That correlation is worth 1.6x and it has
to be measured, not assumed.

This records the real routed set per (token, layer) from a real decode and
reports union over sliding windows of t consecutive tokens. Near-deterministic:
it counts experts, not time.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")

MODEL = os.environ.get("MODEL", "120b")
NTOK = int(os.environ.get("TOKENS", "128"))
os.environ.setdefault("MOE_COLD", "1")

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

if MODEL == "120b":
    import engine_120b as E
else:
    import engine_v3 as E

TRACE = []          # list over tokens of {layer: frozenset(experts)}


def install_trace(pool):
    real = pool.acquire
    cur = {}

    def acquire(layer, experts):
        cur[layer] = frozenset(int(e) for e in experts)
        return real(layer, experts)
    pool.acquire = acquire
    return cur


PROMPTS = [
    "Explain why mixture-of-experts models are memory-bandwidth bound.",
    "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists.\"\"\"\n",
    "La conquete de l'espace au vingtieme siecle a commence",
    "Solve step by step: a train leaves at 3pm going 60 mph,",
]


def main():
    model, tok, pool = E.load_engine()
    cur = install_trace(pool)
    rows = []
    for pi, prompt in enumerate(PROMPTS):
        ids = tok.encode(prompt)
        c = make_prompt_cache(model)
        y = mx.argmax(model(mx.array(ids)[None], cache=c)[:, -1], axis=-1)
        mx.eval(y)
        cur.clear()
        for _ in range(NTOK):
            y = mx.argmax(model(y[None], cache=c)[:, -1], axis=-1)
            mx.eval(y)
            rows.append((pi, dict(cur)))
            cur.clear()
        print(f"  prompt {pi}: {len(rows)} token-rows so far", flush=True)

    n_layers = pool.n_layers
    top_k = pool.top_k
    E_n = pool.n_experts
    print(f"\n=== {MODEL}: L={n_layers} E={E_n} top_k={top_k}, "
          f"{len(rows)} decode tokens over {len(PROMPTS)} prompts ===")
    print(f"{'t':>3} {'union/layer':>12} {'per token':>10} {'vs t*k':>8} "
          f"{'independent':>12} {'corr gain':>10}")
    out = {}
    for t in (1, 2, 3, 4, 6, 8, 12, 16):
        tot, cnt = 0.0, 0
        for pi in range(len(PROMPTS)):
            idx = [i for i, (p, _) in enumerate(rows) if p == pi]
            for s in range(0, len(idx) - t + 1):
                win = [rows[i][1] for i in idx[s:s + t]]
                for L in range(n_layers):
                    u = set()
                    for w in win:
                        u |= w.get(L, frozenset())
                    tot += len(u)
                    cnt += 1
        u_t = tot / cnt
        indep = E_n * (1 - ((E_n - top_k) / E_n) ** t)
        out[t] = u_t
        print(f"{t:>3} {u_t:>12.2f} {u_t/t:>10.2f} {u_t/(t*top_k):>8.3f} "
              f"{indep:>12.2f} {indep/u_t:>10.3f}x")

    print("\n--- what that buys, per accepted token, at 100% acceptance ---")
    base = out[1]
    for t in (2, 4, 8, 16):
        print(f"  t={t:<3} bytes/token {out[t]/t/base:.3f}x   "
              f"reads per layer per forward {out[t]:.1f} (was {base:.1f})")
    json.dump({str(k): v for k, v in out.items()},
              open(f"union_{MODEL}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
