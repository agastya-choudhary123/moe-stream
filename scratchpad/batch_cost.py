#!/usr/bin/env python3
"""T(t): what does a t-token forward actually cost in THIS engine?

This is the denominator of every speculative-decoding estimate. The model says
a forward costs a batch-INDEPENDENT part (48 CPU-GPU sync round trips, plus
attention/lm_head, which are weight-bandwidth bound and so flat at these sizes)
plus an expert part that scales with union(t) rather than with t. If that holds,
verifying t tokens at once is much cheaper than t forwards, and speculative
decoding pays. If it does not hold, nothing downstream matters.

Discipline (HANDOFF rules):
  - measured on the real engine, not a substitute
  - REPLAY a fixed token sequence, so every arm routes to the same experts in
    the same order and only the batching differs. Any ablation that lets the
    hidden state drift changes routing, which changes I/O volume and pool state.
  - interleaved, order alternated, several rounds
  - report ms/token, which is near-deterministic here, alongside wall
"""
import os
import sys
import time

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")
os.environ.setdefault("MOE_COLD", "1")

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

MODEL = os.environ.get("MODEL", "30b")
E = __import__("engine_120b" if MODEL == "120b" else "engine_v3")

PROMPT = os.environ.get("PROMPT",
                        "Explain why mixture-of-experts models are memory-bandwidth bound.")
NGEN = int(os.environ.get("NGEN", "64"))     # length of the replay sequence
TS = [int(x) for x in os.environ.get("TS", "1,2,4,8").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "3"))


def main():
    model, tok, pool = E.load_engine()
    ids = tok.encode(PROMPT)

    # 1. Generate a real continuation autoregressively. This is the sequence
    #    every arm will replay, so routing is identical across arms.
    c = make_prompt_cache(model)
    y = mx.argmax(model(mx.array(ids)[None], cache=c)[:, -1], axis=-1)
    mx.eval(y)
    seq = []
    for _ in range(NGEN):
        seq.append(y.item())
        y = mx.argmax(model(y[None], cache=c)[:, -1], axis=-1)
        mx.eval(y)
    print(f"replay sequence: {len(seq)} tokens", flush=True)

    results = {t: [] for t in TS}
    bytes_of = {t: [] for t in TS}
    for r in range(ROUNDS):
        order = TS if r % 2 == 0 else list(reversed(TS))
        for t in order:
            # fresh cache each arm: identical starting state, identical work
            c = make_prompt_cache(model)
            y = mx.argmax(model(mx.array(ids)[None], cache=c)[:, -1], axis=-1)
            mx.eval(y)
            n = (len(seq) // t) * t
            b0 = pool.bytes_read
            t0 = time.perf_counter()
            for s in range(0, n, t):
                chunk = mx.array(seq[s:s + t])[None]
                out = model(chunk, cache=c)
                mx.eval(out[:, -1])
            dt = time.perf_counter() - t0
            db = pool.bytes_read - b0
            results[t].append(dt / n * 1e3)          # ms per POSITION
            bytes_of[t].append(db / n / 2 ** 20)     # MB per POSITION
            print(f"  r{r} t={t:<3} {dt/n*1e3:7.2f} ms/pos  "
                  f"{dt/(n//t)*1e3:7.2f} ms/forward  "
                  f"{db/n/2**20:7.1f} MB/pos", flush=True)

    print(f"\n=== {MODEL}: cost of a t-token forward, {ROUNDS} rounds, "
          f"order alternated, replayed sequence ===")
    print(f"{'t':>3} {'ms/forward':>11} {'ms/position':>12} {'speedup':>9} "
          f"{'MB/position':>12} {'byte ratio':>11}")
    base = sorted(results[TS[0]])[len(results[TS[0]]) // 2]
    baseb = sorted(bytes_of[TS[0]])[len(bytes_of[TS[0]]) // 2]
    for t in TS:
        v = sorted(results[t])
        bv = sorted(bytes_of[t])
        m = v[len(v) // 2]
        bm = bv[len(bv) // 2]
        print(f"{t:>3} {m*t:>11.1f} {m:>12.2f} {base/m:>8.2f}x "
              f"{bm:>12.1f} {bm/baseb:>10.3f}x    "
              f"range {v[0]:.2f}-{v[-1]:.2f}")


if __name__ == "__main__":
    main()
