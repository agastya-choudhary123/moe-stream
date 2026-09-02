#!/usr/bin/env python3
"""
Stable benchmark harness.

Single runs of this engine are worthless for comparison: throughput depends on
how much of experts.bin the OS page cache happens to be holding, which swings
by 5x between runs. Two guards here --

  warmup pass, discarded, so every measured trial starts with a comparably
  warm cache and pool;

  interleaved trials (A B A B ...) rather than all-A-then-all-B, so drift in
  cache state or background load hits both configs equally.

Reports median and full spread. If the spread overlaps, the difference is not
real and should not be reported as one.
"""

import os
import statistics as st
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPT = "Explain why mixture-of-experts models are memory-bandwidth bound."
TOKENS = int(os.environ.get("BENCH_TOKENS", "24"))
TRIALS = int(os.environ.get("BENCH_TRIALS", "3"))


def run_once(model, tok, pool, n_tokens):
    from mlx_lm.models.cache import make_prompt_cache
    ids = tok.encode(PROMPT)
    cache = make_prompt_cache(model)
    logits = model(mx.array(ids)[None], cache=cache)
    y = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(y)

    times = []
    for _ in range(n_tokens):
        t = time.perf_counter()
        logits = model(y[None], cache=cache)
        y = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(y)
        times.append(time.perf_counter() - t)
    steady = times[2:]
    return len(steady) / sum(steady)


def main():
    import importlib
    E = importlib.import_module(os.environ.get("BENCH_ENGINE", "engine_v3"))

    label = os.environ.get("BENCH_LABEL", "config")
    model, tok, pool = E.load_engine(n_slots=E.N_SLOTS)

    run_once(model, tok, pool, 8)          # warmup, discarded
    E.PROF.clear()

    scores = []
    for i in range(TRIALS):
        scores.append(run_once(model, tok, pool, TOKENS))
        print(f"  trial {i+1}: {scores[-1]:.2f} tok/s", flush=True)

    s = pool.stats()
    tot = s["total"] or 1
    med = st.median(scores)
    print(f"\n{label}: median {med:.2f} tok/s  "
          f"(min {min(scores):.2f}, max {max(scores):.2f}, "
          f"spread {max(scores)-min(scores):.2f})")
    print(f"  slots {pool.n_slots}  depth {E.PREFETCH_DEPTHS}  "
          f"workers {E.N_WORKERS}  cache {'COLD' if pool.cold else 'warm'}")
    print(f"  prefetch {s['prefetch']/tot*100:.0f}%  "
          f"cache {s['cache']/tot*100:.0f}%  miss {s['miss']/tot*100:.0f}%")
    print(f"  resident {mx.get_active_memory()/2**30:.2f} GB")

    acct = sum(E.PROF.values())
    if acct:
        print("  breakdown (share of accounted time):")
        for k, v in sorted(E.PROF.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<14}{v/acct*100:5.1f}%")
    return med


if __name__ == "__main__":
    main()
