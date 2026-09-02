#!/usr/bin/env python3
"""Interleaved A/B: speculative vs baseline greedy, in one process.

The naive comparison is wrong in this engine and the first run of it said 0.54x
for a change that is positive. Whichever arm runs second inherits a warm slot
pool from the first, and at 3072 slots the difference between a cold and a
steady-state pool is 466 vs ~34 MB/token -- an order of magnitude larger than
the effect being measured.

So: load once, warm the pool to steady state, then alternate arm order every
round. Report MB/token and tok/step alongside tok/s, because HANDOFF measured
tok/s on this machine at a ~12% noise floor while byte counts are
near-deterministic.
"""
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")
os.environ.setdefault("MOE_COLD", "1")

import mlx.core as mx
from spec import (NGramDrafter, NoDrafter, spec_generate, baseline_generate,
                  PROMPTS)

MODEL = os.environ.get("MODEL", "30b")
GAMMA = int(os.environ.get("GAMMA", "4"))
NTOK = int(os.environ.get("TOKENS", "64"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
WARM = int(os.environ.get("WARM", "96"))
PROMPT = os.environ.get("PROMPT", "code")

E = __import__("engine_120b" if MODEL == "120b" else "engine_v3")


def main():
    model, tok, pool = E.load_engine()
    ids = tok.encode(PROMPTS.get(PROMPT, PROMPT))
    drafter = NGramDrafter(gamma=GAMMA)

    # warm the pool to steady state; HANDOFF measured decode drifting +28..96%
    # from the first third of a generation to the last as the pool fills
    print(f"warming {WARM} tokens...", flush=True)
    baseline_generate(model, ids, WARM)
    print("warm.", flush=True)

    res = defaultdict(list)
    for r in range(ROUNDS):
        arms = ["spec", "base"] if r % 2 == 0 else ["base", "spec"]
        for arm in arms:
            b0 = pool.bytes_read
            st = defaultdict(float)
            if arm == "spec":
                out, st = spec_generate(model, pool, ids, NTOK, drafter, GAMMA, st)
            else:
                out, st = baseline_generate(model, ids, NTOK, st)
            mb = (pool.bytes_read - b0) / 2 ** 20 / st["emitted"]
            tps = st["emitted"] / st["t_decode"]
            res[arm + "_tps"].append(tps)
            res[arm + "_mb"].append(mb)
            res[arm + "_tps_step"].append(st["emitted"] / max(1, st["steps"]))
            if arm == "spec":
                res["accept"].append(st["accepted"] / max(1, st["drafted"]))
            res[arm + "_out"] = out
            print(f"  r{r} {arm:<5} {tps:6.2f} tok/s  {mb:6.1f} MB/token  "
                  f"{st['emitted']/max(1,st['steps']):.2f} tok/step", flush=True)

    def med(k):
        v = sorted(res[k])
        return v[len(v) // 2], v[0], v[-1]

    st_, sl, sh = med("spec_tps")
    bt, bl, bh = med("base_tps")
    sm = med("spec_mb")[0]
    bm = med("base_mb")[0]
    ident = list(res["spec_out"]) == list(res["base_out"])
    print(f"\n=== {MODEL} / {PROMPT} / gamma={GAMMA}, {ROUNDS} rounds, "
          f"order alternated, warm pool ===")
    print(f"  baseline     {bt:6.2f} tok/s ({bl:.2f}-{bh:.2f})   {bm:6.1f} MB/token")
    print(f"  speculative  {st_:6.2f} tok/s ({sl:.2f}-{sh:.2f})   {sm:6.1f} MB/token")
    print(f"  ratio        {st_/bt:6.2f}x throughput   {sm/bm:.2f}x bytes")
    print(f"  tok/step     {med('spec_tps_step')[0]:.2f}   "
          f"draft acceptance {med('accept')[0]*100:.0f}%")
    print(f"  LOSSLESS     {'PASS' if ident else 'FAIL'} "
          f"({sum(a==b for a,b in zip(res['spec_out'],res['base_out']))}/{NTOK})")
    overlap = not (sl > bh or sh < bl)
    print(f"  ranges {'OVERLAP -- not a throughput result on this rig' if overlap else 'do not overlap'}")


if __name__ == "__main__":
    main()
