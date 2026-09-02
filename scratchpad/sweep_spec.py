#!/usr/bin/env python3
"""Sweep the drafter against the baseline, all arms in one warm process.

Also answers a question the first A/B raised: the speculative and baseline token
ids differed at 3 of 64 positions on one prompt and 0 of 64 on another. Either
verification is not lossless, or the ENGINE is not run-to-run deterministic once
the batch shape changes. So the baseline is run twice and compared to itself --
if base != base, the 3 flips are the engine's bf16 tie-breaking (HANDOFF already
documents exactly this for the fused kernel: a top1-top2 logit gap of 0.125,
the smallest representable in bf16, against a 0.44 logit perturbation), not a
bug in the accept rule.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")
os.environ.setdefault("MOE_COLD", "1")

from spec import NGramDrafter, spec_generate, baseline_generate, PROMPTS

MODEL = os.environ.get("MODEL", "30b")
NTOK = int(os.environ.get("TOKENS", "64"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
WARM = int(os.environ.get("WARM", "128"))
PROMPT = os.environ.get("PROMPT", "code")
CONFIGS = [tuple(int(x) for x in c.split(":"))
           for c in os.environ.get("CONFIGS", "1:3,2:3,4:3,2:2,4:2").split(",")]

E = __import__("engine_120b" if MODEL == "120b" else "engine_v3")


def main():
    model, tok, pool = E.load_engine()
    ids = tok.encode(PROMPTS.get(PROMPT, PROMPT))
    baseline_generate(model, ids, WARM)
    print("warm.", flush=True)

    arms = [("base", None)] + [(f"g{g}n{n}", (g, n)) for g, n in CONFIGS]
    res = defaultdict(list)
    outs = {}
    for r in range(ROUNDS):
        order = arms if r % 2 == 0 else list(reversed(arms))
        for name, cfg in order:
            b0 = pool.bytes_read
            st = defaultdict(float)
            if cfg is None:
                out, st = baseline_generate(model, ids, NTOK, st)
                # second identical baseline: tests engine determinism itself
                if r == 0:
                    o2, _ = baseline_generate(model, ids, NTOK, defaultdict(float))
                    outs["base2"] = o2
            else:
                g, n = cfg
                d = NGramDrafter(max_n=4, min_n=n, gamma=g)
                out, st = spec_generate(model, pool, ids, NTOK, d, g, st)
            mb = (pool.bytes_read - b0) / 2 ** 20 / st["emitted"]
            res[name + "_tps"].append(st["emitted"] / st["t_decode"])
            res[name + "_mb"].append(mb)
            res[name + "_step"].append(st["emitted"] / max(1, st["steps"]))
            if cfg:
                res[name + "_acc"].append(st["accepted"] / max(1, st["drafted"]))
                res[name + "_drate"].append(st["drafted"] / max(1, st["steps"]))
            outs[name] = out
        print(f"  round {r} done", flush=True)

    def med(k):
        v = sorted(res[k]); return v[len(v) // 2], v[0], v[-1]

    bt = med("base_tps")[0]; bm = med("base_mb")[0]
    print(f"\n=== {MODEL} / {PROMPT} / {ROUNDS} rounds, order alternated, warm ===")
    print(f"{'arm':<8}{'tok/s':>8}{'range':>16}{'MB/tok':>9}{'ratio':>8}"
          f"{'bytes':>8}{'tok/step':>10}{'accept':>8}{'draft/step':>11}")
    print(f"{'base':<8}{bt:>8.2f}{f'{med(chr(98)+chr(97)+chr(115)+chr(101)+chr(95)+chr(116)+chr(112)+chr(115))[1]:.2f}-{med(chr(98)+chr(97)+chr(115)+chr(101)+chr(95)+chr(116)+chr(112)+chr(115))[2]:.2f}':>16}"
          f"{bm:>9.1f}{1.0:>8.2f}{1.0:>8.2f}{1.0:>10.2f}{'-':>8}{'-':>11}")
    for name, cfg in arms[1:]:
        t, lo, hi = med(name + "_tps")
        print(f"{name:<8}{t:>8.2f}{f'{lo:.2f}-{hi:.2f}':>16}"
              f"{med(name+'_mb')[0]:>9.1f}{t/bt:>8.2f}"
              f"{med(name+'_mb')[0]/bm:>8.2f}{med(name+'_step')[0]:>10.2f}"
              f"{med(name+'_acc')[0]*100:>7.0f}%{med(name+'_drate')[0]:>11.2f}")

    b1, b2 = outs.get("base"), outs.get("base2")
    if b2 is not None:
        same = sum(a == b for a, b in zip(b1, b2))
        print(f"\nENGINE DETERMINISM: baseline vs baseline {same}/{len(b1)} identical"
              f"  -> {'deterministic' if same==len(b1) else 'NOT deterministic'}")
    for name, cfg in arms[1:]:
        s = sum(a == b for a, b in zip(outs[name], b1))
        print(f"  {name} vs base: {s}/{len(b1)} identical")


if __name__ == "__main__":
    main()
