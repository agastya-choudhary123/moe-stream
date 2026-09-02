#!/usr/bin/env python3
"""The ceiling: what does multi-token verification buy with a PERFECT drafter?

Separates the two factors that any speculative-decoding estimate multiplies
together -- how much cheaper a t-token forward is, and how many drafted tokens
survive. If the oracle does not pay, no drafter will.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")
os.environ.setdefault("MOE_COLD", "1")

from spec import OracleDrafter, spec_generate, baseline_generate, PROMPTS

MODEL = os.environ.get("MODEL", "30b")
NTOK = int(os.environ.get("TOKENS", "96"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
WARM = int(os.environ.get("WARM", "128"))
PROMPT = os.environ.get("PROMPT", "code")
GAMMAS = [int(g) for g in os.environ.get("GAMMAS", "1,3,7").split(",")]

E = __import__("engine_120b" if MODEL == "120b" else "engine_v3")


def main():
    model, tok, pool = E.load_engine()
    ids = tok.encode(PROMPTS.get(PROMPT, PROMPT))
    script_tail, _ = baseline_generate(model, ids, max(WARM, NTOK + 8))
    script = list(ids) + list(script_tail)
    print(f"warm; script {len(script)} tokens", flush=True)

    arms = ["base"] + [f"orc{g}" for g in GAMMAS]
    res = defaultdict(list)
    outs = {}
    for r in range(ROUNDS):
        order = arms if r % 2 == 0 else list(reversed(arms))
        for name in order:
            b0 = pool.bytes_read
            st = defaultdict(float)
            if name == "base":
                out, st = baseline_generate(model, ids, NTOK, st)
            else:
                g = int(name[3:])
                out, st = spec_generate(model, pool, ids, NTOK,
                                        OracleDrafter(script, g), g, st)
            res[name + "_tps"].append(st["emitted"] / st["t_decode"])
            res[name + "_mb"].append((pool.bytes_read - b0) / 2**20 / st["emitted"])
            res[name + "_step"].append(st["emitted"] / max(1, st["steps"]))
            if name != "base":
                res[name + "_acc"].append(st["accepted"] / max(1, st["drafted"]))
            outs[name] = out
        print(f"  round {r}", flush=True)

    def med(k):
        v = sorted(res[k]); return v[len(v)//2], v[0], v[-1]
    bt = med("base_tps")[0]; bm = med("base_mb")[0]
    print(f"\n=== {MODEL} / {PROMPT} / ORACLE drafter, {ROUNDS} rounds, "
          f"order alternated ===")
    print(f"{'arm':<8}{'tok/s':>8}{'range':>15}{'MB/tok':>9}{'ratio':>8}"
          f"{'bytes':>8}{'tok/step':>10}{'accept':>8}")
    lo, hi = med("base_tps")[1], med("base_tps")[2]
    print(f"{'base':<8}{bt:>8.2f}{f'{lo:.2f}-{hi:.2f}':>15}{bm:>9.1f}"
          f"{1.0:>8.2f}{1.0:>8.2f}{1.0:>10.2f}{'-':>8}")
    for name in arms[1:]:
        t, lo, hi = med(name + "_tps")
        print(f"{name:<8}{t:>8.2f}{f'{lo:.2f}-{hi:.2f}':>15}"
              f"{med(name+'_mb')[0]:>9.1f}{t/bt:>8.2f}"
              f"{med(name+'_mb')[0]/bm:>8.2f}{med(name+'_step')[0]:>10.2f}"
              f"{med(name+'_acc')[0]*100:>7.0f}%")
    b = outs["base"]
    for name in arms[1:]:
        print(f"  {name} vs base: "
              f"{sum(a==c for a,c in zip(outs[name],b))}/{len(b)} identical")


if __name__ == "__main__":
    main()
