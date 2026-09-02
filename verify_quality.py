#!/usr/bin/env python3
"""
Re-verify the quality claims the pool bug invalidated.

Throughput results never depended on the quantization scales being read from the
right address, so they stand. Everything about *values* was measured on a model
with roughly half its scales wrong, including the robustness pass (160 tokens,
no repetition collapse, no mojibake) and the claim that the 30B produces
bit-identical output at 1536 and 3072 slots. That last one cannot have been true
as stated -- 1536 was under the overflow limit and 3072 was well past it -- so it
is the sharpest available test that the fix is real.

  python3 verify_quality.py 120b      # long generation, leak check, pool state
  python3 verify_quality.py 30b       # 1536 vs 3072 slots, token-for-token
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = {
    "prose": "Explain why mixture-of-experts models are memory-bandwidth bound.",
    "code": "Write a Python function that merges two sorted lists.",
    "french": "Explique en francais pourquoi la memoire unifiee change "
              "l'inference des grands modeles.",
}

RUNNER = r'''
import json, os, sys, time
sys.path.insert(0, {here!r})
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
import {engine} as E
model, tok, pool = E.load_engine()
ids = tok.encode({prompt!r})
before = mx.get_active_memory() / 2**30
c = make_prompt_cache(model)
y = mx.argmax(model(mx.array(ids)[None], cache=c)[:, -1], axis=-1)
mx.eval(y)
out, t0 = [], time.perf_counter()
for _ in range({ntok}):
    y = mx.argmax(model(y[None], cache=c)[:, -1], axis=-1)
    mx.eval(y)
    out.append(y.item())
s = pool.stats()
print("@@" + json.dumps(dict(
    ids=out, text=tok.decode(out), tok_s={ntok} / (time.perf_counter() - t0),
    mem_before=before, mem_after=mx.get_active_memory() / 2**30,
    peak=mx.get_peak_memory() / 2**30, slots=pool.n_slots,
    resident_slots=len(pool.slot_of), pending=len(pool.pending),
    pinned=len(pool.pinned), hit=s["cache"] / max(s["total"], 1))))
'''


def run(engine, prompt, ntok, slots):
    env = dict(os.environ, PF_SLOTS=str(slots), MOE_COLD="0")
    src = RUNNER.format(here=HERE, engine=engine, prompt=prompt, ntok=ntok)
    p = subprocess.run([sys.executable, "-c", src], cwd=HERE, env=env,
                       text=True, capture_output=True)
    line = [l for l in p.stdout.split("\n") if l.startswith("@@")]
    if not line:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise SystemExit("run failed")
    return json.loads(line[0][2:])


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "120b"
    if which == "120b":
        print("gpt-oss-120b: 160 tokens per prompt on the fixed pool\n")
        for name, pr in PROMPTS.items():
            t0 = time.perf_counter()
            r = run("engine_120b", pr, 160, 600)
            print(f"[{name}] {r['tok_s']:.2f} tok/s, "
                  f"{time.perf_counter()-t0:.0f}s")
            print(f"  memory {r['mem_before']:.2f} -> {r['mem_after']:.2f} GiB "
                  f"(peak {r['peak']:.2f})   leak: "
                  f"{'NO' if r['mem_after'] - r['mem_before'] < 0.05 else 'YES'}")
            print(f"  pool {r['resident_slots']}/{r['slots']} resident, "
                  f"pending {r['pending']}, pinned {r['pinned']}, "
                  f"cache hit {r['hit']*100:.0f}%")
            print(f"  {r['text'][:400]!r}\n", flush=True)
    else:
        print("Qwen3-30B: are 1536 and 3072 slots token-for-token identical?")
        print("(they must be -- same weights, same order, only the cache differs;")
        print(" before the fix 3072 was past the overflow limit and 1536 was not)\n")
        a = run("engine_v3", PROMPTS["prose"], 48, 1536)
        b = run("engine_v3", PROMPTS["prose"], 48, 3072)
        same = a["ids"] == b["ids"]
        n = sum(1 for x, y in zip(a["ids"], b["ids"]) if x == y)
        print(f"  1536 slots: {a['tok_s']:.2f} tok/s   {a['text'][:120]!r}")
        print(f"  3072 slots: {b['tok_s']:.2f} tok/s   {b['text'][:120]!r}")
        print(f"\n  identical: {same}  ({n}/{len(a['ids'])} tokens match)")


if __name__ == "__main__":
    main()
