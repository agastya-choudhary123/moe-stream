"""Speculative decoding for the streaming MoE engine.

STATUS: the mechanism is proven and the drafter is not. With a perfect drafter
this is worth 1.45-1.53x end to end at gamma=7 and costs no extra bytes; with
the n-gram drafter shipped here it is 1.11x on a 96-token generation and
0.78-0.88x on a 192-token one. It is a research instrument, not a default. See
HANDOFF.md, "Multi-token verification", for every number and for why.

WHY THE MECHANISM IS RIGHT HERE, and why it is not the usual argument.

On a GPU that holds the whole model, speculative decoding on an A3B-shaped MoE
is famously marginal-to-negative -- the target forward is already cheap, so the
drafter's cost dominates. This engine is in a different regime for a specific
reason: a forward pass costs a large FIXED amount that does not depend on how
many tokens are in it. Measured (scratchpad/batch_cost.py, replayed token
sequence so routing is identical across arms, 3 rounds, order alternated,
Qwen3-30B at 3072 slots):

    t     ms/forward   ms/position   speedup
    1         40.9        40.91       1.00x
    2         55.0        27.51       1.49x
    4         85.7        21.42       1.91x
    8        151.7        18.96       2.16x

which fits T(t) = 25.1 + 15.8*t ms. That 25.1 ms floor is 48 CPU-GPU sync round
trips plus attention and lm_head, all of which HANDOFF measured as
batch-independent (~171 us of every 210 us router invocation is the round trip,
and the non-expert weights are bandwidth-bound at any batch). Verifying t tokens
in one forward pays that floor once instead of t times. HANDOFF had already
concluded this 25.1 ms was architectural and unfixable; it is unfixable per
forward and amortisable per token.

The expert path amortises too, independently: the union of routed experts grows
far slower than t because consecutive tokens reuse experts above chance
(scratchpad/union_trace.py). At t=8 the 30B touches 29.1 experts per layer where
independent routing predicts 51.6.

THE DRAFTER HAS TO BE FREE, and that is the binding constraint. Any drafter that
runs the full stack -- self-speculation at reduced top-k, an early-exit head, a
smaller MoE -- pays that same 25.1 ms per drafted token, which is most of what
we are trying to save. A 0.6B dense draft model is ~8-12 ms/token here, so
gamma=3 at 75% acceptance costs 118.3 ms for 2.73 tokens against a 40.9 ms
baseline: a loss before it starts. So this ships prompt-lookup (n-gram)
drafting, which costs a dict lookup and proposes nothing when it has no
confident match.

That turns out not to be enough. The n-gram drafter fires on only 0.45-1.03
steps out of every step and at 42-78% acceptance, giving 1.19-1.80 tokens per
step against the 7.31 the oracle reaches, while every rejected position still
pays a full set of expert reads. The open problem is a drafter that is free AND
fires every step; HANDOFF argues for Medusa-style heads on the final hidden
state, which cost ~1-2 ms and no extra pass through the 48 layers.

LOSSLESS, with one honest caveat. The accept rule is exact: a drafted token is
kept only where it equals the target model's own argmax at that position. But a
batched forward does not compute bit-identical logits to t sequential forwards,
and at a bf16 near-tie the argmax itself moves. The engine alone IS
deterministic (baseline vs baseline: 192/192 identical ids). Speculative vs
baseline is 64/64 at small gamma and 61/192 at gamma=4 -- they agree until one
near-tie flips, then separate. Read that as "exact accept rule, bf16-order-
dependent argmax", not as bit-identical output.
"""


import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


# ---------------------------------------------------------------- drafters
class NGramDrafter:
    """Prompt-lookup drafting: propose the continuation that followed the last
    time this n-gram appeared.

    Costs no forward pass, which is the whole point (see module docstring). It
    fires on the repetition that real decoding is full of -- quoted context,
    code, identifiers, list structure -- and proposes nothing otherwise, so the
    step degrades to an ordinary single-token step rather than to a loss.

    Longest match first: a 3-gram match is a much better predictor than a
    1-gram, so try wide before narrow and take the first hit.

    min_n matters more than it looks. With min_n=1 the drafter almost always
    finds *something* -- any repeated single token -- and proposes a
    continuation that is close to noise. Measured on the 30B: min_n=1 drafts on
    nearly every step at 34% acceptance, and every rejected position still costs
    a full set of expert reads. A draft that is not going to be accepted is
    strictly worse than no draft at all here, because bytes are the scarce
    resource. Refusing to draft is a real option and the drafter has to use it.
    """

    def __init__(self, max_n=4, min_n=2, gamma=4):
        self.max_n, self.min_n, self.gamma = max_n, min_n, gamma

    def draft(self, ids, gamma=None):
        g = gamma or self.gamma
        n_ids = len(ids)
        for n in range(self.max_n, self.min_n - 1, -1):
            if n_ids < n + 1:
                continue
            key = ids[-n:]
            # scan backwards: the most recent occurrence is the best predictor
            for s in range(n_ids - n - 1, -1, -1):
                if ids[s:s + n] == key:
                    cand = ids[s + n:s + n + g]
                    if cand:
                        return list(cand)
            # nothing at this width; fall through to a narrower one
        return []


class OracleDrafter:
    """A drafter that is always right, for measuring the ceiling.

    Every estimate of what speculative decoding is worth here depends on two
    unknowns multiplied together: how much cheaper a t-token forward is
    (measured, batch_cost.py) and how many of the drafted tokens survive
    verification (a property of the drafter). Separating them matters, because
    if even a PERFECT drafter does not pay, no amount of drafter engineering
    will -- which is exactly how this project closed contextual sparsity, where
    the oracle row of the table was the one that settled it.

    So: pre-generate the greedy continuation, hand it to the drafter, and
    measure. Acceptance is 100% by construction as long as the run stays on the
    script; the moment it leaves (a bf16 near-tie flipping an argmax) the
    drafter stops proposing rather than proposing garbage.
    """

    def __init__(self, script, gamma=4):
        self.script = list(script)
        self.gamma = gamma

    def draft(self, ids, gamma=None):
        g = gamma or self.gamma
        n = len(ids)
        if n + 1 > len(self.script) or list(ids) != self.script[:n]:
            return []                      # off script: stop proposing
        return self.script[n:n + g]


class NoDrafter:
    def draft(self, ids, gamma=None):
        return []


# ---------------------------------------------------------------- the loop
def spec_generate(model, pool, ids, max_tokens, drafter, gamma=4,
                  stats=None):
    """Greedy decoding, accelerated by verifying gamma+1 positions per forward.

    One step:
      cur                      the token sampled last step, not yet in the cache
      d_1..d_g                 the drafter's guesses for what follows it
      forward([cur, d_1..d_g]) gives the model's own argmax after each position
      accept the longest prefix where the model agrees with the drafter, then
      take the model's argmax at the first disagreement as a free bonus token

    So a step emits between 1 and gamma+1 tokens and always emits at least one,
    which is what makes an unlucky draft cost time but never correctness. The
    cache is then trimmed back to the accepted prefix: the rejected positions
    were really evaluated, so their KV entries exist and are wrong.
    """
    st = stats if stats is not None else defaultdict(float)
    cache = make_prompt_cache(model)

    t0 = time.perf_counter()
    logits = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(logits[:, -1], axis=-1).item())
    st["prefill_s"] = time.perf_counter() - t0

    out = [cur]
    ctx = list(ids) + [cur]
    st["t_decode"] = 0.0
    tdec = time.perf_counter()

    while len(out) < max_tokens:
        d = drafter.draft(ctx, gamma)
        # never draft past the token budget
        d = d[:max(0, max_tokens - len(out) - 1)]
        st["steps"] += 1
        st["drafted"] += len(d)
        st["hist_%d" % len(d)] += 1

        batch = [cur] + d
        logits = model(mx.array(batch)[None], cache=cache)
        m = np.array(mx.argmax(logits[0], axis=-1))   # m[i] follows batch[i]
        mx.eval(logits)

        n = 0
        while n < len(d) and int(m[n]) == d[n]:
            n += 1
        # d[:n] confirmed; m[n] is the model's own token at the first divergence
        new = d[:n] + [int(m[n])]

        # rejected positions were evaluated and are in the cache; drop them
        reject = len(d) - n
        if reject:
            trim_prompt_cache(cache, reject)

        out.extend(new)
        ctx.extend(new)
        cur = new[-1]
        st["accepted"] += n
        st["emitted"] += len(new)

    st["t_decode"] = time.perf_counter() - tdec
    return out[:max_tokens], st


def baseline_generate(model, ids, max_tokens, stats=None):
    st = stats if stats is not None else defaultdict(float)
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(mx.array(ids)[None], cache=cache)
    y = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(y)
    st["prefill_s"] = time.perf_counter() - t0
    out = [y.item()]
    tdec = time.perf_counter()
    while len(out) < max_tokens:
        y = mx.argmax(model(y[None], cache=cache)[:, -1], axis=-1)
        mx.eval(y)
        out.append(y.item())
    st["t_decode"] = time.perf_counter() - tdec
    st["steps"] = len(out) - 1
    st["emitted"] = len(out) - 1
    return out, st


# ---------------------------------------------------------------- driver
PROMPTS = {
    "prose": "Explain why mixture-of-experts models are memory-bandwidth bound.",
    "code": ("Here is a Python function:\n\n"
             "def merge_sorted(a, b):\n"
             "    out = []\n"
             "    i = j = 0\n"
             "    while i < len(a) and j < len(b):\n"
             "        if a[i] <= b[j]:\n"
             "            out.append(a[i]); i += 1\n"
             "        else:\n"
             "            out.append(b[j]); j += 1\n"
             "    out.extend(a[i:]); out.extend(b[j:])\n"
             "    return out\n\n"
             "Now write the same function for three sorted lists, "
             "following the same style exactly:\n\n"
             "def merge_sorted3(a, b, c):\n"),
    "repeat": ("Repeat the following list back exactly, then continue it:\n"
               "alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, "
               "india, juliet, kilo, lima, mike, november, oscar, papa\n\n"
               "The list is: alpha, bravo, charlie, delta, echo, foxtrot, "),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="30b", choices=["30b", "120b"])
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompt", default="prose")
    ap.add_argument("--drafter", default="ngram", choices=["ngram", "none"])
    ap.add_argument("--verify", action="store_true",
                    help="also decode greedily and assert identical token ids")
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    E = __import__("engine_120b" if args.model == "120b" else "engine_v3")
    model, tok, pool = E.load_engine()
    prompt = PROMPTS.get(args.prompt, args.prompt)
    ids = tok.encode(prompt)
    drafter = NGramDrafter(gamma=args.gamma) if args.drafter == "ngram" else NoDrafter()

    for r in range(args.rounds):
        st = defaultdict(float)
        b0 = pool.bytes_read
        out, st = spec_generate(model, pool, ids, args.tokens, drafter,
                                args.gamma, st)
        mb = (pool.bytes_read - b0) / 2 ** 20
        tps = st["emitted"] / st["t_decode"]
        acc = st["accepted"] / max(1, st["drafted"])
        print(f"[{args.model} {args.prompt} gamma={args.gamma} "
              f"drafter={args.drafter}] "
              f"{tps:6.2f} tok/s   {st['steps']:.0f} steps for "
              f"{st['emitted']:.0f} tokens   "
              f"{st['emitted']/st['steps']:.2f} tok/step   "
              f"accept {acc*100:.0f}% of {st['drafted']:.0f} drafted   "
              f"{mb/max(1,st['emitted']):.0f} MB/token", flush=True)

    if args.verify:
        base, bst = baseline_generate(model, ids, args.tokens)
        same = list(out[:len(base)]) == list(base[:len(out)])
        print(f"\nLOSSLESS CHECK: {'PASS' if same else 'FAIL'}  "
              f"({sum(a == b for a, b in zip(out, base))}/{min(len(out),len(base))}"
              f" ids identical)")
        print(f"  baseline {bst['emitted']/bst['t_decode']:.2f} tok/s   "
              f"speculative {tps:.2f} tok/s   "
              f"{tps/(bst['emitted']/bst['t_decode']):.2f}x")
        if not same:
            print(f"  spec: {tok.decode(out)!r}")
            print(f"  base: {tok.decode(base)!r}")
        else:
            print(f"  text: {tok.decode(out)!r}")


if __name__ == "__main__":
    main()
