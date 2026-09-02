#!/usr/bin/env python3
"""
Q2: end-to-end quality at a given k, on the real 120b engine.

No repack is needed to measure this, and that is worth stating plainly:
storing `W Q` per expert and computing `z = Q^T (x - mu)` produces exactly the
same numbers as leaving the weights alone and feeding the expert block
`mu + Q Q^T (x - mu)`, because W (Q Q^T u) = (W Q) (Q^T u). So the engine runs
unmodified except for one projection on the expert-block input, and the output
is what a compressed blob would produce with bf16 factors. (4-bit factors add
quantization on top; that is Q3, in subspace_q1b.py.)

The router is deliberately NOT projected: it lives in nonexpert.safetensors,
stays resident, and is never compressed, so in the real design it sees the full
x. Routing therefore does not change, which keeps the comparison clean -- any
quality loss is the projection itself, not a different set of experts. That is
what engine_120b.EXPERT_INPUT exists for.

Teacher-forced over held-out windows the basis never saw:

    NLL      mean negative log likelihood of the true next token
    dKL      KL(baseline || projected), the sensitive metric
    top1     fraction of positions whose argmax is unchanged

  PF_SLOTS=460 K=320 python3 subspace_q2.py
"""

import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_120b

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")
NSUB = 256          # positions kept for the logit-level metrics


def logprobs(model, ids):
    lg = model(ids, cache=make_prompt_cache(model))[0, :-1]
    lg = lg.astype(mx.float32)
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)


def main():
    k = int(os.environ.get("K", "320"))
    nwin = int(os.environ.get("NWIN", "8"))
    b = np.load(f"{ACTS}/basis.npz")
    man = json.load(open(f"{ACTS}/manifest.json"))
    last = max(x["window"] for x in man["windows"])
    wins = [w for w in man["windows"] if w["window"] == last][:nwin]

    import corpus
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from pathlib import Path
    tok = load_tokenizer(Path(engine_120b.MODEL_DIR))
    by = {(g, i): ids for g, i, ids in corpus.build(tok, per_genre=last + 1)}

    model, _, pool = engine_120b.load_engine()
    print(f"k={k}, {len(wins)} held-out windows, pool {pool.n_slots} slots")

    MU = [mx.array(b["mu"][l]) for l in range(man["n_layers"])]
    Q = [mx.array(np.ascontiguousarray(b["V"][l][:, :k]))
         for l in range(man["n_layers"])]

    def project(layer, x):
        u = x.astype(mx.float32) - MU[layer]
        return (MU[layer] + (u @ Q[layer]) @ Q[layer].T).astype(x.dtype)

    out = []
    for w in wins:
        ids = mx.array(by[(w["genre"], w["window"])])[None]
        tgt = ids[0, 1:]
        sub = mx.array(np.linspace(0, tgt.size - 1, NSUB).astype(np.int32))
        row = dict(genre=w["genre"])
        ref = None
        for mode in ("base", "proj"):
            engine_120b.EXPERT_INPUT = project if mode == "proj" else None
            t0 = time.perf_counter()
            lp = logprobs(model, ids)
            nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).mean()
            keep = lp[sub]
            am = mx.argmax(keep, axis=-1)
            mx.eval(nll, keep, am)
            row[f"nll_{mode}"] = float(nll)
            row[f"sec_{mode}"] = time.perf_counter() - t0
            if mode == "base":
                ref, ref_am = keep, am
            else:
                kl = mx.sum(mx.exp(ref) * (ref - keep), axis=-1)
                mx.eval(kl)
                kls = np.sort(np.array(kl))
                row["kl_mean"] = float(kls.mean())
                row["kl_p99"] = float(kls[int(0.99 * len(kls))])
                row["top1_agree"] = float((am == ref_am).mean())
            del lp
        engine_120b.EXPERT_INPUT = None
        out.append(row)
        print(f"  {row['genre']:<10} nll {row['nll_base']:.4f} -> "
              f"{row['nll_proj']:.4f} ({row['nll_proj']-row['nll_base']:+.4f})  "
              f"KL {row['kl_mean']:.5f}  top1 {row['top1_agree']*100:.1f}%",
              flush=True)
        json.dump(dict(k=k, nsub=NSUB, rows=out),
                  open(f"{ACTS}/q2_e2e_k{k}.json", "w"), indent=1)

    nb = float(np.mean([r["nll_base"] for r in out]))
    npj = float(np.mean([r["nll_proj"] for r in out]))
    print(f"\nk={k}: NLL {nb:.4f} -> {npj:.4f} ({npj-nb:+.4f}), "
          f"ppl {np.exp(nb):.3f} -> {np.exp(npj):.3f} "
          f"(x{np.exp(npj-nb):.4f}), KL {np.mean([r['kl_mean'] for r in out]):.5f}, "
          f"top1 {np.mean([r['top1_agree'] for r in out])*100:.2f}%")


if __name__ == "__main__":
    main()
