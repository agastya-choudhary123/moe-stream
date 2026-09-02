#!/usr/bin/env python3
"""
The intermediate dimension: are some of an expert's 2880 neurons simply dead?

Every compression idea in this project so far attacked a 2880-wide *space* --
the input subspace, the output subspace, the weight rank. None of them touched
the one axis where a saving costs nothing to implement: the intermediate width.

Neuron j of an expert contributes exactly  h_j * W_down[:, j]  to that expert's
output. Deleting it means dropping row j of gate_proj and up_proj and column j
of down_proj, which shrinks ALL THREE matrices by the same fraction. Nothing
else considered here scales the whole blob like that -- the input-side schemes
could never touch down_proj, and the output-side scheme could never touch
gate/up. And it needs no kernel, no layout change and no runtime decision: a
pruned expert is just a smaller expert.

The prior is good for MoE specifically. Each expert sees roughly 1/32 of the
tokens, so individual experts are far less trained than a dense FFN of the same
width, and dead units are exactly what under-training produces.

This is a STATIC question, unlike the contextual sparsity already recorded in
HANDOFF.md (90.2% of h entries below 5% of the row max, per token). Contextual
sparsity needs a partial-read kernel and a transposed layout to exploit. A dead
neuron needs nothing.

Ranking is by the contribution a neuron actually makes,
E[h_j^2] * ||W_down[:, j]||^2, not by h alone -- a large activation into a small
down-projection column moves the output no more than the reverse.

Selection uses training tokens, error is measured on held-out tokens of the same
expert, because "which neurons matter" fitted and scored on the same tokens is
the artifact this project has already been caught by three times.

  LAYERS=4,18,35 NEXP=16 python3 expert_neurons.py
"""

import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.gpt_oss import swiglu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_q1b import Experts

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")
KEEP = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def main():
    layers = [int(s) for s in os.environ.get("LAYERS", "4,18,35").split(",")]
    nexp = int(os.environ.get("NEXP", "16"))
    man = json.load(open(f"{ACTS}/manifest.json"))
    wins = man["windows"]
    n_per = max(w["window"] for w in wins) + 1
    tr_rows = np.concatenate([np.arange(w["lo"], w["hi"]) for w in wins
                              if w["window"] < n_per - 1])
    te_rows = np.concatenate([np.arange(w["lo"], w["hi"]) for w in wins
                              if w["window"] == n_per - 1])
    routes = np.load(f"{ACTS}/routes.npy", mmap_mode="r")
    ex = Experts()
    rng = np.random.default_rng(0)
    out = {}

    for L in layers:
        t0 = time.perf_counter()
        X = np.load(f"{ACTS}/x_layer{L:02d}.npy", mmap_mode="r")
        rt = np.array(routes[L])
        picks = rng.choice(ex.n_experts, nexp, replace=False)
        errs = {f: [] for f in KEEP}
        dead, ntok = [], []

        for e in picks:
            tr_e = tr_rows[np.any(rt[tr_rows] == e, axis=1)]
            te_e = te_rows[np.any(rt[te_rows] == e, axis=1)]
            if len(tr_e) < 64 or len(te_e) < 32:
                continue
            W = ex.get(L, int(e))
            Wg, bg = W["gate_proj"]
            Wu, bu = W["up_proj"]
            Wd, bd = W["down_proj"]

            def hidden(rows):
                xs = mx.array(np.array(X[np.sort(rows)]))
                return swiglu(xs @ Wu.T + bu, xs @ Wg.T + bg)

            Htr, Hte = hidden(tr_e), hidden(te_e)
            col = mx.sum(Wd * Wd, axis=0)               # ||W_down[:, j]||^2
            score = mx.mean(Htr * Htr, axis=0) * col     # contribution of neuron j
            mx.eval(score, Hte)
            sc = np.array(score)
            order = np.argsort(-sc)
            # "dead" = contributes less than a millionth of the mean neuron
            dead.append(float((sc < sc.mean() * 1e-6).mean()))
            ntok.append(len(tr_e))

            y_full = Hte @ Wd.T + bd
            nrm = float(mx.sqrt(mx.sum(y_full ** 2)))
            for f in KEEP:
                keep = np.zeros(2880, dtype=np.float32)
                keep[order[:int(round(f * 2880))]] = 1.0
                y = (Hte * mx.array(keep)) @ Wd.T + bd
                errs[f].append(float(mx.sqrt(mx.sum((y - y_full) ** 2))) / nrm)

        r = dict(n_experts=len(ntok), tokens_per_expert=float(np.mean(ntok)),
                 dead_fraction=float(np.mean(dead)),
                 err={f: float(np.mean(v)) for f, v in errs.items()})
        out[L] = r
        print(f"L{L:02d}  {time.perf_counter()-t0:5.1f}s  {r['n_experts']} experts, "
              f"{r['tokens_per_expert']:.0f} train tokens each, "
              f"{r['dead_fraction']*100:.2f}% of neurons effectively dead")
        for f in KEEP:
            print(f"      keep {f*100:5.1f}% of neurons -> blob {f:.2f}x, "
                  f"held-out output error {r['err'][f]:.4f}", flush=True)
        json.dump({str(k): v for k, v in out.items()},
                  open(f"{ACTS}/expert_neurons.json", "w"), indent=1)


if __name__ == "__main__":
    main()
