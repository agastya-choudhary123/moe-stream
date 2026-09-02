#!/usr/bin/env python3
"""
Does contextual sparsity survive the prefetch pipeline?

HANDOFF records that post-SwiGLU `h` is contextually sparse -- 90.2% of entries
below 5% of the row max, top 576 of 2880 holding 80.5% of the energy -- and
files it as worth ~25% of bytes/token, blocked on storing down_proj transposed
plus a partial-read kernel. Both of those are real engineering, but neither is
the actual obstacle.

The obstacle is scheduling. Which rows of down_proj matter depends on `h`, and
`h` is not known until gate/up have been read and multiplied. So the down_proj
read is DEPENDENT: it cannot be issued a layer early like everything else in
this engine, and 36 un-hidden SSD round trips per token would eat the 25% and
more. Building the kernel first and discovering that afterwards would be the
expensive order to do this in.

There is one way out, and it is the trick this project already relies on for
routing: predict from the previous layer. At layer L-1 the prefetcher has
already fetched layer L's expert weights, so it can compute a *provisional* h
using layer L-1's hidden state, rank the neurons by that, and issue the
down_proj row reads a full layer early. Nothing is on the critical path if the
prediction is good.

So the question is the same one Phase 1 asked about routing, one level down:
does x_{L-1} predict which neurons of layer L's experts will fire?

Three rankings are compared at equal bytes read:

  oracle      top rows by true contribution |h_j| * ||W_down[:,j]||  (upper bound)
  predicted   the same ranking computed from x_{L-1} instead of x_L
  static      one fixed row set per expert, by average contribution -- the
              baseline expert_neurons.py already measured as hopeless, included
              so the dynamic schemes are scored against something

Error is always measured with the TRUE h, since the engine has h; what it lacks
is the rows it did not read.

  LAYERS=4,18,35 NEXP=16 python3 sparsity_predict.py
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
FRAC = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


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
        Xl = np.load(f"{ACTS}/x_layer{L:02d}.npy", mmap_mode="r")
        Xp = np.load(f"{ACTS}/x_layer{L-1:02d}.npy", mmap_mode="r")
        rt = np.array(routes[L])
        picks = rng.choice(ex.n_experts, nexp, replace=False)
        err = {m: {f: [] for f in FRAC} for m in ("oracle", "predicted", "static")}
        overlap = {f: [] for f in FRAC}

        for e in picks:
            te_e = te_rows[np.any(rt[te_rows] == e, axis=1)]
            tr_e = tr_rows[np.any(rt[tr_rows] == e, axis=1)]
            if len(te_e) < 32 or len(tr_e) < 64:
                continue
            r = np.sort(te_e)
            W = ex.get(L, int(e))
            Wg, bg = W["gate_proj"]
            Wu, bu = W["up_proj"]
            Wd, bd = W["down_proj"]
            cn = mx.sqrt(mx.sum(Wd * Wd, axis=0))        # ||W_down[:, j]||

            def hid(src, rows):
                xs = mx.array(np.array(src[rows]))
                return swiglu(xs @ Wu.T + bu, xs @ Wg.T + bg)

            h_true = hid(Xl, r)
            h_pred = hid(Xp, r)                          # available a layer early
            h_stat = hid(Xl, np.sort(tr_e))              # for the static ranking
            score_t = mx.abs(h_true) * cn
            score_p = mx.abs(h_pred) * cn
            score_s = mx.mean(mx.abs(h_stat), axis=0) * cn
            mx.eval(h_true, score_t, score_p, score_s)
            st_, sp_ = np.array(score_t), np.array(score_p)
            ss_ = np.array(score_s)

            y_full = h_true @ Wd.T + bd
            nrm = float(mx.sqrt(mx.sum(y_full ** 2)))
            ot = np.argsort(-st_, axis=1)
            op = np.argsort(-sp_, axis=1)
            os_ = np.argsort(-ss_)

            for f in FRAC:
                m = max(1, int(round(f * 2880)))
                for name, order in (("oracle", ot), ("predicted", op)):
                    keep = np.zeros((len(r), 2880), dtype=np.float32)
                    np.put_along_axis(keep, order[:, :m], 1.0, axis=1)
                    y = (h_true * mx.array(keep)) @ Wd.T + bd
                    err[name][f].append(
                        float(mx.sqrt(mx.sum((y - y_full) ** 2))) / nrm)
                keep = np.zeros(2880, dtype=np.float32)
                keep[os_[:m]] = 1.0
                y = (h_true * mx.array(keep)) @ Wd.T + bd
                err["static"][f].append(
                    float(mx.sqrt(mx.sum((y - y_full) ** 2))) / nrm)
                inter = [len(set(a[:m]) & set(b[:m])) / m
                         for a, b in zip(ot, op)]
                overlap[f].append(float(np.mean(inter)))

        r_ = dict(n_experts=len(err["oracle"][1.0]),
                  err={k: {f: float(np.mean(v)) for f, v in d.items()}
                       for k, d in err.items()},
                  overlap={f: float(np.mean(v)) for f, v in overlap.items()})
        out[L] = r_
        print(f"L{L:02d}  {time.perf_counter()-t0:5.1f}s  {r_['n_experts']} experts")
        print(f"      {'rows read':>10} {'oracle':>9} {'predicted':>11} "
              f"{'static':>9} {'set overlap':>13}")
        for f in FRAC:
            print(f"      {f*100:9.0f}% {r_['err']['oracle'][f]:9.4f} "
                  f"{r_['err']['predicted'][f]:11.4f} "
                  f"{r_['err']['static'][f]:9.4f} "
                  f"{r_['overlap'][f]*100:12.1f}%", flush=True)
        json.dump({str(k): v for k, v in out.items()},
                  open(f"{ACTS}/sparsity_predict.json", "w"), indent=1)

    print("\nBytes: down_proj is a third of the blob, so reading a fraction f of "
          "its rows takes the blob to (2 + f)/3 of its size.")


if __name__ == "__main__":
    main()
