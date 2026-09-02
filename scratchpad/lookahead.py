#!/usr/bin/env python3
"""How far ahead can the engine see its own routing?

Two horizons matter for eviction and prefetch:
  DEPTH  -- from x at layer i, predict layer j>i routes for the SAME token.
            This is what the existing prefetcher does at d=1 (84.5% recall).
  TOKEN  -- predict the NEXT token's routes at the same layer.

Baseline for both is the "free projection": apply the target layer's real router
to the hidden state we already have. No training, no parameters.
"""
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")
from subspace_q1b import router_logits

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")
NTOK = int(os.environ.get("NTOK", "4096"))
TOPK = 4


def topk(logits, k=TOPK):
    return np.argpartition(np.asarray(logits), -k, axis=-1)[:, -k:]


def recall(pred, true, k):
    """Fraction of the 4 true experts present in the top-k prediction."""
    hit = (pred[:, :, None] == true[:, None, :]).any(1)
    return hit.mean()


def main():
    routes = np.load(f"{ACTS}/routes.npy", mmap_mode="r")
    L = routes.shape[0]
    print(f"{NTOK} tokens, {L} layers, top-{TOPK}\n")

    print("DEPTH: apply layer (i+d)'s router to layer i's hidden state")
    print(f"  {'d':>3} {'recall@4':>9} {'recall@8':>9} {'recall@16':>10}"
          f" {'recall@32':>10}")
    src = {}
    for i in (0, 9, 18):
        src[i] = mx.array(np.array(np.load(f"{ACTS}/x_layer{i:02d}.npy",
                                           mmap_mode="r")[:NTOK]))
    for d in (1, 2, 4, 8, 18, 35):
        rs = []
        for i in (0, 9, 18):
            j = i + d
            if j >= L:
                continue
            lg = np.array(router_logits(j, src[i]))
            true = np.array(routes[j, :NTOK]).astype(np.int64)
            rs.append([recall(topk(lg, k), true, k) for k in (4, 8, 16, 32)])
        if rs:
            m = np.mean(rs, axis=0)
            print(f"  {d:3d} {m[0]:9.3f} {m[1]:9.3f} {m[2]:10.3f} "
                  f"{m[3]:10.3f}")

    print("\nTOKEN: predict token t+1's routes at the SAME layer")
    print(f"  {'layer':>5} {'persist':>9} {'freeproj@4':>11} {'freeproj@8':>11}"
          f" {'freeproj@16':>12}")
    for i in (0, 9, 18):
        true_next = np.array(routes[i, 1:NTOK + 1]).astype(np.int64)
        cur = np.array(routes[i, :NTOK]).astype(np.int64)
        n = min(len(true_next), len(cur))
        persist = recall(cur[:n], true_next[:n], 4)
        lg = np.array(router_logits(i, src[i]))[:n]
        fr = [recall(topk(lg, k)[:n], true_next[:n], k) for k in (4, 8, 16)]
        print(f"  {i:5d} {persist:9.3f} {fr[0]:11.3f} {fr[1]:11.3f} "
              f"{fr[2]:12.3f}")
    print(f"\n  chance @4 = {TOPK/128:.3f}")


if __name__ == "__main__":
    main()
