#!/usr/bin/env python3
"""Cache the engine's real prefetch predictions for offline policy simulation.

The engine predicts layer L+1's experts by applying layer L+1's router to layer
L's hidden state (the "free projection"). Reproducing that here means the
simulator sees wrong guesses at their true rate -- which is exactly what the
demand-paging simulator missed.
"""
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/agastya/Desktop/moe-stream")
from subspace_q1b import router_logits

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")
NTOK = int(os.environ.get("NTOK", "6144"))
OUT = (f"/private/tmp/claude-501/-Users-agastya-Desktop-moe-stream/"
       f"32819ea4-feb4-45b0-94d1-5314981e9000/scratchpad/pred_{NTOK}.npy")


def main():
    routes = np.load(f"{ACTS}/routes.npy", mmap_mode="r")
    L = routes.shape[0]
    pred = np.zeros((L, NTOK, 4), dtype=np.int16)
    pred[0] = np.array(routes[0, :NTOK])          # layer 0: nothing to predict
    agree = []
    for i in range(L - 1):
        x = mx.array(np.array(np.load(f"{ACTS}/x_layer{i:02d}.npy",
                                      mmap_mode="r")[:NTOK]))
        lg = np.array(router_logits(i + 1, x))
        top = np.argpartition(lg, -4, axis=-1)[:, -4:]
        pred[i + 1] = top.astype(np.int16)
        true = np.array(routes[i + 1, :NTOK]).astype(np.int64)
        r = (top[:, :, None] == true[:, None, :]).any(1).mean()
        agree.append(r)
        del x, lg
    np.save(OUT, pred)
    print(f"saved {OUT}  shape {pred.shape}")
    print(f"mean prefetch recall {np.mean(agree)*100:.1f}%  "
          f"(engine reports 81.8%)")


if __name__ == "__main__":
    main()
