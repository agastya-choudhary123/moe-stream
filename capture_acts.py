#!/usr/bin/env python3
"""
Capture the MLP-block input activations of gpt-oss-120b over a diverse corpus.

`x` here is exactly what the expert GEMM multiplies: the output of
post_attention_layernorm, before the router. That is the vector whose subspace
the activation-compression idea is about -- store `W Q` per expert, compute
`z = Q^T x` once per layer.

Capture happens during PREFILL, one window per forward pass, because a prefill
of this model reads ~64 GB off the SSD whatever the length: a 1024-token window
costs the same I/O as a 1-token one and yields 1024 samples per layer. Decode
would need 1024 separate 400 ms tokens for the same data.

Everything is written as fp32 per layer into one memmap covering the whole
corpus, plus the routing indices, plus a manifest saying which rows came from
which genre and window. The splits the analysis needs (held-out documents,
leave-one-genre-out) are then just row selections.

  PF_SLOTS=300 python3 capture_acts.py
"""

import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models import gpt_oss
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_120b
from corpus import SEQ

OUT = os.path.expanduser("~/Desktop/moe-stream/acts")
N_LAYERS, D = 36, 2880

_REC = {}          # layer -> (x, routing indices) for the window in flight


def recording_call(self, x):
    """Wraps the streaming expert block: record x, then run it unchanged."""
    _REC[self._layer] = x
    return engine_120b.streaming_call(self, x)


def main():
    per_genre = int(os.environ.get("PER_GENRE", "3"))
    limit = int(os.environ.get("LIMIT", "0"))       # stop after N windows
    os.makedirs(OUT, exist_ok=True)

    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from pathlib import Path
    tok = load_tokenizer(Path(engine_120b.MODEL_DIR))
    import corpus
    windows = corpus.build(tok, per_genre=per_genre)
    if limit:
        windows = windows[:limit]
    n_rows = len(windows) * SEQ
    print(f"{len(windows)} windows x {SEQ} = {n_rows:,} samples/layer, "
          f"{n_rows*D*4/2**30*N_LAYERS:.1f} GiB")

    model, _, pool = engine_120b.load_engine()
    gpt_oss.MLPBlock.__call__ = recording_call
    print(f"pool {pool.n_slots} slots, resident "
          f"{mx.get_active_memory()/2**30:.2f} GiB")

    mm = [np.lib.format.open_memmap(f"{OUT}/x_layer{L:02d}.npy", mode="w+",
                                    dtype=np.float32, shape=(n_rows, D))
          for L in range(N_LAYERS)]
    routes = np.zeros((N_LAYERS, n_rows, 4), dtype=np.int16)
    manifest = []
    peak = 0.0

    for w, (genre, wi, ids) in enumerate(windows):
        t0 = time.perf_counter()
        cache = make_prompt_cache(model)
        _REC.clear()
        logits = model(mx.array(ids)[None], cache=cache)
        # routing is recomputed here rather than plumbed out of the engine:
        # same router, same input, deterministic.
        rec = [_REC[L].astype(mx.float32) for L in range(N_LAYERS)]
        blocks = engine_120b._BLOCKS
        rt = [mx.argpartition(blocks[L].router(_REC[L]), kth=-4, axis=-1)[..., -4:]
              for L in range(N_LAYERS)]
        mx.eval(logits, *rec, *rt)
        lo, hi = w * SEQ, (w + 1) * SEQ
        for L in range(N_LAYERS):
            a = np.array(rec[L], copy=False).reshape(SEQ, D).astype(np.float32)
            mm[L][lo:hi] = a
            routes[L, lo:hi] = np.array(rt[L], copy=False).reshape(SEQ, 4)
            peak = max(peak, float(np.abs(a).max()))
        manifest.append(dict(genre=genre, window=wi, lo=lo, hi=hi,
                             ids=ids[:8], seconds=time.perf_counter() - t0))
        print(f"  [{w+1}/{len(windows)}] {genre:<10} {time.perf_counter()-t0:6.1f}s "
              f"peak|x| {peak:8.1f}  resident {mx.get_active_memory()/2**30:.2f} GiB",
              flush=True)
        json.dump(dict(seq=SEQ, d=D, n_layers=N_LAYERS, windows=manifest,
                       peak_abs_x=peak, rows_done=hi),
                  open(f"{OUT}/manifest.json", "w"), indent=1)
        np.save(f"{OUT}/routes.npy", routes)

    for m in mm:
        m.flush()
    print(f"done. peak |x| = {peak:.1f}")


if __name__ == "__main__":
    main()
