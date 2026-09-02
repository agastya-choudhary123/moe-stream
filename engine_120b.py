#!/usr/bin/env python3
"""
Streaming engine for gpt-oss-120b on a 16 GB M4.

Same machinery as engine_v3 -- zero-copy slot pool, LFU residency, free
cross-layer prefetch -- pointed at a 64.6 GB model instead of a 16 GB one. The
pool and the prefetcher are model-agnostic; what changes is geometry and the
shape of one MoE block.

Differences from Qwen3-30B that matter:

  Geometry. 36 layers x 128 experts, top-4 (Qwen3: 48 x 128, top-8), hidden
  2880, intermediate 2880. An expert blob is 14.02 MB against Qwen3's 2.53 MB,
  so the SAME 8 GiB of pool holds only ~613 of 4608 experts -- 13% of the model
  resident, where Qwen3 gets 50%. The capacity curve in HANDOFF.md says that is
  the wrong side of the cliff, so expect single-digit tok/s, not the 21.8 the
  30B does. Fitting at all is the result here.

  Routing. gpt-oss takes top-k of the RAW router logits and softmaxes only the
  k selected, where Qwen3 softmaxes all 128 then takes top-k. Same argmax, but
  it means the speculation path needs no softmax at all -- argpartition on the
  logits is exactly what routing does.

  Biases. Every projection carries a real bias vector on top of the
  quantization biases, so a blob has twelve components, not nine.

  Activation. Clamped SwiGLU: clip gate to <=7, clip linear to [-7, 7],
  gate*sigmoid(1.702*gate) * (linear + 1). The +1 on the linear branch is not
  a typo, it is what gpt-oss does.
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import gpt_oss
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tokenizer_utils import load as load_tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_v3 import (ExpertPool, PREFETCH_DEPTHS, PREFETCH_DEPTH, PROF,
                       PF_WIDEN, N_WORKERS)

MODEL_DIR = os.path.expanduser("~/Desktop/moe-stream/model-120b")
# 600 slots = 8.41 GB pool, 8.97 GiB resident. Capacity is the only lever that
# moves this model (measured: 360 -> 1.34 tok/s, 480 -> 1.84, 600 -> 2.51, with
# true hit rate 31.5 -> 38.9 -> 53.4%), and 613 is the hard ceiling because
# Metal's max_buffer_length is 8 GiB. RAM would allow ~720; going past 613 needs
# the pool split across several MTLBuffers, which gather_qmm cannot address
# (it takes one weight tensor) -- that is what kernels.py would be for here.
N_SLOTS = int(os.environ.get("PF_SLOTS", "600"))

_BLOCKS = []
_POOL = None
EXPERT_INPUT = None      # optional (layer, x) -> x' applied to the expert path only
_PRED = {}
_RECALL = []


def top_k_route(router, x, k):
    """gpt-oss routing: top-k of the raw logits, softmax over just those k."""
    g = router(x)
    inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
    return inds, mx.softmax(mx.take_along_axis(g, inds, axis=-1), axis=-1,
                            precise=True)


def top_k_indices(router, x, k):
    """Indices only, for speculation -- no softmax, no take_along_axis."""
    return mx.argpartition(router(x), kth=-k, axis=-1)[..., -k:]


def streaming_call(self, x):
    pool, layer, blocks = _POOL, self._layer, _BLOCKS
    k = self.num_experts_per_tok

    # route + speculate, one sync for both (see HANDOFF "one sync per layer")
    _t = time.perf_counter()
    inds, scores = top_k_route(self.router, x, k)
    preds = []
    for d in PREFETCH_DEPTHS:
        tgt = layer + d
        if tgt >= len(blocks):
            continue
        kp = min(blocks[tgt].num_experts_per_tok * PF_WIDEN, pool.n_experts)
        preds.append((tgt, d, top_k_indices(blocks[tgt].router, x, kp)))
    mx.eval(inds, *(p for _, _, p in preds))
    PROF["router"] += time.perf_counter() - _t

    _t = time.perf_counter()
    for tgt, d, pi in preds:
        pred = np.unique(np.array(pi, copy=False).reshape(-1))
        if d == PREFETCH_DEPTH:
            _PRED[tgt] = set(pred.tolist())
        pool.prefetch(tgt, pred)
    PROF["predict"] += time.perf_counter() - _t

    _t = time.perf_counter()
    flat = np.array(inds, copy=False).reshape(-1)
    if flat.size == k:                    # batch 1: top-k are distinct already
        uniq, inverse = flat, None
    else:
        uniq, inverse = np.unique(flat, return_inverse=True)
    PROF["bookkeep"] += time.perf_counter() - _t

    if layer in _PRED:
        want = set(uniq.tolist())
        _RECALL.append(len(want & _PRED.pop(layer)) / len(want))

    _t = time.perf_counter()
    slots = pool.acquire(layer, uniq)
    PROF["acquire"] += time.perf_counter() - _t

    _t = time.perf_counter()
    sl = np.asarray(slots, dtype=np.uint32)
    idx = mx.array((sl if inverse is None else sl[inverse]).reshape(inds.shape))
    PROF["bookkeep"] += time.perf_counter() - _t

    _t = time.perf_counter()
    # Hook for the activation-subspace experiment: the expert block may run on a
    # transformed activation while routing above still sees the real one, which
    # is the true division of labour -- the router is a non-expert weight, stays
    # resident, and is never compressed. Default None, so the engine is
    # unchanged unless something installs it.
    xe = mx.expand_dims(x if EXPERT_INPUT is None else EXPERT_INPUT(layer, x),
                        (-2, -3))
    v = pool.views
    common = dict(rhs_indices=idx, transpose=True,
                  group_size=pool.group_size, bits=pool.bits)

    def proj(name, inp):
        y = mx.gather_qmm(inp, v[f"{name}.weight"], scales=v[f"{name}.scales"],
                          biases=v[f"{name}.biases"], **common)
        return y + mx.expand_dims(v[f"{name}.bias"][idx], -2)

    x_up = proj("up_proj", xe)
    x_gate = proj("gate_proj", xe)
    h = gpt_oss.swiglu(x_up, x_gate)        # clamped, alpha 1.702, limit 7.0
    y = proj("down_proj", h)
    y = (y.squeeze(-2) * scores[..., None]).sum(axis=-2)
    if pool.needs_barrier:
        mx.eval(y)
    PROF["expert_gemm"] += time.perf_counter() - _t
    return y


def load_engine(model_dir=MODEL_DIR, n_slots=N_SLOTS):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    model = gpt_oss.Model(gpt_oss.ModelArgs.from_dict(cfg))
    weights = mx.load(os.path.join(model_dir, "nonexpert.safetensors"))
    q = cfg["quantization"]

    def class_predicate(p, m):
        if p in q:
            return q[p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights

    nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                mode=q.get("mode", "affine"), class_predicate=class_predicate)

    # top_k drives the pool's barrier heuristic (a slot must not be recycled
    # within a token). The repack index does not carry it, and the Qwen3
    # default of 8 would make 36*8*2*1.25 = 720 > 560 slots and switch the
    # per-layer barrier on -- 36 pointless GPU round trips per token.
    ipath = os.path.join(model_dir, "experts_index.json")
    ridx = json.load(open(ipath))
    if ridx.get("top_k") != cfg["num_experts_per_tok"]:
        ridx["top_k"] = cfg["num_experts_per_tok"]
        json.dump(ridx, open(ipath, "w"), indent=1)

    global _POOL
    _POOL = ExpertPool(model_dir, n_slots=n_slots)
    gpt_oss.MLPBlock.__call__ = streaming_call
    _BLOCKS.clear()
    _BLOCKS.extend(l.mlp for l in model.model.layers)
    for i, blk in enumerate(_BLOCKS):
        blk._layer = i
        del blk.experts          # drop while still a lazy node, before any eval

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model, load_tokenizer(Path(model_dir)), _POOL


def main():
    prompt = os.environ.get("PROMPT", "Explain what a mixture-of-experts model is.")
    n = int(os.environ.get("TOKENS", "24"))
    model, tok, pool = load_engine()
    print(f"gpt-oss-120b: {pool.n_slots} slots x {pool.blob/1e6:.2f} MB = "
          f"{pool.pool_bytes/2**30:.2f} GiB pool, "
          f"resident {mx.get_active_memory()/2**30:.2f} GiB")
    print(f"  {pool.n_slots}/{pool.n_layers*pool.n_experts} experts "
          f"({pool.n_slots/(pool.n_layers*pool.n_experts)*100:.1f}% of model)")

    ids = tok.encode(prompt)
    c = make_prompt_cache(model)
    t0 = time.perf_counter()
    y = mx.argmax(model(mx.array(ids)[None], cache=c)[:, -1], axis=-1)
    mx.eval(y)
    print(f"  prefill {len(ids)} tok in {time.perf_counter()-t0:.1f}s")

    out, times = [], []
    for _ in range(n):
        t = time.perf_counter()
        y = mx.argmax(model(y[None], cache=c)[:, -1], axis=-1)
        mx.eval(y)
        times.append(time.perf_counter() - t)
        out.append(y.item())
    steady = sorted(times[2:]) or times
    s = pool.stats()
    print(f"\n  text: {tok.decode(out)!r}")
    print(f"  {len(steady)/sum(steady):.2f} tok/s median "
          f"{steady[len(steady)//2]*1e3:.0f} ms/token")
    print(f"  hit: prefetch {s['prefetch']/s['total']*100:.0f}% "
          f"cache {s['cache']/s['total']*100:.0f}% miss {s['miss']/s['total']*100:.0f}%")
    print(f"  read {s['gb']/max(len(times),1)*1024:.0f} MB/token, "
          f"resident {mx.get_active_memory()/2**30:.2f} GiB")
    if _RECALL:
        import statistics
        print(f"  predict recall {statistics.mean(_RECALL)*100:.1f}%")


if __name__ == "__main__":
    main()
