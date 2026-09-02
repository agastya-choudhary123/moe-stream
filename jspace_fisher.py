#!/usr/bin/env python3
"""
Is the expert block's output error measured in the wrong space?

Every negative result in the activation-subspace study scored error in
activation space: how well `y` is reproduced. That set the bar at ~99.5%
retention. But an error only costs anything if it changes the model's output,
and the J-space result (Anthropic, July 2026) claims the causally active part of
the residual stream is small -- ~25 verbalizable concepts, under 10% of
activation variance -- found by looking at the input-output Jacobian rather than
at variance. If that holds here, most of the `y` error a projection makes could
be causally inert, and the bar was computed in the wrong metric.

The right metric is the Fisher in the block's output space,

    F_L = E[ g g^T ],   g = d NLL / d y_L

which says how much the model's predictions move when the layer-L expert output
moves. Two things follow:

  1. F_L's spectrum is a direct test of the J-space claim in this setting. If a
     few dozen directions carry the Fisher energy, there is enormous headroom
     that variance-weighted PCA was throwing away.
  2. The subspace floor generalizes by swapping G = sum_e W_e^T W_e for
     G = sum_e W_e^T F_L W_e, which is then the best possible rank-k basis
     measured in output-relevant units instead of activation units.

Getting the gradient needs two accommodations. Routing is replayed from a clean
forward pass, because `np.array(inds)` cannot run on a traced value and because
argpartition's derivative is zero almost everywhere anyway; the softmax scores
stay differentiable. And the sequence is short enough that every expert the pass
touches stays resident, since a backward pass reads the weights again and the
pool would otherwise have recycled the slots underneath it -- the same hazard
that corrupted the model before, which is why `needs_barrier` is forced on.

One backward pass yields a gradient for every layer at once.

  PF_SLOTS=600 NPASS=150 python3 jspace_fisher.py
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

ACTS = os.path.expanduser("~/Desktop/moe-stream/acts")
FISHER_LAYERS = [int(s) for s in
                 os.environ.get("FLAYERS", "0,9,18,27,31,35").split(",")]
T = int(os.environ.get("SEQ", "4"))          # tokens per pass; see docstring
_ROUTE = {}                                   # layer -> recorded expert ids
_DELTA = {}                                   # layer -> traced perturbation
_RECORD = True


def recording_call(self, x):
    """Normal engine call, but remember which experts fired."""
    y = engine_120b.streaming_call(self, x)
    return y


def replay_call(self, x):
    """Differentiable expert block with routing pinned to the recorded choice.

    The perturbation goes on the EXPERT PATH INPUT, not on the block output and
    not on x itself, because that is exactly where the compression would act:
    the router is a non-expert weight, stays resident and always sees the full
    x, so a projection never touches it. Measuring the Fisher anywhere else
    prices a scheme nobody is proposing.
    """
    pool, layer = engine_120b._POOL, self._layer
    inds = _ROUTE[layer]                      # mx.array of recorded ids, constant
    g = self.router(x)
    scores = mx.softmax(mx.take_along_axis(g, inds, axis=-1), axis=-1,
                        precise=True)
    flat = np.array(inds, copy=False).reshape(-1)
    uniq, inverse = np.unique(flat, return_inverse=True)
    slots = pool.acquire(layer, uniq)
    sl = np.asarray(slots, dtype=np.uint32)[inverse].reshape(inds.shape)
    idx = mx.array(sl)

    d = _DELTA.get(layer)
    xe = mx.expand_dims(x if d is None else x + d, (-2, -3))
    v = pool.views
    common = dict(rhs_indices=idx, transpose=True,
                  group_size=pool.group_size, bits=pool.bits)

    def proj(name, inp):
        y = mx.gather_qmm(inp, v[f"{name}.weight"], scales=v[f"{name}.scales"],
                          biases=v[f"{name}.biases"], **common)
        return y + mx.expand_dims(v[f"{name}.bias"][idx], -2)

    h = gpt_oss.swiglu(proj("up_proj", xe), proj("gate_proj", xe))
    y = proj("down_proj", h).squeeze(-2)
    return (y * scores[..., None]).sum(axis=-2)


def main():
    npass = int(os.environ.get("NPASS", "150"))
    model, tok, pool = engine_120b.load_engine()
    pool.needs_barrier = True                 # backward re-reads the slots
    n_layers = len(engine_120b._BLOCKS)

    man = json.load(open(f"{ACTS}/manifest.json"))
    import corpus
    from pathlib import Path
    windows = corpus.build(tok, per_genre=max(w["window"]
                                              for w in man["windows"]) + 1)
    rng = np.random.default_rng(0)

    F = {L: np.zeros((2880, 2880), dtype=np.float64) for L in FISHER_LAYERS}
    Fhalf = {L: np.zeros((2880, 2880), dtype=np.float64) for L in FISHER_LAYERS}
    n_seen = 0
    t0 = time.perf_counter()

    for p in range(npass):
        g_, wi, ids = windows[rng.integers(len(windows))]
        s = int(rng.integers(0, len(ids) - T - 1))
        chunk = mx.array(ids[s:s + T + 1])[None]

        # 1. one unperturbed pass to pin the routing indices
        _ROUTE.clear()
        _DELTA.clear()
        gpt_oss.MLPBlock.__call__ = _capture_route
        _ROUTE.clear()
        cache = make_prompt_cache(model)
        out = model(chunk[:, :T], cache=cache)
        mx.eval(out, *_ROUTE.values())

        # 2. differentiable replay, routing now constant
        gpt_oss.MLPBlock.__call__ = replay_call

        def loss(deltas):
            for L in range(n_layers):
                _DELTA[L] = deltas[L]
            c = make_prompt_cache(model)
            lg = model(chunk[:, :T], cache=c)[0].astype(mx.float32)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            tgt = chunk[0, 1:T + 1]
            return -mx.take_along_axis(lp, tgt[:, None], axis=-1).sum()

        zeros = [mx.zeros((1, T, 2880), dtype=mx.bfloat16)
                 for _ in range(n_layers)]
        grads = mx.grad(loss)(zeros)
        mx.eval(*grads)
        for L in FISHER_LAYERS:
            G = np.array(grads[L].astype(mx.float32)).reshape(T, 2880)
            F[L] += G.T.astype(np.float64) @ G.astype(np.float64)
            if p % 2 == 0:
                Fhalf[L] += G.T.astype(np.float64) @ G.astype(np.float64)
        n_seen += T
        if (p + 1) % 10 == 0:
            print(f"  pass {p+1}/{npass}  {n_seen} samples  "
                  f"{(time.perf_counter()-t0)/(p+1):.1f} s/pass", flush=True)

    out = {}
    for L in FISHER_LAYERS:
        w = np.linalg.eigvalsh(F[L])[::-1]
        w = np.clip(w, 0, None)
        pr = w.sum() ** 2 / (w ** 2).sum()
        cum = np.cumsum(w) / w.sum()
        # held-out: basis from the even passes, energy measured on the odd ones
        Fo = F[L] - Fhalf[L]
        wv, V = np.linalg.eigh(Fhalf[L])
        V = V[:, ::-1]
        E = np.einsum("ij,jk,ki->i", V.T, Fo, V)
        cum_ho = np.cumsum(E) / np.trace(Fo)
        out[L] = dict(participation_ratio=float(pr), n=n_seen,
                      k_at={t: int(np.searchsorted(cum, t) + 1)
                            for t in (0.5, 0.9, 0.99)},
                      k_at_heldout={t: int(np.searchsorted(cum_ho, t) + 1)
                                    for t in (0.5, 0.9, 0.99)},
                      ret={k: float(cum[k - 1]) for k in (25, 100, 320, 960)},
                      ret_heldout={k: float(cum_ho[k - 1])
                                   for k in (25, 100, 320, 960)})
        r = out[L]
        print(f"L{L:02d}  participation ratio {pr:7.1f} of 2880   "
              f"k@90% {r['k_at'][0.9]:>5} (held out {r['k_at_heldout'][0.9]:>5})"
              f"   Fisher energy in top 25 dims: {r['ret'][25]*100:.1f}% "
              f"(held out {r['ret_heldout'][25]*100:.1f}%)", flush=True)
        # both halves are saved: pricing a basis with the same Fisher that
        # chose it is exactly the rank artifact this project keeps tripping over
        np.save(f"{ACTS}/fisher_L{L:02d}.npy", F[L].astype(np.float32))
        np.save(f"{ACTS}/fisher_even_L{L:02d}.npy", Fhalf[L].astype(np.float32))
        np.save(f"{ACTS}/fisher_odd_L{L:02d}.npy",
                (F[L] - Fhalf[L]).astype(np.float32))
    json.dump({str(k): v for k, v in out.items()},
              open(f"{ACTS}/jspace_fisher.json", "w"), indent=1)


def _capture_route(self, x):
    """One replay pass with no perturbation, to pin the routing indices."""
    k = self.num_experts_per_tok
    g = self.router(x)
    inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
    mx.eval(inds)
    _ROUTE[self._layer] = inds
    return replay_call(self, x)


if __name__ == "__main__":
    main()
