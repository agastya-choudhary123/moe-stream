"""How skewed are gpt-oss-120b's top-4 gate weights, and does residency correlate?

The streaming engine spends an identical 14 MB on every routed expert, but the
router hands over a gate weight first -- i.e. the VALUE of those bytes is known
before the cost is paid. Contribution-proportional fidelity (full 4-bit for
dominant experts, an MSB-only half-read for negligible ones) is only worth
building if the weights are actually skewed.

HANDOFF's counter-evidence: static top-3 truncation was destructive, because
the load-balancing loss "flattened the router, so there is no cheap expert to
drop". That is an argument about the MEAN. What matters for a threshold policy
is the TAIL: how often the smallest of the four is negligible.

Records, per (token, layer): the 4 softmax gate weights, and whether each
routed expert was resident at acquire time.
"""
import os, sys, time, json
import numpy as np

os.environ.setdefault("PF_SLOTS", "600")
os.environ.setdefault("PF_WORKERS", "8")
import mlx.core as mx
import engine_120b as E
from engine_v3 import PREFETCH_DEPTHS
from mlx_lm.models.cache import make_prompt_cache

PROMPT = "Explain why mixture-of-experts models are hard to run on small machines."

REC = []          # sorted gate weights desc, one row per (token, layer)
RESIDENT = []     # (weight, was_resident) for every routed expert

_LAST = {}        # expert id -> gate weight, from the most recent route call


def wrap(pool):
    """Record without touching the expert math: wrap the two calls that already
    carry the information. top_k_route has the weights; acquire knows what was
    resident. streaming_call runs them in order on one thread per layer, so
    pairing them by "most recent" is exact."""
    orig_route, orig_acquire = E.top_k_route, pool.acquire

    def route(router, x, k):
        inds, scores = orig_route(router, x, k)
        mx.eval(inds, scores)
        w = np.array(scores.astype(mx.float32)).reshape(-1)
        e = np.array(inds.astype(mx.int32)).reshape(-1)
        REC.append(np.sort(w)[::-1].copy())
        _LAST.clear()
        _LAST.update({int(i): float(v) for i, v in zip(e, w)})
        return inds, scores

    def acquire(layer, experts):
        with pool.lock:
            for ex in experts:
                ex = int(ex)
                if ex in _LAST:
                    RESIDENT.append((_LAST[ex], (layer, ex) in pool.slot_of))
        return orig_acquire(layer, experts)

    E.top_k_route = route
    pool.acquire = acquire


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    model, tok, pool = E.load_engine()
    ids = tok.encode(PROMPT)
    cache = make_prompt_cache(model)
    y = mx.argmax(model(mx.array(ids)[None], cache=cache)[:, -1], axis=-1)
    mx.eval(y)
    for _ in range(8):                       # warm the pool
        y = mx.argmax(model(y[None], cache=cache)[:, -1], axis=-1)
        mx.eval(y)

    wrap(pool)
    REC.clear(); RESIDENT.clear()
    for _ in range(n):
        y = mx.argmax(model(y[None], cache=cache)[:, -1], axis=-1)
        mx.eval(y)

    W = np.stack(REC)                        # [n_obs, 4] sorted desc
    print(f"\n{W.shape[0]} (token, layer) routing decisions, top-4 gate weights\n")
    print("rank      mean    median     p10     p90")
    for i in range(W.shape[1]):
        c = W[:, i]
        print(f"  w{i+1}   {c.mean():7.4f} {np.median(c):8.4f} "
              f"{np.percentile(c,10):7.4f} {np.percentile(c,90):7.4f}")

    smallest = W[:, -1]
    print(f"\nsmallest of the four:")
    for t in (0.02, 0.05, 0.08, 0.10, 0.15):
        print(f"  P(w4 < {t:.2f}) = {(smallest < t).mean()*100:5.1f}%")

    # bytes recoverable if every expert under threshold is read at HALF size
    print(f"\nbytes saved by half-reading every expert under a threshold:")
    allw = W.reshape(-1)
    for t in (0.05, 0.08, 0.10, 0.15, 0.20):
        frac = (allw < t).mean()
        print(f"  tau={t:.2f}: {frac*100:5.1f}% of routed experts "
              f"-> {frac*50:5.1f}% fewer expert bytes")

    # does low weight coincide with being a cache MISS? if so the policy is
    # even cheaper -- skip precisely the reads that would have stalled
    R = np.array([(w, res) for w, res in RESIDENT])
    miss = R[R[:, 1] == 0][:, 0]
    hit = R[R[:, 1] == 1][:, 0]
    print(f"\nmean gate weight | resident {hit.mean():.4f}  (n={len(hit)})")
    print(f"mean gate weight | MISS     {miss.mean():.4f}  (n={len(miss)})")

    json.dump({"W": W.tolist()}, open("gate_skew.json", "w"))
    print("\nwrote gate_skew.json")


if __name__ == "__main__":
    main()
