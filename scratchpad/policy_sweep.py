#!/usr/bin/env python3
"""Which online eviction policy closes the LFU->OPT gap on the 120b trace?

Includes a cyclic-aware policy: access sweeps layer 0..35 every token, so an
expert belonging to the layer just finished is ~36 layers from its next possible
use, while one belonging to an upcoming layer may be needed within milliseconds.
Belady's distance is therefore partly KNOWN a priori from the layer index --
no prediction required.
"""
import sys
from collections import OrderedDict

import numpy as np

EPL, L_N = 128, 36
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 600


def trace(ntok):
    r = np.array(np.load("acts/routes.npy")[:, :ntok, :])
    L, T, K = r.shape
    gid = (np.arange(L)[:, None, None] * EPL + r.astype(np.int64))
    return np.ascontiguousarray(gid.transpose(1, 0, 2).reshape(-1)), L, T, K


def run(seq, cap, policy, K=4):
    """policy: 'lru' | 'lfu' | 'cyclic' | 'cyclic_lru'"""
    hits = 0
    if policy == "lru":
        d = OrderedDict()
        for x in seq:
            if x in d:
                d.move_to_end(x); hits += 1
            else:
                if len(d) >= cap:
                    d.popitem(last=False)
                d[x] = 1
        return hits / len(seq)

    freq, born, recency = {}, {}, {}
    res = set()
    for i, x in enumerate(seq):
        cur_layer = (i // K) % L_N
        if x in res:
            hits += 1
            freq[x] = freq.get(x, 0) + 1
            recency[x] = i
        else:
            if len(res) >= cap:
                if policy == "lfu":
                    v = min(res, key=lambda k: (freq.get(k, 0), born[k]))
                elif policy == "cyclic":
                    # furthest ahead in the sweep first, then coldest
                    v = max(res, key=lambda k: (
                        (k // EPL - cur_layer) % L_N, -freq.get(k, 0)))
                else:   # cyclic_lru: sweep distance, then least recent
                    v = max(res, key=lambda k: (
                        (k // EPL - cur_layer) % L_N, -recency.get(k, 0)))
                res.discard(v)
                freq.pop(v, None); born.pop(v, None); recency.pop(v, None)
            res.add(x); freq[x] = 1; born[x] = i; recency[x] = i
    return hits / len(seq)


def main():
    seq, L, T, K = trace(NTOK)
    per = L * K * 14.02
    print(f"{T} tokens, {len(seq):,} accesses, {CAP} slots "
          f"({100*CAP/(L*EPL):.1f}% of model)\n")
    print(f"  {'policy':<12} {'hit':>7} {'MB/token':>9} {'vs LFU':>8}")
    base = None
    for p in ("lfu", "lru", "cyclic", "cyclic_lru"):
        h = run(seq, CAP, p, K)
        mb = per * (1 - h)
        base = base or mb
        print(f"  {p:<12} {100*h:6.1f}% {mb:9.0f} {base/mb:7.2f}x")


if __name__ == "__main__":
    main()
