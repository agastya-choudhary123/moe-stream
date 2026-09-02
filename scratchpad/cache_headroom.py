#!/usr/bin/env python3
"""How much of the 120b's miss rate is policy, and how much is capacity?

Replays the real routing trace (acts/routes.npy) through LRU / LFU / Belady-OPT
at several slot counts. OPT is the offline optimum -- no online policy can beat
it -- so the LFU-to-OPT gap is the entire headroom available to a smarter
eviction policy, with zero quality cost.
"""
import heapq
import json
import sys

import numpy as np

E_PER_LAYER = 128
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 6144


def trace(ntok):
    r = np.array(np.load("acts/routes.npy")[:, :ntok, :])      # [L, T, 4]
    L, T, K = r.shape
    gid = (np.arange(L)[:, None, None] * E_PER_LAYER + r.astype(np.int64))
    return np.ascontiguousarray(gid.transpose(1, 0, 2).reshape(-1)), L, T, K


def sim_lru(seq, cap):
    from collections import OrderedDict
    d, hits = OrderedDict(), 0
    for x in seq:
        if x in d:
            d.move_to_end(x); hits += 1
        else:
            if len(d) >= cap:
                d.popitem(last=False)
            d[x] = 1
    return hits / len(seq)


def sim_lfu(seq, cap, protect=144):
    """LFU with the engine's young-entry protection window."""
    freq, born, res = {}, {}, set()
    hits, clock = 0, 0
    for x in seq:
        clock += 1
        if x in res:
            hits += 1
            freq[x] = freq.get(x, 0) + 1
        else:
            if len(res) >= cap:
                cand = [k for k in res if clock - born[k] > protect] or list(res)
                v = min(cand, key=lambda k: (freq.get(k, 0), born[k]))
                res.discard(v); freq.pop(v, None); born.pop(v, None)
            res.add(x); freq[x] = 1; born[x] = clock
    return hits / len(seq)


def sim_opt(seq, cap):
    """Belady: evict the resident entry whose next use is furthest away."""
    n = len(seq)
    nxt = np.full(n, n, dtype=np.int64)
    last = {}
    for i in range(n - 1, -1, -1):
        x = seq[i]
        nxt[i] = last.get(x, n)
        last[x] = i
    res, heap, hits = set(), [], 0
    for i, x in enumerate(seq):
        if x in res:
            hits += 1
        else:
            if len(res) >= cap:
                while True:
                    negt, cand = heapq.heappop(heap)
                    if cand in res and -negt >= i:
                        break
                res.discard(cand)
            res.add(x)
        heapq.heappush(heap, (-int(nxt[i]), int(x)))
    return hits / n


def main():
    seq, L, T, K = trace(NTOK)
    blob = 14.02
    print(f"trace: {T} tokens x {L} layers x top-{K} = {len(seq):,} accesses, "
          f"{L*E_PER_LAYER} experts total")
    print(f"\n  {'slots':>6} {'%model':>7} {'LRU':>8} {'LFU':>8} {'OPT':>8} "
          f"{'LFU MB/tok':>11} {'OPT MB/tok':>11} {'headroom':>9}")
    for cap in (300, 600, 700, 1000, 1500):
        lru, lfu, opt = sim_lru(seq, cap), sim_lfu(seq, cap), sim_opt(seq, cap)
        per = L * K * blob
        print(f"  {cap:6d} {100*cap/(L*E_PER_LAYER):6.1f}% {100*lru:7.1f}% "
              f"{100*lfu:7.1f}% {100*opt:7.1f}% {per*(1-lfu):11.0f} "
              f"{per*(1-opt):11.0f} {(1-lfu)/(1-opt):8.2f}x")


if __name__ == "__main__":
    main()
