#!/usr/bin/env python3
"""Prefetch-aware simulator matching engine_v3.ExpertPool.

The demand-paging simulator asked the wrong question. In this engine a
"prefetch hit" STILL performed a full read, so bytes/token is proportional to
the number of pool ADMISSIONS, not to the demand hit rate. Every admission --
speculative or not, right or wrong -- is 14.02 MB off the SSD.

Loop mirrors the engine: at layer L, acquire L's true experts, then prefetch
layer L+1's predicted experts (PF_DEPTH=1).
"""
import sys
from collections import defaultdict

import numpy as np

EPL, BLOB = 128, 14.02
SAMPLE, STRIDE = 24, 7919


def simulate(true, pred, cap, mode, window=24, stamp_on_insert=False):
    """true/pred: [L, T, 4]. Returns (admissions_per_token, cache_hit_rate)."""
    L, T, K = true.shape
    key_of = [None] * cap
    slot_of, freq, born, last_use = {}, defaultdict(int), {}, {}
    free = list(range(cap))
    clock = use_clock = 0
    admits = hits = accesses = 0
    pinned = set()

    def alloc(key):
        nonlocal clock, admits, use_clock
        clock += 1
        admits += 1
        if free:
            slot = free.pop()
        else:
            best = victim = None
            start = clock % cap
            seen = 0
            for j in range(cap):
                sidx = (start + j * STRIDE) % cap
                k = key_of[sidx]
                if k is None or sidx in pinned:
                    continue
                young = clock - born.get(k, 0) < window
                if mode == "asis":
                    base = 0
                elif mode == "lru":
                    base = last_use.get(k, 0)
                else:
                    base = freq.get(k, 0)
                f = base + ((1 << 30) if young else 0)
                if best is None or f < best:
                    best, victim = f, sidx
                seen += 1
                if seen >= SAMPLE and best is not None and best < (1 << 30):
                    break
            if victim is None:
                victim = next(i for i in range(cap) if i not in pinned)
            vk = key_of[victim]
            del slot_of[vk]
            born.pop(vk, None); last_use.pop(vk, None); freq.pop(vk, None)
            slot = victim
        key_of[slot] = key; slot_of[key] = slot; born[key] = clock
        if stamp_on_insert:
            use_clock += 1
            last_use[key] = use_clock
        return slot

    def touch(key):
        nonlocal use_clock
        use_clock += 1
        last_use[key] = use_clock
        freq[key] += 1

    for t in range(T):
        for l in range(L):
            pinned = set()
            for e in true[l, t]:
                key = l * EPL + int(e)
                accesses += 1
                if key in slot_of:
                    hits += 1
                else:
                    alloc(key)
                touch(key)
                pinned.add(slot_of[key])
            if l + 1 < L:
                for e in pred[l + 1, t]:
                    key = (l + 1) * EPL + int(e)
                    if key not in slot_of:
                        alloc(key)
    return admits / T, hits / accesses


def main():
    ntok = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    base = ("/private/tmp/claude-501/-Users-agastya-Desktop-moe-stream/"
            "32819ea4-feb4-45b0-94d1-5314981e9000/scratchpad")
    true = np.array(np.load("acts/routes.npy")[:, :ntok, :])
    pred = np.load(f"{base}/pred_6144.npy")[:, :ntok, :]
    L, T, K = true.shape
    print(f"{T} tokens, {L} layers, {cap} slots, prefetch depth 1")
    print("engine measured: 1451-1519 MB/token, cache hit 43-48%\n")
    print(f"  {'policy':<26} {'MB/token':>9} {'cache hit':>10} {'vs asis':>8}")
    ref = None
    for name, mode, stamp in [
            ("asis (shipped)", "asis", False),
            ("lfu", "lfu", False),
            ("lru, stamp on insert", "lru", True),
            ("lru, stamp on use only", "lru", False)]:
        a, h = simulate(true, pred, cap, mode, stamp_on_insert=stamp)
        mb = a * BLOB
        ref = ref or mb
        print(f"  {name:<26} {mb:9.0f} {100*h:9.1f}% {ref/mb:7.2f}x")


if __name__ == "__main__":
    main()
